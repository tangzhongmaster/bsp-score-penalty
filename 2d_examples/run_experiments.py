"""
Run the CSBM / USBM / QCSBM / BSP comparison across the three toy datasets in one call and collect
Asym, NAsym, score error, NLL, NFE, precision and recall in a single table (summary.csv).

Usage (from 2d_examples/):
    python run_experiments.py
    python run_experiments.py --methods usbm qcsbm bsp --datasets checkerboard spiral_2
    python run_experiments.py --results_root results_dir

Training and evaluation call the functions used by run_lib.py and run_lib_eval.py. Runs are independent:
if one fails, the others still complete and the failure is reported as a row of NaNs with the error message.
"""
import argparse
import importlib
import os
import time
import traceback

import torch
import numpy as np
import pandas as pd

import datasets
import losses
import likelihood
import sampling_fn
import sde_lib
from models import simple_score_fn
from utils import get_oracle_score_pointwise, calculate_score_error, calculate_asymmetricity
from prdc import compute_prdc

# method name -> (config module path, human label)
METHOD_CONFIGS = {
    'csbm':  'configs.csbm.{dataset}_config',
    'usbm':  'configs.usbm.{dataset}_config',
    'qcsbm': 'configs.qcsbm.{dataset}_config',  # reg_type='lqc' (exact)
    'bsp':   'configs.bsp.{dataset}_config',    # ours, VJP implementation
    'bsp_jvp': 'configs.bsp_jvp.{dataset}_config',  # ours, JVP (double-backward) implementation -- for equivalence checking
}
DATASETS = ['checkerboard', 'gaussian_8', 'spiral_2']


def load_config(method, dataset, reg_noise_std=None, bsp_std_weighting=None):
    module_path = METHOD_CONFIGS[method].format(dataset=dataset)
    module = importlib.import_module(module_path)
    config = module.get_config()
    config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    if reg_noise_std is not None:
        config.loss.reg_noise_std = reg_noise_std
    if bsp_std_weighting is not None:
        config.loss.bsp_std_weighting = bsp_std_weighting
    return config


def train(config, workdir):
    os.makedirs(workdir, exist_ok=True)
    checkpoint_dir = os.path.join(workdir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    score_model = simple_score_fn.simple_noise_conditioned_score_fn(config).to(config.device, dtype=config.data.dtype)
    train_step_fn = losses.get_step_fn(
        reg_type=config.loss.reg_type, std_value_max=config.loss.std_max,
        std_value_min=config.loss.std_min, balancing_fac=config.loss.balancing_fac,
        slices=config.loss.slices, dtype=config.data.dtype,
        reg_noise_std=config.loss.get('reg_noise_std', 0.0),
        bsp_std_weighting=config.loss.get('bsp_std_weighting', False),
    )
    optimizer = torch.optim.Adam(
        score_model.parameters(), lr=config.optim.lr, betas=(config.optim.beta1, 0.999),
        eps=config.optim.eps, weight_decay=config.optim.weight_decay,
    )

    ds = datasets.get_dataset(config)
    iter_ds = iter(ds)
    is_score_model = (config.model.type == 'score_model')

    t0 = time.time()
    if config.device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()

    for step in range(config.training.n_iters + 1):
        data = next(iter_ds)
        batch = data['position'].clone().to(config.device, dtype=config.data.dtype)
        train_step_fn(score_model, optimizer, batch, is_score_model)

    elapsed = time.time() - t0
    peak_mem = torch.cuda.max_memory_allocated() / 1024**2 if config.device.type == 'cuda' else float('nan')

    ckpt_path = os.path.join(checkpoint_dir, 'checkpoint_{}.pth'.format(config.training.n_iters))
    torch.save({'model': score_model.state_dict()}, ckpt_path)
    return score_model, elapsed, peak_mem


def evaluate(config, score_model, eval_points_n=5000):
    is_score_model = (config.model.type == 'score_model')

    # -- Score error / Asym / NAsym --
    # NOTE: get_oracle_score_pointwise is O(eval_points_n * n_scale * dataset_size)
    # (an unvectorized Python double-loop in utils.py, faithfully reproduced from
    # the original repo's evaluation protocol) -- this is the slowest part of
    # evaluation. Lower --eval_points for faster iteration at the cost of a
    # noisier score-error/Asym estimate; the paper's own protocol uses 5000.
    config.training.batch_size, bs_backup = eval_points_n, config.training.batch_size
    eval_points = next(iter(datasets.get_dataset(config)))['position'].to(config.device, dtype=config.data.dtype)
    score_oracle, _, noise_points = get_oracle_score_pointwise(config, points=eval_points, ds=config.data.dataset)
    noise_points_t = torch.tensor(noise_points).to(config.device, dtype=config.data.dtype)

    score_error = calculate_score_error(noise_points_t, score_oracle, score_model,
                                         is_score_model=is_score_model, n_scale=config.model.n_scales)
    nasym, asym = calculate_asymmetricity(noise_points_t, score_model,
                                           is_score_model=is_score_model, n_scale=config.model.n_scales)

    # -- NLL --
    sde = sde_lib.VESDE(sigma_min=config.loss.std_min, sigma_max=config.loss.std_max)
    likelihood_fn = likelihood.get_likelihood_fn(sde=sde, rtol=1e-3, atol=1e-5, is_score_model=is_score_model)
    eval_batch = next(iter(datasets.get_dataset(config)))['position'].to(config.device, dtype=config.data.dtype)
    nll, _, nfe = likelihood_fn(score_model, eval_batch)

    # -- Sampling: precision / recall --
    shape = (config.training.batch_size, 2)
    sampler = sampling_fn.get_ode_sampler(
        sde, shape, rtol=1e-3, atol=1e-5, method='RK45', eps=1e-5,
        device=str(config.device), is_score_model=is_score_model,
    )
    samples, _ = sampler(score_model)
    samples = samples.cpu().detach().numpy()
    real = eval_batch.cpu().detach().numpy()
    prdc = compute_prdc(real_features=real, fake_features=samples, nearest_k=config.eval.nearest_k)

    config.training.batch_size = bs_backup
    return {
        'asym': asym, 'nasym': nasym, 'score_error': score_error,
        'nll': nll.mean().item(), 'nfe': nfe,
        'precision': prdc['precision'], 'recall': prdc['recall'],
    }


def run_one(method, dataset, results_root, eval_points_n=5000, seed=0, reg_noise_std=None, bsp_std_weighting=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    config = load_config(method, dataset, reg_noise_std=reg_noise_std, bsp_std_weighting=bsp_std_weighting)
    tag = f"{dataset}_{method}_seed{seed}" + (f"_rns{reg_noise_std}" if reg_noise_std else "") \
        + ("_noweight" if bsp_std_weighting is False else "")
    workdir = os.path.join(results_root, tag)
    row = {'method': method, 'dataset': dataset, 'seed': seed, 'reg_noise_std': reg_noise_std or 0.0,
           'bsp_std_weighting': False if bsp_std_weighting is None else bsp_std_weighting}
    try:
        score_model, elapsed, peak_mem = train(config, workdir)
        row['train_time_s'] = elapsed
        row['peak_mem_MB'] = peak_mem
        metrics = evaluate(config, score_model, eval_points_n=eval_points_n)
        row.update(metrics)
        row['status'] = 'ok'
    except Exception as e:
        row['status'] = f'FAILED: {e}'
        print(f"[{dataset} | {method} | seed={seed}] FAILED:")
        traceback.print_exc()
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--methods', nargs='+', default=['usbm', 'qcsbm', 'bsp'], choices=list(METHOD_CONFIGS))
    p.add_argument('--datasets', nargs='+', default=DATASETS, choices=DATASETS)
    p.add_argument('--results_root', default='results')
    p.add_argument('--csv_out', default='summary.csv')
    p.add_argument('--eval_points', type=int, default=5000,
                    help='Number of points used for the score error and Asym comparison with the oracle score.')
    p.add_argument('--seeds', type=int, nargs='+', default=[0],
                    help='One run per seed for each (method, dataset).')
    p.add_argument('--reg_noise_std', type=float, default=None,
                    help="Only affects 'bsp': evaluate the Jacobian of the regularizer at the perturbed input "
                         "plus N(0, reg_noise_std^2) noise. Ignored for other methods.")
    p.add_argument('--bsp_std_weighting', action='store_true',
                    help="Only affects 'bsp' and 'bsp_jvp': multiply the penalty by std**2, "
                         "as in lqc and lqc_est. Off by default.")
    args = p.parse_args()

    rows = []
    for dataset in args.datasets:
        for method in args.methods:
            for seed in args.seeds:
                print(f"\n=== {dataset} | {method} | seed={seed} ===")
                row = run_one(method, dataset, args.results_root, eval_points_n=args.eval_points,
                               seed=seed, reg_noise_std=args.reg_noise_std,
                               bsp_std_weighting=(True if args.bsp_std_weighting else None))
                rows.append(row)
                df_partial = pd.DataFrame(rows)
                df_partial.to_csv(os.path.join(args.results_root, args.csv_out), index=False)
                print(row)

    df = pd.DataFrame(rows)
    os.makedirs(args.results_root, exist_ok=True)
    df.to_csv(os.path.join(args.results_root, args.csv_out), index=False)
    print("\n" + "=" * 80)
    print("Per-run results:")
    print(df.to_string(index=False))
    print(f"\nSaved to {os.path.join(args.results_root, args.csv_out)}")

    if len(args.seeds) > 1:
        numeric_cols = ['asym', 'nasym', 'score_error', 'nll', 'nfe', 'precision', 'recall',
                         'train_time_s', 'peak_mem_MB']
        ok = df[df['status'] == 'ok']
        agg = ok.groupby(['dataset', 'method'])[numeric_cols].agg(['mean', 'std'])
        agg.columns = ['_'.join(c) for c in agg.columns]
        agg = agg.reset_index()
        summary_path = os.path.join(args.results_root, 'summary_mean_std.csv')
        agg.to_csv(summary_path, index=False)
        print("\n" + "=" * 80)
        print(f"Mean +/- std over {len(args.seeds)} seeds:")
        print(agg.to_string(index=False))
        print(f"\nSaved to {summary_path}")
    else:
        print("\nOnly 1 seed run -- pass --seeds 0 1 2 (etc.) to get mean/std across seeds "
              "before trusting these numbers for the paper.")


if __name__ == "__main__":
    main()
