# Decoupling the Conservativeness Penalty from the Symmetric Jacobian in Score-Based Models

Code for the **Bilinear Symmetry Penalty (BSP)**, a regularizer that pushes a score network towards
a conservative vector field, that is, towards a symmetric Jacobian.

Let J be the Jacobian of the score network and B = J − Jᵀ its antisymmetric part. A conservative score
has B = 0. The quasi-conservative estimator of QCSBM penalizes ‖B‖²_F through a Hutchinson-type estimator whose
variance depends on the symmetric part of J. BSP instead estimates ‖B‖²_F with the bilinear form (uᵀBv)² of two
independent random probes u and v. This estimator has the same mean, and its variance no longer carries the
interaction between the symmetric and antisymmetric parts of the Jacobian.

The code builds on [QCSBM](https://github.com/chen-hao-chao/qcsbm) (Chao et al., ICML 2023) and
[score_sde_pytorch](https://github.com/yang-song/score_sde_pytorch) (Song et al., ICLR 2021). See `NOTICE`.

## Layout

| Path | Content |
|---|---|
| `2d_examples/` | Two-dimensional experiments (checkerboard, spiral, eight Gaussians) |
| `real_world/` | CIFAR-10 fine-tuning of a pretrained NCSN++ and evaluation |
| `*/losses.py` | The regularizers, including the BSP estimator |

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

FID and IS evaluation (`real_world/run_lib_evaluation_fidis.py`) additionally needs
`pip install tensorflow tensorflow-gan tensorflow-hub`.
The CUDA kernels in `real_world/op/` are compiled on first import, which requires `ninja` and a CUDA toolchain.

## Two-dimensional experiments

Methods are selected through the configuration file: `csbm`, `usbm`, `qcsbm` and `bsp`.
The configuration `bsp_jvp` is an equivalent implementation of BSP based on Jacobian-vector products.

```bash
cd 2d_examples

# train
python main.py --workdir checkerboard_bsp --mode train --config configs/bsp/checkerboard_config.py

# evaluate the score error, with the last saved checkpoint
python main.py --workdir checkerboard_bsp --mode eval --type score_err \
  --config configs/bsp/checkerboard_config.py \
  --restore $(ls -v results/checkerboard_bsp/checkpoints/*.pth | tail -1)
```

`--type` accepts `nll`, `score_err` and `sampling`.

To train and evaluate several methods on several datasets in one call, and collect Asym, NAsym, score error,
NLL, NFE, precision and recall in a single table (`summary.csv`):

```bash
python run_experiments.py --methods usbm qcsbm bsp --datasets checkerboard spiral_2 gaussian_8
```

`--eval_points` sets the number of points used to compare with the oracle score (5000 by default), and
`--seeds` the random seeds.

## CIFAR-10

We fine-tune the pretrained deep continuous NCSN++ of Song et al.
([`ve/cifar10_ncsnpp_deep_continuous`](https://drive.google.com/drive/folders/11s6A_xM7qiztdj8AHQWqaIAUSC3I7uX2?usp=sharing))
with the BSP regularizer.

```bash
cd real_world

# fine-tune
python main.py --config configs/bsp_ncsn/training/cifar10_ncsnpp_deep_continuous.py \
  --reg_type bsp --bs 128 --workdir bsp_cifar10 --restore checkpoint.pth

# asymmetry of the learned score (Asym / NAsym)
python run_lib_evaluation_asym.py --config configs/bsp_ncsn/evaluation/eval_cifar10_config.py \
  --workdir bsp_cifar10 --checkpoint_idx N --bs 2500 --calculate

# negative log-likelihood
python run_lib_evaluation_nll.py --config configs/bsp_ncsn/evaluation/eval_cifar10_config.py \
  --workdir bsp_cifar10 --checkpoint_idx N --bs 2500 --calculate
```

`--checkpoint_idx N` selects the checkpoint `checkpoint_N` saved in `bsp_cifar10/checkpoints`.
`--balancing_fac` sets the coefficient λ of the penalty, and `--bsp_std_weighting` multiplies it by the squared
noise level (off by default).

Sampling and the sampling metrics use `run_lib_sampling.py` and `run_lib_evaluation_fidis.py`, with the same options
as in the [QCSBM repository](https://github.com/chen-hao-chao/qcsbm/tree/master/real_world).

## License

Apache License 2.0, see `LICENSE`. Parts of the code originate from
[QCSBM](https://github.com/chen-hao-chao/qcsbm) and
[score_sde_pytorch](https://github.com/yang-song/score_sde_pytorch) (Copyright 2020 The Google Research Authors).
