"""
Check that the two-VJP implementation of the BSP regularizer in losses.py gives the same value as a
reference implementation based on the double-backward Jacobian-vector product, on identical inputs and
probe vectors, and compare their peak GPU memory and step time.

Run from inside 2d_examples/:

    python verify_bsp.py
"""
import torch
from configs.bsp.checkerboard_config import get_config
from models import simple_score_fn


def reference_bsp_reg(score_model, perturbed_batch, t, x_p, y_p):
    """Jacobian-vector product via the double-backward trick (reference,
    slower implementation -- kept here only to check against)."""
    current_score = score_model(perturbed_batch, t)

    def get_jvp(y, x, v):
        v_base = torch.ones_like(y).requires_grad_(True)
        grads = torch.autograd.grad(y, x, grad_outputs=v_base, create_graph=True, retain_graph=True)[0]
        jvp = torch.autograd.grad(grads, v_base, grad_outputs=v, create_graph=True, retain_graph=True)[0]
        return jvp

    Jx = get_jvp(current_score, perturbed_batch, x_p)
    Jy = get_jvp(current_score, perturbed_batch, y_p)
    x_Jy = torch.sum(x_p * Jy, dim=1)
    y_Jx = torch.sum(y_p * Jx, dim=1)
    return torch.square(x_Jy - y_Jx)


def efficient_bsp_reg(score_model, perturbed_batch, t, u, v):
    """The implementation now in losses.py's 'bsp' branch (2 VJPs)."""
    score = score_model(perturbed_batch, t)
    Jt_u, = torch.autograd.grad(score, perturbed_batch, grad_outputs=u, create_graph=True, retain_graph=True)
    u_Jv = torch.sum(Jt_u * v, dim=1)
    Jt_v, = torch.autograd.grad(score, perturbed_batch, grad_outputs=v, create_graph=True, retain_graph=True)
    Ju_v = torch.sum(Jt_v * u, dim=1)
    return torch.square(u_Jv - Ju_v)


def build_model(device):
    config = get_config()
    config.device = torch.device(device)
    model = simple_score_fn.simple_noise_conditioned_score_fn(config).to(device, dtype=config.data.dtype)
    return model, config


def check_equivalence(device="cuda" if torch.cuda.is_available() else "cpu"):
    torch.manual_seed(0)
    model, config = build_model(device)
    dtype = config.data.dtype

    batch = torch.randn(256, 2, device=device, dtype=dtype)
    t = torch.rand(256, device=device, dtype=dtype)
    perturbed_batch = batch.clone().detach().requires_grad_(True)

    u = torch.randn_like(perturbed_batch)
    v = torch.randn_like(perturbed_batch)

    reg_ref = reference_bsp_reg(model, perturbed_batch, t, u, v)
    reg_eff = efficient_bsp_reg(model, perturbed_batch, t, u, v)

    max_abs_diff = (reg_ref - reg_eff).abs().max().item()
    print(f"max |reference - efficient| = {max_abs_diff:.3e}  (should be ~0)")
    assert max_abs_diff < 1e-4, "Mismatch -- do not trust the efficient BSP branch yet."
    print("OK: efficient BSP matches the reference implementation.")


def benchmark(device="cuda" if torch.cuda.is_available() else "cpu", n_steps=200, batch_size=5000):
    if device != "cuda":
        print("Skipping benchmark (no CUDA device available; peak-memory tracking needs one).")
        return

    for name, use_reference in [("reference (4-pass JVP)", True), ("efficient (2-pass VJP)", False)]:
        model, config = build_model(device)
        dtype = config.data.dtype
        opt = torch.optim.Adam(model.parameters(), lr=7.5e-4)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()

        for _ in range(n_steps):
            batch = torch.randn(batch_size, 2, device=device, dtype=dtype)
            t = torch.rand(batch_size, device=device, dtype=dtype)
            perturbed_batch = batch.clone().detach().requires_grad_(True)
            u = torch.randn_like(perturbed_batch)
            v = torch.randn_like(perturbed_batch)

            opt.zero_grad()
            if use_reference:
                reg = reference_bsp_reg(model, perturbed_batch, t, u, v).mean()
            else:
                reg = efficient_bsp_reg(model, perturbed_batch, t, u, v).mean()
            reg.backward()
            opt.step()

        t1.record()
        torch.cuda.synchronize()
        elapsed_ms = t0.elapsed_time(t1)
        peak_mem = torch.cuda.max_memory_allocated() / 1024**2
        print(f"{name:>24}: {elapsed_ms/n_steps:.3f} ms/step, peak mem {peak_mem:.1f} MB")


if __name__ == "__main__":
    check_equivalence()
    benchmark()
