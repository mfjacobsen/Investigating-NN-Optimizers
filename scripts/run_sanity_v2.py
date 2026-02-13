# Quick sanity: 20 steps, one lr, no BC, full-batch accumulation. Check lambda_A and edge_ratio.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn as nn
import json
import pandas as pd

import src.seed as seed
import src.models as models
import src.aeos as aeos

device = seed.device

def lambda_lim(eta, beta1):
    return (2 + 2 * beta1) / ((1 - beta1) * eta)

def power_iteration_top_eigenvalue(matvec, dim, iters=20, seed_val=42):
    gen = torch.Generator(device=device).manual_seed(seed_val)
    v = torch.randn(dim, device=device, generator=gen)
    v = v / v.norm()
    for _ in range(iters):
        Hv = matvec(v)
        v = Hv / (Hv.norm() + 1e-12)
    return v.dot(matvec(v)).item()

def compute_lambda_A(model, optimizer, criterion, params, X, y, bias_correction, batch_size, iters=20):
    inv_sqrt_p = aeos.get_P_inv_sqrt_from_optimizer(optimizer, params, bias_correction)
    if inv_sqrt_p is None:
        return float("nan")
    model.eval()
    dim = inv_sqrt_p.numel()
    def matvec(v):
        u = inv_sqrt_p * v
        _, _, Hu = aeos.full_batch_loss_grad_hvp(model, criterion, params, X, y, u, batch_size)
        return inv_sqrt_p * Hu
    lam = power_iteration_top_eigenvalue(matvec, dim, iters=iters)
    model.train()
    return lam

def main():
    import src.functions as fn
    # Small data for fast sanity
    X, y, X_test, y_test = fn.load_cifar_10(use_full=False)
    X, y = X.to(device), y.to(device)
    N = X.shape[0]
    batch_size = min(5000, N)
    steps = 20
    sharpness_every = 10
    lr = 1e-4
    beta1, beta2, eps = 0.9, 0.999, 1e-7
    bias_correction = False

    torch.manual_seed(seed.SEED)
    model = models.PaperFCNet(input_size=3072, num_labels=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = aeos.AdamNoBiasCorrection(model.parameters(), lr=lr, betas=(beta1, beta2), eps=eps)
    params = list(optimizer.param_groups[0]["params"])

    lim = lambda_lim(lr, beta1)
    rows = []
    for step in range(steps):
        loss_val = aeos.full_batch_grad_step(model, criterion, optimizer, X, y, params, batch_size)
        lam_A = float("nan")
        if step % sharpness_every == 0:
            lam_A = compute_lambda_A(model, optimizer, criterion, params, X, y, bias_correction, batch_size, iters=20)
        rows.append({"step": step, "train_loss": loss_val, "lambda_A": lam_A})
        print("  step %d/%d, loss=%.4f, lambda_A=%s" % (step + 1, steps, loss_val, lam_A))

    df = pd.DataFrame(rows)
    valid = df["lambda_A"].dropna()
    if len(valid) > 0:
        er = valid / lim
        print("\nlambda_lim = %.1f" % lim)
        print("edge_ratio (all): median=%.3f" % er.median())
        last_20 = valid.iloc[-max(1, len(valid)//5):]
        print("last_20%%_median = %.3f (expect ~1 for AEoS)" % (last_20/lim).median())
    out_dir = Path("output/eos/adam_PP_v2_full")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sanity").mkdir(exist_ok=True)
    df.to_csv(out_dir / "sanity" / "metrics.csv", index=False)
    with open(out_dir / "sanity" / "metadata.json", "w") as f:
        json.dump({"lr": lr, "beta1": beta1, "beta2": beta2, "eps": eps, "bias_correction": bias_correction, "steps": steps}, f, indent=2)

if __name__ == "__main__":
    main()
