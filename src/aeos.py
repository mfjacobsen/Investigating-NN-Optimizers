"""AEoS paper reproduction: Adam no bias correction, full-batch accumulation."""
from . import seed
import torch
import torch.nn as nn

device = seed.device


class AdamNoBiasCorrection:
    """Adam without bias correction. P_t uses raw v_t (exp_avg_sq)."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        self.param_groups = [{"params": list(params), "lr": lr, "betas": betas, "eps": eps}]
        self.state = {}

    def zero_grad(self):
        for p in self.param_groups[0]["params"]:
            if p.grad is not None:
                p.grad.zero_()

    def step(self):
        lr = self.param_groups[0]["lr"]
        b1, b2 = self.param_groups[0]["betas"]
        eps = self.param_groups[0]["eps"]
        for p in self.param_groups[0]["params"]:
            if p.grad is None:
                continue
            g = p.grad
            if p not in self.state:
                self.state[p] = {
                    "step": 0,
                    "exp_avg": torch.zeros_like(p, device=p.device),
                    "exp_avg_sq": torch.zeros_like(p, device=p.device),
                }
            s = self.state[p]
            s["step"] += 1
            s["exp_avg"] = b1 * s["exp_avg"] + (1 - b1) * g
            s["exp_avg_sq"] = b2 * s["exp_avg_sq"] + (1 - b2) * (g * g)
            m, v = s["exp_avg"], s["exp_avg_sq"]
            denom = torch.sqrt(v) + eps
            p.data.addcdiv_(m, denom, value=-lr)


def full_batch_grad_step(model, criterion, optimizer, X, y, params, batch_size):
    """One full-batch step: accumulate gradient over minibatches then optimizer.step()."""
    N = X.shape[0]
    optimizer.zero_grad()
    if batch_size >= N:
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
    else:
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            Xb, yb = X[start:end], y[start:end]
            out = model(Xb)
            loss = criterion(out, yb) * (end - start) / N
            loss.backward()
    optimizer.step()
    with torch.no_grad():
        out = model(X)
        full_loss = criterion(out, y).item()
    return full_loss


def full_batch_loss_grad_hvp(model, criterion, params, X, y, vec, batch_size):
    """Full-batch loss (scalar), gradient (flat), and H*v (flat) via accumulation.
    params: list of tensors (order used for flatten/HVP).
    """
    N = X.shape[0]
    n_params = sum(p.numel() for p in params)
    device = next(model.parameters()).device

    g_accum = [torch.zeros_like(p, device=p.device) for p in params]
    Hv_accum = [torch.zeros_like(p, device=p.device) for p in params]

    if batch_size >= N:
        out = model(X)
        loss = criterion(out, y)
        grad = torch.autograd.grad(loss, params, create_graph=True)
        g_flat = torch.cat([g.reshape(-1) for g in grad])
        gv = g_flat.dot(vec)
        hvp_grad = torch.autograd.grad(gv, params, retain_graph=True)
        loss_val = loss.item()
        return loss_val, g_flat, torch.cat([h.reshape(-1) for h in hvp_grad])

    loss_sum = 0.0
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        Xb, yb = X[start:end], y[start:end]
        w = (end - start) / N
        out = model(Xb)
        loss_i = criterion(out, yb) * w
        loss_sum += loss_i.item()
        grad_i = torch.autograd.grad(loss_i, params, create_graph=True)
        for j, g in enumerate(grad_i):
            g_accum[j] = g_accum[j] + g
        g_flat_i = torch.cat([g.reshape(-1) for g in grad_i])
        gv_i = g_flat_i.dot(vec)
        hvp_i = torch.autograd.grad(gv_i, params, retain_graph=False)
        for j, h in enumerate(hvp_i):
            Hv_accum[j] = Hv_accum[j] + h
    g_flat = torch.cat([g.reshape(-1) for g in g_accum])
    Hv_flat = torch.cat([h.reshape(-1) for h in Hv_accum])
    return loss_sum, g_flat, Hv_flat


def get_P_inv_sqrt_from_optimizer(optimizer, params, bias_correction=False):
    """P = diag(sqrt(v) + eps). Return inv_sqrt_p = 1/sqrt(p) in params order.
    v is raw exp_avg_sq if not bias_correction else v_hat.
    """
    if isinstance(optimizer, AdamNoBiasCorrection):
        eps = optimizer.param_groups[0]["eps"]
        parts = []
        for p in params:
            s = optimizer.state.get(p)
            if s is None or "exp_avg_sq" not in s:
                return None
            v = s["exp_avg_sq"].detach()
            p_i = torch.sqrt(v) + eps
            inv_sqrt_p = 1.0 / (torch.sqrt(p_i) + 1e-12)
            parts.append(inv_sqrt_p.reshape(-1))
        return torch.cat(parts)
    # torch.optim.Adam
    beta2 = optimizer.param_groups[0].get("betas", (0.9, 0.999))[1]
    eps = optimizer.param_groups[0].get("eps", 1e-8)
    parts = []
    for p in params:
        s = optimizer.state.get(p)
        if s is None or "exp_avg_sq" not in s:
            return None
        v = s["exp_avg_sq"].detach()
        step = s.get("step", 1) or 1
        if bias_correction:
            bc = 1.0 - beta2 ** step
            v_hat = v / bc if abs(bc) >= 1e-12 else v
        else:
            v_hat = v
        p_i = torch.sqrt(v_hat) + eps
        inv_sqrt_p = 1.0 / (torch.sqrt(p_i) + 1e-12)
        parts.append(inv_sqrt_p.reshape(-1))
    return torch.cat(parts)
