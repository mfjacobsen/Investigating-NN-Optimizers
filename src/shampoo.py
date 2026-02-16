"""
shampoo.py -- Shampoo optimizer and MLP4 architecture for EoS experiments.

Contains experiment-specific code for the Shampoo EoS investigation:
  - Shampoo optimizer (Gupta, Koren and Singer, 2018)
  - MLP4 architecture (4 hidden layers, used for all Shampoo experiments)
  - max_shampoo_layer_sharpness (per-layer sharpness via power iteration)
  - train_shampoo_model (training loop with dual optimizer and sharpness tracking)
"""

import time
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

import src.functions as fn


class Shampoo(torch.optim.Optimizer):
    """Shampoo optimizer for 2D weight matrices (Gupta, Koren and Singer, 2018).

    Keeps two small matrices per layer that accumulate gradient statistics
    across rows and columns of each weight matrix. Uses these to rescale
    the gradient before each step -- stretching in unexplored directions
    and shrinking in directions with large historical gradients.

    1D parameters (biases) fall back to plain SGD.
    """

    def __init__(self, params, lr=1e-2, momentum=0.9, weight_decay=0.0,
                 update_freq=100, epsilon=1e-6):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        update_freq=update_freq, epsilon=epsilon)
        super().__init__(params, defaults)

    @staticmethod
    def _inv_quarter(M):
        """Computes M to the power of -1/4 via eigendecomposition."""
        eigvals, eigvecs = torch.linalg.eigh(M)
        inv_quarter = eigvals.clamp(min=1e-30).pow(-0.25)
        return eigvecs @ torch.diag(inv_quarter) @ eigvecs.T

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr          = group['lr']
            momentum    = group['momentum']
            wd          = group['weight_decay']
            update_freq = group['update_freq']
            eps         = group['epsilon']

            for p in group['params']:
                if p.grad is None:
                    continue

                G = p.grad.data
                if wd != 0:
                    G = G.add(p.data, alpha=wd)

                state = self.state[p]

                if p.ndim == 2:
                    m, n = p.shape

                    if len(state) == 0:
                        state['step'] = 0
                        state['L'] = eps * torch.eye(m, device=p.device, dtype=p.dtype)
                        state['R'] = eps * torch.eye(n, device=p.device, dtype=p.dtype)
                        state['L_inv_q'] = torch.eye(m, device=p.device, dtype=p.dtype)
                        state['R_inv_q'] = torch.eye(n, device=p.device, dtype=p.dtype)
                        if momentum > 0:
                            state['G_avg'] = torch.zeros_like(G)

                    state['step'] += 1

                    if momentum > 0:
                        state['G_avg'].mul_(momentum).add_(G, alpha=1.0 - momentum)
                        G_eff = state['G_avg']
                    else:
                        G_eff = G

                    state['L'].add_(G @ G.T)
                    state['R'].add_(G.T @ G)

                    if state['step'] % update_freq == 1 or update_freq == 1:
                        state['L_inv_q'] = Shampoo._inv_quarter(state['L'])
                        state['R_inv_q'] = Shampoo._inv_quarter(state['R'])

                    update = state['L_inv_q'] @ G_eff @ state['R_inv_q']
                    p.data.add_(update, alpha=-lr)

                else:
                    p.data.add_(G, alpha=-lr)

        return loss


class MLP4(nn.Module):
    """4 hidden layer MLP for CIFAR-10 classification."""

    def __init__(self, input_size, hidden_layer_size, num_labels, activation):
        super().__init__()
        self.input_size         = input_size
        self.hidden_layers_size = hidden_layer_size
        self.num_labels         = num_labels
        self.activation         = activation

        self.h1  = nn.Linear(input_size,        hidden_layer_size)
        self.h2  = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.h3  = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.h4  = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.out = nn.Linear(hidden_layer_size, num_labels)

        self.param_list = list(self.parameters())

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.activation(self.h1(x))
        x = self.activation(self.h2(x))
        x = self.activation(self.h3(x))
        x = self.activation(self.h4(x))
        return self.out(x)


def max_shampoo_layer_sharpness(model, opt_shampoo, criterion, X, y,
                                generator, subsample_dim=1024, iters=30, tol=1e-4):
    """Returns the maximum sharpness across all Shampoo weight matrices.

    Sharpness is estimated via power iteration on the Hessian restricted
    to each 2D weight matrix. Uses a random subsample of training data
    for efficiency.
    """
    shampoo_ws = [
        p for g in opt_shampoo.param_groups
        for p in g['params'] if p.ndim == 2
    ]
    n   = X.shape[0]
    m   = min(subsample_dim, n)
    idx = torch.randperm(n, device=X.device, generator=generator)[:m]
    Xs, ys = X[idx], y[idx]

    outputs = model(Xs)
    loss    = criterion(outputs, ys)

    def _layer_sharpness(W):
        (gW,) = torch.autograd.grad(loss, W, create_graph=True, retain_graph=True)
        g_flat = gW.reshape(-1)

        def Hv(v):
            (hW,) = torch.autograd.grad(g_flat @ v, W, retain_graph=True)
            return hW.reshape(-1)

        v = torch.randn(g_flat.numel(), device=W.device, generator=generator)
        v = v / (v.norm() + 1e-12)
        eig_old = 0.0
        for _ in range(iters):
            w   = Hv(v)
            eig = (v @ w).item()
            v   = w / (w.norm() + 1e-12)
            if abs(eig - eig_old) / (abs(eig_old) + 1e-12) < tol:
                break
            eig_old = eig
        return (v @ Hv(v)).item()

    return max(_layer_sharpness(W) for W in shampoo_ws)


def train_shampoo_model(model, opt_shampoo, opt_adam, criterion, epochs, accuracy,
                        X, y, X_test, y_test, output_dir, generator,
                        device=torch.device('cpu')):
    """Trains MLP4 with Shampoo on the inner layers and Adam on the rest.
    Records loss, accuracy, and sharpness every 50 epochs.
    Stops early if loss becomes NaN.
    """
    lr       = opt_shampoo.param_groups[0]['lr']
    momentum = opt_shampoo.param_groups[0].get('momentum', 0.0)
    print(f"Training {model.__class__.__name__} with "
          f"{opt_shampoo.__class__.__name__} and learning rate "
          f"{lr} for {epochs} epochs.")

    model.to(device)
    model.train()

    train_losses     = np.full(epochs, np.nan)
    train_accuracies = np.full(epochs, np.nan)
    test_accuracies  = np.full(epochs, np.nan)
    H_sharps         = np.full(epochs, np.nan)
    A_sharps         = np.full(epochs, np.nan)

    if isinstance(criterion, nn.MSELoss):
        y_loss = F.one_hot(y, num_classes=model.num_labels).float().to(device)
    else:
        y_loss = y.to(device)

    start     = time.time()
    train_acc = 0.0
    epoch     = 0
    loss      = torch.tensor(0.0)

    while train_acc < accuracy and epoch < epochs:
        opt_shampoo.zero_grad(set_to_none=True)
        opt_adam.zero_grad(set_to_none=True)

        outputs = model(X)
        loss    = criterion(outputs, y_loss)
        loss.backward()

        opt_shampoo.step()
        opt_adam.step()

        train_losses[epoch] = loss.item()

        if torch.isnan(loss):
            print(f"  Loss is NaN at epoch {epoch+1}, stopping early.")
            break

        if epoch % 50 == 0:
            H_sharps[epoch], _ = fn.get_hessian_metrics(
                model, opt_shampoo, criterion, X, y_loss, t=epoch, generator=generator
            )
            A_sharps[epoch] = max_shampoo_layer_sharpness(
                model, opt_shampoo, criterion, X, y_loss, generator=generator
            )

        with torch.no_grad():
            model.eval()
            train_preds = outputs.argmax(dim=1)
            test_preds  = model(X_test).argmax(dim=1)
            train_acc   = (train_preds == y).float().mean().item()
            test_acc    = (test_preds  == y_test).float().mean().item()
            train_accuracies[epoch] = train_acc
            test_accuracies[epoch]  = test_acc
        model.train()

        if (epoch + 1) % 100 == 0:
            print(f"  Epoch [{epoch+1}/{epochs}]  "
                  f"loss={loss.item():.4f}  "
                  f"time={round((time.time()-start)/60, 2)}m  "
                  f"train_acc={train_acc:.4f}  "
                  f"test_acc={test_acc:.4f}")
        epoch += 1

    metadata, output_data = fn.setup_output_files(output_dir)
    model_id = metadata.shape[0] + 1

    metadata.loc[metadata.shape[0]] = {
        'model_id':            model_id,
        'model_type':          model.__class__.__name__,
        'activation_function': model.activation.__name__,
        'optimizer':           opt_shampoo.__class__.__name__,
        'criterion':           criterion.__class__.__name__,
        'learning_rate':       lr,
        'momentum':            momentum,
        'num_epochs':          epochs,
        'time_minutes':        round((time.time() - start) / 60, 2),
    }

    output_data = pd.concat([output_data, pd.DataFrame({
        'model_id':       np.ones_like(train_losses) * model_id,
        'epoch':          np.arange(1, epochs + 1),
        'train_loss':     train_losses,
        'sharpness_H':    H_sharps.round(4),
        'sharpness_A':    A_sharps.round(4),
        'test_accuracy':  test_accuracies,
        'train_accuracy': train_accuracies,
    })], ignore_index=True)

    fn.save_output_files(metadata, output_data, output_dir)