from . import seed

import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import datasets
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

try:
    from tqdm.auto import tqdm  # Uses notebook widget in Jupyter, text bar in terminal
except ImportError:
    def tqdm(iterable=None, total=None, desc=None, unit=None, **kwargs):
        if iterable is not None:
            return iterable
        return _NullProgressBar()

    class _NullProgressBar:
        def update(self, n=1): pass
        def set_postfix(self, **kwargs): pass
        def close(self): pass
        def write(self, s, file=None, end="\n"): print(s, end=end)
        def __enter__(self): return self
        def __exit__(self, *args): pass

device = seed.device
generator = seed.generator

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_ROOT = os.path.join(BASE_DIR, "output")

def sample_data(X, y, num_per_class):
    """Used to subsample CIFAR-10

    Args:
        X (_type_): The dataset
        y (_type_): The labels
        num_per_class (_type_): Number of samples to draw per class

    Returns:
        _type_: The subsampled data and labels
    """
    X = np.asarray(X)
    y = np.asarray(y)

    classes = np.unique(y)
    indices = []

    for c in classes:
        cls_idx = np.where(y == c)[0]
        chosen = np.random.choice(cls_idx, num_per_class, replace=False)
        indices.append(chosen)

    indices = np.concatenate(indices)
    return X[indices], y[indices]

def load_cifar_10(num_per_class=500, test_num_per_class=100):
    """Loads CIFAR-10. Defaults to a 5k image subset with 1k test images.

    Args:
        num_per_class (int, optional): The number of training samples per class. Defaults to 500.
        test_num_per_class (int, optional): The number of test samples per class. Defaults to 100.

    Returns:
        tuple: Tuple containing training data, training labels, test data, test labels.
    """

    DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    
    # Load raw CIFAR-10 
    train = datasets.CIFAR10(root=DATA_DIR, train=True,  download=True)
    test  = datasets.CIFAR10(root=DATA_DIR, train=False, download=True)
    # # Subsample
    X, y  = sample_data(train.data, train.targets, num_per_class)
    X_test, y_test = sample_data(test.data, test.targets, test_num_per_class)

    # Convert to float and scale
    X  = X.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0

    # Normalize
    mean = X.mean(axis=(0,1,2),keepdims=True)
    std = X.std(axis=(0,1,2),keepdims=True)
    X = (X - mean) / std
    X_test = (X_test - mean) / std

    # Reshape to NCHW
    X = np.transpose(X, (0, 3, 1, 2))
    X_test = np.transpose(X_test, (0, 3, 1, 2))

    # Convert to torch
    X = torch.tensor(X, dtype=torch.float32, device=device)
    y = torch.tensor(y, dtype=torch.long, device=device)
    X_test = torch.tensor(X_test, dtype=torch.float32, device=device)
    y_test = torch.tensor(y_test, dtype=torch.long, device=device)

    return X, y, X_test, y_test

def resolve_output_dir(subdir):
    if subdir is None or subdir == "":
        raise ValueError("You must provide a subdirectory name inside output/")

    path = os.path.normpath(os.path.join(OUTPUT_ROOT, subdir))

    if not os.path.commonpath([path, OUTPUT_ROOT]) == OUTPUT_ROOT:
        raise ValueError("output_dir must stay inside output/")

    return path

def setup_output_files(output_dir): 
    """Sets up the output files for model training. Metadata contains information
    about the model and the model's id. Output contains data at each epoch of training
    for each model. If the files already exist, they are loaded as dataframes. 

    Args:
        output_dir (str): The directory where output files will be saved.

    Returns:
        tuple: Tuple containing metadata and output data dataframes.
    """

    output_dir = resolve_output_dir(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    metadata_path = os.path.join(output_dir, "metadata.csv")
    output_data_path = os.path.join(output_dir, "output.csv")

    if os.path.exists(metadata_path):
            metadata = pd.read_csv(metadata_path)
    else:
        metadata = pd.DataFrame({
            "model_id": pd.Series(dtype="int"),
            "model_type": pd.Series(dtype="str"),
            "activation_function": pd.Series(dtype="str"),
            "optimizer": pd.Series(dtype="str"),
            "criterion": pd.Series(dtype="str"),
            "learning_rate": pd.Series(dtype="float"),
            "beta1": pd.Series(dtype="float"),
            "beta2": pd.Series(dtype="float"),
            "num_epochs": pd.Series(dtype="int"),
            "time_minutes": pd.Series(dtype="float"),
        })

    if os.path.exists(output_data_path):
        output_data = pd.read_csv(output_data_path)
    else:
        output_data = pd.DataFrame({
            "model_id": pd.Series(dtype="int"),
            "epoch": pd.Series(dtype="int"),
            "train_loss": pd.Series(dtype="float"),
            "train_accuracy": pd.Series(dtype="float"),
            "test_accuracy": pd.Series(dtype="float"),
            "sharpness_H": pd.Series(dtype="float"),
            "sharpness_A": pd.Series(dtype="float"),
        })

    return metadata, output_data

def load_output_files(output_dir):
    """Loads the metadata and output files into dataframes

    Args:
        output_dir (str): Path to the output directory.

    Returns:
        tuple: Tuple containing metadata and output data dataframes.
    """
    output_dir = resolve_output_dir(output_dir)

    metadata_path = os.path.join(output_dir, "metadata.csv")
    output_data_path = os.path.join(output_dir, "output.csv")

    metadata = pd.read_csv(metadata_path)
    output_data = pd.read_csv(output_data_path)

    return metadata, output_data

def save_output_files(metadata, output_data, output_dir):
    """Saves metadata and output dataframes to csv.

    Args:
        metadata (_type_): metadata dataframe
        output_data (_type_): output data dataframe
        output_dir (str): Directory to save the output files.
    """
    output_dir = resolve_output_dir(output_dir)

    metadata_path = os.path.join(output_dir, "metadata.csv")
    output_data_path = os.path.join(output_dir, "output.csv")

    metadata.to_csv(metadata_path, index=False)
    output_data.to_csv(output_data_path, index=False)

def delete_model_data(model_ids, output_dir):
    """Deletes all model data in the list of provided model_ids from the metadata
    and output files.

    Args:
        model_ids (_type_): The model ids to delete
        output_dir (str): Directory where output files are stored.
    """
    output_dir = resolve_output_dir(output_dir)
    
    metadata, output_data = load_output_files(output_dir)
    metadata = metadata[~metadata['model_id'].isin(model_ids)]
    output_data = output_data[~output_data['model_id'].isin(model_ids)]
    save_output_files(metadata, output_data, output_dir)

def get_hessian_metrics(model, optimizer, criterion, X, y, 
                        subsample_dim = 1024, iters=30, tol = 1e-4):
    """Gets the sharpness of the Hessian or effective Hessian, depending on the
    optimizer. 

    Args:
        model (_type_): The neural network model
        optimizer (_type_): The optimizer used for training
        criterion (_type_): The loss function
        X (_type_): Input data
        y (_type_): Target labels
        subsample_dim (int, optional): Number of samples to subsample for Hessian computation. Defaults to 1024.
        iters (int, optional): Number of power iteration steps. Defaults to 30.
        tol (float, optional): Tolerance for convergence in power iteration. Defaults to 1e-4.

    Returns:
        tuple: Tuple containing the sharpness of the Hessian and effective Hessian (if applicable).
    """
    # Subsample data for compute efficiency
    subsample_dim = min(subsample_dim, len(X))
    idx = torch.randperm(len(X), device=X.device, generator=generator)[:subsample_dim]
    X = X[idx]
    y = y[idx]
    
    # Build graph for gradient
    outputs = model(X)
    loss = criterion(outputs, y)

    grads = torch.autograd.grad(
        loss, model.param_list,
        create_graph=True
    )
    g_flat = torch.cat([g.reshape(-1) for g in grads])
    dim    = g_flat.numel()
    device = g_flat.device

    # Computes Hessian-vector product with Pearlmutter trick
    def Hv(v):
        Hv_list = torch.autograd.grad(
            g_flat @ v,
            model.param_list,
            retain_graph=True
        )
        return torch.cat([h.reshape(-1) for h in Hv_list])
    
    # Performs power iteration to estimate largest eigenvalue
    def power_iteration(matvec):
        v = torch.randn(dim, device=device, generator=generator)
        v /= v.norm()

        eig_old = 0.0
        for _ in range(iters):
            Hv_v = matvec(v)
            eig = (v @ Hv_v).item()   
            v = Hv_v / Hv_v.norm()

            if abs(eig - eig_old) / (abs(eig_old) + 1e-12) < tol:
                break
            eig_old = eig

        Hv_v = matvec(v)
        eig = (v @ Hv_v).item()
        return eig

    lambda_H = power_iteration(Hv)
    lambda_A = None

    if isinstance(optimizer, torch.optim.RMSprop):
        # Compute adaptive scaling matrix D (sqrt) for effective Hessian
        v_t = torch.cat([state['square_avg'].reshape(-1)
                        for state in optimizer.state.values()]
                        ).detach()
        eps = optimizer.param_groups[0]['eps']
        D_sqrt = torch.sqrt(1 / torch.sqrt(v_t + eps))
        def Av(v):
            return D_sqrt * Hv(D_sqrt * v)
        lambda_A = power_iteration(Av)

    elif isinstance(optimizer, (torch.optim.Adam, torch.optim.AdamW)):
        # Adam effective Hessian: S^(1/2) H S^(1/2), S = diag(1/(sqrt(v_hat)+eps))
        scaling = _get_adam_scaling_vector(model, optimizer)
        if scaling is not None:
            if scaling.numel() != dim:
                import warnings
                warnings.warn(
                    f"Adam scaling dim {scaling.numel()} != param dim {dim}; "
                    "skipping lambda_eff."
                )
            else:
                # S = diag(s), s = 1/(sqrt(v_hat)+eps) => S^(1/2) = diag(sqrt(s))
                D_sqrt = torch.sqrt(torch.clamp(scaling, min=1e-12))
                def Av(v):
                    return D_sqrt * Hv(D_sqrt * v)
                lambda_A = power_iteration(Av)
        else:
            lambda_A = None

    return lambda_H, lambda_A


def _get_adam_scaling_vector(model, optimizer):
    """Compute Adam scaling vector s = 1/(sqrt(v_hat)+eps) with bias correction.
    Order matches model.param_list (same as gradient flattening).
    Returns None if optimizer state not initialized.
    """
    if not hasattr(model, 'param_list'):
        return None
    eps = optimizer.param_groups[0].get('eps', 1e-8)
    beta2 = optimizer.param_groups[0].get('betas', (0.9, 0.999))[1]
    scaling_parts = []
    for p in model.param_list:
        state = optimizer.state.get(p)
        if state is None or 'exp_avg_sq' not in state:
            return None
        exp_avg_sq = state['exp_avg_sq'].detach()
        step = state.get('step', 1)
        if step is None:
            step = 1
        # Bias-corrected second moment: v_hat = exp_avg_sq / (1 - beta2^step)
        bc = 1.0 - beta2 ** step
        if abs(bc) < 1e-12:
            v_hat = exp_avg_sq
        else:
            v_hat = exp_avg_sq / bc
        # s = 1 / (sqrt(v_hat) + eps)
        s = 1.0 / (torch.sqrt(v_hat) + eps)
        scaling_parts.append(s.reshape(-1))
    return torch.cat(scaling_parts)


def _get_adam_effective_step_stats(model, optimizer):
    """Compute effective step size stats for Adam/AdamW: α/(√v̂+ε) per parameter.
    Returns (mean, std, max) as Python floats, or (None, None, None) if not Adam or state missing.
    """
    if not isinstance(optimizer, (torch.optim.Adam, torch.optim.AdamW)):
        return None, None, None
    if not hasattr(model, 'param_list'):
        return None, None, None
    lr = optimizer.param_groups[0]['lr']
    eps = optimizer.param_groups[0].get('eps', 1e-8)
    beta2 = optimizer.param_groups[0].get('betas', (0.9, 0.999))[1]
    steps = []
    for p in model.param_list:
        state = optimizer.state.get(p)
        if state is None or 'exp_avg_sq' not in state:
            return None, None, None
        exp_avg_sq = state['exp_avg_sq'].detach()
        step = state.get('step', 1)
        if step is None:
            step = 1
        bc = 1.0 - beta2 ** step
        v_hat = exp_avg_sq / bc if abs(bc) >= 1e-12 else exp_avg_sq
        eff_lr = lr / (torch.sqrt(v_hat) + eps)
        steps.append(eff_lr.reshape(-1))
    all_steps = torch.cat(steps)
    return all_steps.mean().item(), all_steps.std().item(), all_steps.max().item()


def train_model(model, optimizer, criterion, epochs, accuracy, X, y, X_test, y_test, output_dir):
    """Trains the provided model with the specified optimizer and criterion for 
    a set number of epochs or until the desired accuracy is reached. Records 
    training loss, training accuracy, test accuracy, and sharpness metrics at 
    each epoch.

    Args:
        model (_type_): The neural network model to train
        optimizer (_type_): The optimizer used for training
        criterion (_type_): The loss function used for training
        epochs (_type_): The maximum number of training epochs
        accuracy (_type_): The target accuracy to stop training early
        X (_type_): Training input data
        y (_type_): Training target labels
        X_test (_type_): Test input data
        y_test (_type_): Test target labels
        output_dir (_type_): Directory to save output files
    """
    print(f"Training {model.__class__.__name__} with " +
          f"{optimizer.__class__.__name__} and learning rate " +
          f"{optimizer.param_groups[0]['lr']} for {epochs} epochs.")

    learning_rate = optimizer.param_groups[0]['lr']
    momentum = optimizer.param_groups[0].get('momentum', 0.0)

    model.to(device)
    model.train()

    train_losses = np.full(epochs, np.nan)
    train_accuracies = np.full(epochs, np.nan)
    test_accuracies = np.full(epochs, np.nan)
    H_sharps = np.full(epochs, np.nan)
    A_sharps = np.full(epochs, np.nan)
    eff_step_mean = np.full(epochs, np.nan)
    eff_step_std = np.full(epochs, np.nan)
    eff_step_max = np.full(epochs, np.nan)

    if isinstance(criterion, nn.MSELoss):
        y_loss = torch.nn.functional.one_hot(
            y, num_classes=model.num_labels).float().to(device)
       
    else:
        y_loss = y.to(device)

    start = time.time()
    train_acc = 0.0
    epoch = 0

    pbar = tqdm(total=epochs, desc="Epochs", unit="epoch", dynamic_ncols=True)

    while train_acc < accuracy and epoch < epochs:

        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y_loss)
        loss.backward()
        optimizer.step()

        train_losses[epoch] = loss.item()

        # Skip Hessian for short runs (epochs < 100) to speed up training
        # For longer runs, compute at most ~20 times
        if epochs >= 100 and epoch % max(100, epochs // 20) == 0:
            H_sharps[epoch], A_sharps[epoch] = get_hessian_metrics(
                model, optimizer, criterion, X, y_loss
            )
            em, es, ex = _get_adam_effective_step_stats(model, optimizer)
            if em is not None:
                eff_step_mean[epoch], eff_step_std[epoch], eff_step_max[epoch] = em, es, ex

        with torch.no_grad():
            model.eval()
            train_preds = outputs.argmax(dim=1)
            test_preds = model(X_test).argmax(dim=1)
            train_acc = (train_preds == y).float().mean().item()
            test_acc = (test_preds == y_test).float().mean().item()
            train_accuracies[epoch] = train_acc
            test_accuracies[epoch] = test_acc
        model.train()

        pbar.update(1)
        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            train_acc=f"{train_acc:.3f}",
            test_acc=f"{test_acc:.3f}",
        )

        # Print progress at sensible intervals (works even without tqdm)
        print_interval = max(1, epochs // 20)
        if (epoch + 1) % print_interval == 0 or (epoch + 1) == epochs:
            elapsed = round((time.time() - start) / 60, 2)
            pbar.write(
                f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, "
                f"Time: {elapsed} min, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}"
            )
        epoch += 1

    pbar.close()

    metadata, output_data = setup_output_files(output_dir)
    model_id = metadata.shape[0] + 1

    meta_row = {
        "model_id": model_id,
        "model_type": model.__class__.__name__,
        "activation_function": model.activation.__name__,
        "optimizer": optimizer.__class__.__name__,
        "criterion": criterion.__class__.__name__,
        "learning_rate": learning_rate,
        "num_epochs": epochs,
        "time_minutes": round((time.time() - start) / 60, 2),
    }
    if hasattr(optimizer, 'param_groups') and optimizer.param_groups:
        pg = optimizer.param_groups[0]
        if 'betas' in pg:
            meta_row["beta1"], meta_row["beta2"] = pg["betas"][0], pg["betas"][1]
        else:
            meta_row["momentum"] = pg.get("momentum", 0.0)
    else:
        meta_row["momentum"] = momentum
    metadata.loc[metadata.shape[0]] = meta_row

    lr_lambda_eff = (learning_rate * A_sharps).round(4)
    out_dict = {
        "model_id": np.ones_like(train_losses) * model_id,
        "epoch": np.arange(1, epochs + 1),
        "train_loss": train_losses,
        "sharpness_H": H_sharps.round(4),
        "sharpness_A": A_sharps.round(4),
        "lambda_eff": A_sharps.round(4),
        "lr_lambda_eff": lr_lambda_eff,
        "test_accuracy": test_accuracies,
        "train_accuracy": train_accuracies,
    }
    out_dict["effective_step_mean"] = np.round(eff_step_mean, 6)
    out_dict["effective_step_std"] = np.round(eff_step_std, 6)
    out_dict["effective_step_max"] = np.round(eff_step_max, 6)
    output_data = pd.concat([output_data, pd.DataFrame(out_dict)], ignore_index=True)

    save_output_files(metadata, output_data, output_dir)