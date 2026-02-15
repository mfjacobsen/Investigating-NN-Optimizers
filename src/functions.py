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
    rng = np.random.RandomState(seed.SEED)
    
    X = np.asarray(X)
    y = np.asarray(y)

    classes = np.unique(y)
    indices = []

    for c in classes:
        cls_idx = np.where(y == c)[0]
        chosen = rng.choice(cls_idx, num_per_class, replace=False)
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

def get_hessian_metrics(model, optimizer, criterion, X, y, t,
                        subsample_dim = 1024, iters=30, tol = 1e-4,
                        generator=generator):
    """Gets the sharpness of the Hessian or effective Hessian, depending on the
    optimizer. 

    Args:
        model (_type_): The neural network model
        optimizer (_type_): The optimizer used for training
        criterion (_type_): The loss function
        X (_type_): Input data
        y (_type_): Target labels
        t (_type_): Current epoch or iteration
        subsample_dim (int, optional): Number of samples to subsample for Hessian computation. Defaults to 1024.
        iters (int, optional): Number of power iteration steps. Defaults to 30.
        tol (float, optional): Tolerance for convergence in power iteration. Defaults to 1e-4.
        generator (_type_, optional): Random generator for reproducibility. Defaults to seed.generator.
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
    
    if isinstance(optimizer, torch.optim.RMSprop):
        
        # Compute adaptive scaling matrix D (sqrt) for effective Hessian
        v_t = torch.cat([state['square_avg'].reshape(-1)
                        for state in optimizer.state.values()]
                        ).detach()

        eps = optimizer.param_groups[0]['eps']
        D_sqrt = torch.sqrt(1 / torch.sqrt(v_t + eps))

        # Compute effective Hessian-vector product
        def Av(v):
            return D_sqrt * Hv(D_sqrt * v)
        
        lambda_A = power_iteration(Av)

  
    elif isinstance(optimizer, torch.optim.Adam):

        g0 = optimizer.param_groups[0]
        beta1, beta2 = g0["betas"]
        eps = g0["eps"]

        # get v_t
        v_t = torch.cat([
            optimizer.state[p]["exp_avg_sq"].reshape(-1)
            for p in model.param_list
            if p in optimizer.state and "exp_avg_sq" in optimizer.state[p]
        ]).detach()

        # bias corrections
        bc1 = 1.0 - (beta1 ** t)
        bc2 = 1.0 - (beta2 ** t)
        v_hat = v_t / bc2

        # Compute adaptive scaling matrix D (sqrt) for effective Hessian
        D_sqrt = (v_hat + eps).pow(-0.25) / (bc1 ** 0.5)

        def Av(v):
            return D_sqrt * Hv(D_sqrt * v)

        lambda_A = power_iteration(Av)
    else:
        lambda_A = None

    return lambda_H, lambda_A

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

    if isinstance(criterion, nn.MSELoss):
        y_loss = torch.nn.functional.one_hot(
            y, num_classes=model.num_labels).float().to(device)
       
    else:
        y_loss = y.to(device)

    start = time.time()
    
    train_acc = 0.0
    epoch = 0

    while train_acc < accuracy and epoch < epochs :

        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y_loss)
        loss.backward()
        optimizer.step()

        train_losses[epoch] = loss.item()

        if epoch % (epochs // 100) == 0:
            H_sharps[epoch], A_sharps[epoch] = get_hessian_metrics(
                model, optimizer, criterion, X, y_loss, epoch + 1
            )

        with torch.no_grad():
            model.eval()
            train_preds = outputs.argmax(dim=1)
            test_preds = model(X_test).argmax(dim=1)
            train_acc = (train_preds == y).float().mean().item()
            test_acc = (test_preds == y_test).float().mean().item()
            train_accuracies[epoch] = train_acc
            test_accuracies[epoch] = test_acc
        model.train()

        if (epoch+1) % 1000 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, " +
                  f"Time: {round(((time.time() - start) / 60), 2)}, " +
                  f"Train Acc: {train_accuracies[epoch]:.4f}, " +
                  f"Test Acc: {test_accuracies[epoch]:.4f}, ")
        epoch += 1

    metadata, output_data = setup_output_files(output_dir)
    model_id = metadata.shape[0] + 1

    metadata.loc[metadata.shape[0]] ={
        "model_id": model_id,
        "model_type": model.__class__.__name__,
        "activation_function": model.activation.__name__,
        "optimizer": optimizer.__class__.__name__,
        "criterion": criterion.__class__.__name__,
        "learning_rate": learning_rate,
        "beta1": optimizer.param_groups[0].get('betas', (np.nan, np.nan))[0],
        "beta2": optimizer.param_groups[0].get('betas', (np.nan, np.nan))[1],
        "momentum": momentum,
        "num_epochs": epochs,
        "time_minutes": round((time.time() - start) / 60, 2),
    }

    output_data = pd.concat([output_data, pd.DataFrame({
        "model_id": np.ones_like(train_losses) * model_id,
        "epoch": np.arange(1, epochs + 1),
        "train_loss": train_losses,
        "sharpness_H": H_sharps.round(4),
        "sharpness_A": A_sharps.round(4),
        "test_accuracy": test_accuracies,
        "train_accuracy": train_accuracies,
    })], ignore_index=True)

    save_output_files(metadata, output_data, output_dir)

def train_minibatch_sgd_model(model, optimizer, criterion, epochs, accuracy,
                         train_loader, test_loader, X_full, y_full, output_dir):
    """
    Trains a model using minibatch sgd and computes Hessian metrics for both the batch and full dataset.

    Args:
        model: The neural network model
        optimizer: The optimizer (e.g., SGD, Adam)
        criterion: Loss function
        epochs: Maximum number of epochs
        accuracy: Target accuracy to stop training early
        train_loader: DataLoader for mini-batch training
        test_loader: DataLoader for test evaluation
        X_full: Full training dataset tensor
        y_full: Full training labels tensor
        output_dir: Directory to save output files
    """
    print(f"Training {model.__class__.__name__} with " +
          f"{optimizer.__class__.__name__} and learning rate " +
          f"{optimizer.param_groups[0]['lr']} for {epochs} epochs.")

    learning_rate = optimizer.param_groups[0]['lr']
    momentum = optimizer.param_groups[0].get('momentum', 0.0)

    model.to(device)
    model.train()

    # Arrays to store metrics per epoch
    train_losses = np.full(epochs, np.nan)
    train_accuracies = np.full(epochs, np.nan)
    test_accuracies = np.full(epochs, np.nan)

    # Separate sharpness metrics for mini-batch and full dataset
    H_sharps_batch = np.full(epochs, np.nan)  # Mini-batch sharpness
    H_sharps_full = np.full(epochs, np.nan)   # Full-dataset sharpness

    start = time.time()
    train_acc = 0.0
    epoch = 0

    # Prepare y_full for loss computation
    if isinstance(criterion, nn.MSELoss):
        y_full_loss = torch.nn.functional.one_hot(
            y_full, num_classes=model.num_labels).float().to(device)
    else:
        y_full_loss = y_full

    while train_acc < accuracy and epoch < epochs:
        epoch_loss = 0.0
        num_batches = 0

        batch_H_list = []
        batch_A_list = []

        # Mini-batch training
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            if isinstance(criterion, nn.MSELoss):
                y_batch_loss = torch.nn.functional.one_hot(
                    y_batch, num_classes=model.num_labels).float().to(device)
            else:
                y_batch_loss = y_batch

            # Calculate mini-batch sharpness periodically
            if num_batches % 10 == 0:
                h_s, a_s = get_hessian_metrics(
                    model, optimizer, criterion, X_batch, y_batch_loss, epoch + 1
                )
                if h_s is not None: batch_H_list.append(h_s)
                if a_s is not None: batch_A_list.append(a_s)

            # Standard training step
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch_loss)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        train_losses[epoch] = epoch_loss / num_batches

        # Store average batch sharpness for this epoch
        if batch_H_list:
            H_sharps_batch[epoch] = np.mean(batch_H_list)


        # Full-dataset sharpness
        h_s_full, a_s_full = get_hessian_metrics(
            model, optimizer, criterion, X_full, y_full_loss, epoch + 1
        )
        H_sharps_full[epoch] = h_s_full

        # Calculate full-dataset sharpness periodically
        eval_interval = max(1, epochs // 100)
        if epoch % eval_interval == 0 or epoch == epochs - 1:
            # Accuracy evaluation
            with torch.no_grad():
                model.eval()

                # Train accuracy
                train_correct, train_total = 0, 0
                for xb, yb in train_loader:
                    xb = xb.to(device)
                    out = model(xb)
                    train_correct += (out.argmax(dim=1) == yb.to(device)).sum().item()
                    train_total += yb.size(0)
                train_acc = train_correct / train_total
                train_accuracies[epoch] = train_acc

                # Test accuracy
                test_correct, test_total = 0, 0
                for xb, yb in test_loader:
                    xb = xb.to(device)
                    out = model(xb)
                    test_correct += (out.argmax(dim=1) == yb.to(device)).sum().item()
                    test_total += yb.size(0)
                test_accuracies[epoch] = test_correct / test_total

            model.train()

        if (epoch+1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {train_losses[epoch]:.4f}, " +
                  f"Batch Sharp: {H_sharps_batch[epoch]:.2f}, " +
                  f"Full Sharp: {H_sharps_full[epoch]:.2f}, " +
                  f"Train Acc: {train_acc:.4f}")

        epoch += 1

    print(f"Completed training of {model.__class__.__name__} with " +
          f"{optimizer.__class__.__name__} and learning rate " +
          f"{optimizer.param_groups[0]['lr']}. Took {epoch} epochs and " +
          f"{round(time.time() - start, 2)} seconds. " +
          f"Final training accuracy: {train_acc:.4f}; " +
          f"Final testing accuracy: {test_accuracies[epoch-1]:.4f}")

    # Save results
    metadata, output_data = setup_output_files(output_dir)
    model_id = metadata.shape[0] + 1

    print(f"Saved with model_id {model_id}")
    print("=================================================")

    metadata.loc[metadata.shape[0]] = {
        "model_id": model_id,
        "model_type": model.__class__.__name__,
        "activation_function": model.activation.__name__,
        "optimizer": optimizer.__class__.__name__,
        "criterion": criterion.__class__.__name__,
        "learning_rate": learning_rate,
        "momentum": momentum,
        "num_epochs": epoch,
        "time_minutes": round((time.time() - start) / 60, 2),
    }

    # Create output dataframe with both batch and full sharpness metrics
    new_output = pd.DataFrame({
        "model_id": np.ones(epoch) * model_id,
        "epoch": np.arange(1, epoch + 1),
        "train_loss": train_losses[:epoch],
        "sharpness_H_batch": H_sharps_batch[:epoch],      # Mini-batch H sharpness
        "sharpness_H_full": H_sharps_full[:epoch],        # Full-dataset H sharpness
        "test_accuracy": test_accuracies[:epoch],
        "train_accuracy": train_accuracies[:epoch],
    })

    output_data = pd.concat([output_data, new_output], ignore_index=True)
    save_output_files(metadata, output_data, output_dir)
