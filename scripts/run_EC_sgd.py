#!/usr/bin/env python
"""
Run script for the SGD EoS experiment.

Sweeps over learning rates and trains FullyConnectedNet models using
mini-batch SGD. Sharpness is computed on the full dataset at regular
intervals and saved to output/eos/sgd_EC/.

Usage:
    python scripts/run_EC_sgd.py

Results are saved to output/eos/sgd_EC/.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import src.seed as seed
import src.models as models
import src.functions as fn

device    = seed.device
generator = seed.generator

print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

print("Loading CIFAR-10 data...")
X, y, X_test, y_test = fn.load_cifar_10()
print(f"Data loaded. Training set: {X.shape}, Test set: {X_test.shape}")

# Experiment configuration
output_dir        = "eos/sgd_EC"
input_size        = X.shape[1] * X.shape[2] * X.shape[3]
num_hidden_layers = 2
hidden_layer_size = 200
batch_size        = 128
epochs            = 2000
accuracy          = 0.999

learning_rates = [0.05, 0.01, 0.005, 0.001]

# DataLoaders
train_dataset = TensorDataset(X, y)
train_loader  = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    generator=seed.cpu_generator
)
test_dataset = TensorDataset(X_test, y_test)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"\nStarting SGD EoS sweep over {len(learning_rates)} learning rates...")
print(f"Epochs per run: {epochs}")
print(f"Batch size: {batch_size}")
print(f"Output directory: output/{output_dir}/\n")

for i, learning_rate in enumerate(learning_rates, 1):
    print(f"\n{'='*60}")
    print(f"Training run {i}/{len(learning_rates)}: LR = {learning_rate}")
    print(f"{'='*60}")

    model = models.FullyConnectedNet(
        input_size        = input_size,
        num_hidden_layers = num_hidden_layers,
        hidden_layer_size = hidden_layer_size,
        num_labels        = 10,
        activation        = nn.Tanh
    )
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.0)

    fn.train_minibatch_sgd_model(
        model       = model,
        optimizer   = optimizer,
        criterion   = criterion,
        epochs      = epochs,
        accuracy    = accuracy,
        train_loader= train_loader,
        test_loader = test_loader,
        X_full      = X,
        y_full      = y,
        output_dir  = output_dir,
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print(f"\n{'='*60}")
print("All training runs completed!")
print(f"Results saved to: output/{output_dir}/")
print(f"{'='*60}")