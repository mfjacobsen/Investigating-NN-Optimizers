#!/usr/bin/env python
"""
Python script version of PP_adam_eos.ipynb for SLURM execution.
This script runs the training cells from the notebook.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import src.seed as seed
import src.models as models
import src.functions as fn

import torch
import torch.nn as nn

device = seed.device
generator = seed.generator

print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Load data
print("Loading CIFAR-10 data...")
X, y, X_test, y_test = fn.load_cifar_10()
print(f"Data loaded. Training set: {X.shape}, Test set: {X_test.shape}")

# Training configuration
output_dir = "eos/adam_PP"
input_size = X.shape[1] * X.shape[2] * X.shape[3]
num_hidden_layers = 2
hidden_layer_size = 200
epochs = 4000
learning_rates = [3e-3, 1e-3, 3e-4, 1e-4, 3e-5]
accuracy = 1.1

print(f"\nStarting training with {len(learning_rates)} learning rates...")
print(f"Epochs per run: {epochs}")
print(f"Output directory: output/{output_dir}/")

# Training loop
for i, learning_rate in enumerate(learning_rates, 1):
    print(f"\n{'='*60}")
    print(f"Training run {i}/{len(learning_rates)}: LR = {learning_rate}")
    print(f"{'='*60}")
    
    model = models.FullyConnectedNet(
        input_size=input_size,
        num_hidden_layers=num_hidden_layers,
        hidden_layer_size=hidden_layer_size,
        num_labels=10,
        activation=nn.Tanh
    )
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    fn.train_model(
        model, optimizer, criterion, epochs, accuracy,
        X, y, X_test, y_test, output_dir
    )
    
    # Clear GPU cache between runs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print(f"\n{'='*60}")
print("All training runs completed!")
print(f"Results saved to: output/{output_dir}/")
print(f"{'='*60}")
