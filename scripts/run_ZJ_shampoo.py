#!/usr/bin/env python
"""
Run script for the Shampoo EoS experiment.

Sweeps over 5 learning rates and trains one MLP4 model per LR using
the Shampoo optimizer on inner layers and Adam on all other parameters.
Sharpness is measured every 50 epochs and saved to output/eos/shampoo_ZJ/.

Usage:
    python scripts/run_ZJ_shampoo.py

Results are saved to output/eos/shampoo_ZJ/.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import torch
torch.set_num_interop_threads(1)
torch.set_num_threads(1)

import torch.nn as nn
import torch.nn.functional as F

import src.seed as seed
import src.functions as fn
from src.shampoo import Shampoo, MLP4, train_shampoo_model

device    = torch.device("cpu")
generator = seed.generator

print("Loading CIFAR-10 data...")
X, y, X_test, y_test = fn.load_cifar_10()
print(f"Data loaded. Training set: {X.shape}, Test set: {X_test.shape}")

# Experiment configuration
output_dir        = "eos/shampoo_ZJ"
input_size        = 32 * 32 * 3
hidden_layer_size = 170
num_labels        = 10
activation        = F.relu
criterion         = nn.MSELoss()
epochs            = 500
accuracy          = 1.1  # train for full epochs

learning_rates = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4]

print(f"\nStarting Shampoo EoS sweep over {len(learning_rates)} learning rates...")
print(f"Epochs per run: {epochs}")
print(f"Output directory: output/{output_dir}/\n")

for i, lr in enumerate(learning_rates, 1):
    print(f"\n{'='*60}")
    print(f"Training run {i}/{len(learning_rates)}: LR = {lr}")
    print(f"{'='*60}")

    model = MLP4(input_size, hidden_layer_size, num_labels, activation)

    all_params     = set(model.parameters())
    shampoo_params = list({model.h2.weight, model.h3.weight})
    adam_params    = list(all_params - set(shampoo_params))

    opt_shampoo = Shampoo(
        shampoo_params,
        lr=lr,
        momentum=0.9,
        weight_decay=0,
        update_freq=100,
        epsilon=1e-6,
    )
    opt_adam = torch.optim.Adam(adam_params, lr=lr)

    train_shampoo_model(
        model       = model,
        opt_shampoo = opt_shampoo,
        opt_adam    = opt_adam,
        criterion   = criterion,
        epochs      = epochs,
        accuracy    = accuracy,
        X           = X,
        y           = y,
        X_test      = X_test,
        y_test      = y_test,
        output_dir  = output_dir,
        generator   = generator,
        device      = device,
    )

print(f"\n{'='*60}")
print("All training runs completed!")
print(f"Results saved to: output/{output_dir}/")
print(f"{'='*60}")