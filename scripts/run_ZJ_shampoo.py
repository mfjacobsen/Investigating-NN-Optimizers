import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import src.seed as seed
import src.functions as fn
from src.shampoo import Shampoo, MLP4, train_shampoo_model


# Load data

print("Loading CIFAR-10 data...")
X, y, X_test, y_test = fn.load_cifar_10()

generator = seed.generator


# Shampoo experiment config

output_dir = "eos/shampoo_ZJ"

input_size = 32 * 32 * 3
hidden_layer_size = 170
num_labels = 10

activation = F.relu
criterion = nn.MSELoss()

epochs = 500
accuracy = 1.1

learning_rates = [0.0005, 0.0007, 0.0010, 0.0015, 0.0020]


# Training sweep

for lr in learning_rates:

    print(f"\nTraining model with lr={lr}")

    model = MLP4(input_size, hidden_layer_size, num_labels, activation)

    all_params = set(model.parameters())
    shampoo_params = list({model.h2.weight, model.h3.weight})
    adam_params = list(all_params - set(shampoo_params))

    opt_shampoo = Shampoo(
        shampoo_params,
        lr=lr,
        momentum=0.9,
        weight_decay=0,
        update_freq=1,
        epsilon=1e-6,
    )

    opt_adam = torch.optim.Adam(adam_params, lr=lr)

    train_shampoo_model(
        model=model,
        opt_shampoo=opt_shampoo,
        opt_adam=opt_adam,
        criterion=criterion,
        epochs=epochs,
        accuracy=accuracy,
        X=X,
        y=y,
        X_test=X_test,
        y_test=y_test,
        output_dir=output_dir,
        generator=generator,
    )


# Plotting

md = pd.read_csv(f"output/{output_dir}/metadata.csv")
out = pd.read_csv(f"output/{output_dir}/output.csv")

plot_dir = Path("../plots/shampoo_plots")
plot_dir.mkdir(parents=True, exist_ok=True)

colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2']
learning_rates = sorted(md['learning_rate'].unique())


def make_plot(sharpness_col, sharp_label, filename):

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("Training Loss", sharp_label),
    )

    for i, lr in enumerate(learning_rates):

        model_id = md[md['learning_rate'] == lr]['model_id'].values[0]
        subset = out[out['model_id'] == model_id].sort_values('epoch')

        epochs = subset['epoch'].values
        loss = subset['train_loss'].values
        sharp = subset[sharpness_col].values

        color = colors[i]

        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=loss,
                mode='lines',
                line=dict(color=color),
                name=f"lr={lr}"
            ),
            row=1,
            col=1,
        )

        mask = ~np.isnan(sharp)

        fig.add_trace(
            go.Scatter(
                x=epochs[mask],
                y=sharp[mask],
                mode='markers',
                marker=dict(color=color, size=4),
                showlegend=False
            ),
            row=2,
            col=1,
        )

        fig.add_hline(
            y=2 / lr,
            line_dash="dash",
            line_color=color,
            row=2,
            col=1,
        )

    fig.update_yaxes(type="log", row=1, col=1)
    fig.update_yaxes(type="log", row=2, col=1)

    fig.write_image(plot_dir / filename, width=1200, height=800, scale=3)

    print("Saved", filename)


make_plot(
    "sharpness_H",
    "Hessian Sharpness λ_max",
    "shampoo_loss_hessian.png"
)

make_plot(
    "sharpness_P",
    "Preconditioned Sharpness",
    "shampoo_loss_precond.png"
)

print("Plots saved to", plot_dir)