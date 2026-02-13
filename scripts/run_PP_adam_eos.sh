#!/bin/bash
#SBATCH --job-name=PP_adam_eos
#SBATCH --output=PP_adam_eos_%j.out
#SBATCH --error=PP_adam_eos_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# Load necessary modules (adjust if needed for DSMLP)
# module load cuda/11.8  # Uncomment if needed

# Navigate to project directory
cd /dsmlp/home-fs04/85/385/prpotluri/Investigating-NN-Optimizers

# Load any necessary modules (uncomment if needed)
# module load cuda/11.8  # Example - adjust based on DSMLP setup
# module load python/3.11  # Example - adjust based on DSMLP setup

# Use system Python
# If packages aren't available, you may need to install them first:
# python3 -m pip install --user torch torchvision plotly numpy pandas jupyter

# Verify GPU is available
echo "CUDA available: $(python3 -c 'import torch; print(torch.cuda.is_available())' 2>&1)"
echo "GPU device: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")' 2>&1)"

# Run the notebook using jupyter nbconvert (executes and saves output)
jupyter nbconvert --to notebook --execute --inplace notebooks/PP_adam_eos.ipynb

# Alternative: If you prefer to run as a Python script, uncomment below and comment above
# jupyter nbconvert --to script notebooks/PP_adam_eos.ipynb
# python notebooks/PP_adam_eos.py

echo "Training completed!"
