#!/bin/bash
#SBATCH --job-name=PP_adam_eos
#SBATCH --output=PP_adam_eos_%j.out
#SBATCH --error=PP_adam_eos_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# Navigate to project directory
cd /dsmlp/home-fs04/85/385/prpotluri/Investigating-NN-Optimizers

# Load any necessary modules (uncomment if needed)
# module load cuda/11.8  # Example - adjust based on DSMLP setup
# module load python/3.11  # Example - adjust based on DSMLP setup

# Use system Python (or install packages with --user if needed)
# If packages aren't available, you may need to install them first:
# python3 -m pip install --user torch torchvision plotly numpy pandas

# Verify GPU is available
echo "=========================================="
echo "GPU Check:"
echo "=========================================="
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')" 2>&1
echo ""

# Run the Python script
echo "Starting training..."
python3 run_PP_adam_eos_python.py

echo ""
echo "Job completed!"
