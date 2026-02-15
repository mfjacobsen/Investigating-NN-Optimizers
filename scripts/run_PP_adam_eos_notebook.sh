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

# Load any necessary modules (uncomment/modify based on DSMLP setup)
# module load cuda/11.8
# module load python/3.11

# Activate your Python environment if you have one set up
# source ~/.bashrc
# conda activate inv-nn-opt-env  # Uncomment if you get the env working
# Or use: source /path/to/your/venv/bin/activate

# Verify GPU is available
echo "=========================================="
echo "Environment Check:"
echo "=========================================="
echo "Python: $(which python3)"
echo "Python version: $(python3 --version)"
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')" 2>&1
echo ""

# Run the notebook using jupyter nbconvert
echo "Starting notebook execution..."
jupyter nbconvert --to notebook --execute --inplace notebooks/PP_adam_eos.ipynb

echo ""
echo "Job completed! Check the output files in output/eos/adam_PP/"
