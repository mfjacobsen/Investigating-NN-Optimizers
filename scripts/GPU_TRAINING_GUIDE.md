# GPU Training Guide for PP_adam_eos

## Quick Start

### Option 1: Run Notebook Directly (Recommended)
```bash
sbatch run_PP_adam_eos_notebook.sh
```

### Option 2: Run as Python Script
```bash
sbatch run_PP_adam_eos_python.sh
```

### Option 3: Run Original Notebook Script
```bash
sbatch run_PP_adam_eos.sh
```

**Note:** Make sure you have the required Python packages (torch, torchvision, plotly, numpy, pandas, jupyter) installed. If using conda, uncomment the conda activate line in the script. If using system Python, you may need to install packages with `pip install --user` first.

## Check Job Status
```bash
squeue -u $USER
```

## View Output
```bash
# View latest output file
tail -f PP_adam_eos_<JOBID>.out

# View errors
tail -f PP_adam_eos_<JOBID>.err
```

## Adjusting SLURM Parameters

If you need to modify the GPU partition or resources, edit the `.sh` files and change:

- `--partition=gpu` - Change to your DSMLP GPU partition name (check with `sinfo` or ask admin)
- `--gres=gpu:1` - Number of GPUs (1 is usually sufficient)
- `--time=24:00:00` - Maximum job time (adjust based on your needs)
- `--mem=16G` - Memory allocation
- `--cpus-per-task=4` - CPU cores

## Common DSMLP Partition Names
- `gpu`
- `gpu-a100`
- `gpu-v100`
- `gpu-shared`

To find available partitions:
```bash
sinfo
# or
scontrol show partition
```

## Interactive GPU Session (for testing)

If you want to test interactively before submitting a job:

```bash
srun --partition=gpu --gres=gpu:1 --mem=16G --time=2:00:00 --pty bash
conda activate inv-nn-opt-env
cd /dsmlp/home-fs04/85/385/prpotluri/Investigating-NN-Optimizers
python run_PP_adam_eos_python.py
```

## Expected Runtime

With 5 learning rates and 4000 epochs each:
- Estimated time: 4-8 hours depending on GPU
- Output will be saved to: `output/eos/adam_PP/`

## Troubleshooting

1. **GPU not detected**: Check that `torch.cuda.is_available()` returns `True` in the output
2. **Out of memory**: Reduce batch size or model size in the script
3. **Partition error**: Verify partition name with `sinfo` or contact DSMLP support
4. **Environment not found**: Make sure you've run the DSMLP setup from README.md
