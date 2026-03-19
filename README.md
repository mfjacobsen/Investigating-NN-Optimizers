# Investigating-NN-Optimizers

## Introduction

**Edge of Stability (EoS)** is the regime where neural network training operates at the boundary between stable and unstable dynamics. For SGD, stability is roughly when *sharpness*, the maximum eigenvalue of the loss Hessian, stays below about 2/η, with η the learning rate. When sharpness exceeds this threshold, training can diverge or exhibit oscillation, spikes, and degraded convergence.

We investigate EoS behavior across SGD, Adam, Shampoo, and Muon on a shared setup: a small fully connected network on CIFAR-10, with learning rate sweeps and metrics such as sharpness, gradient norm, and update norm. A main focus is how adaptive optimizers like Adam change EoS behavior compared to SGD, e.g. oscillation rather than explosion, and how that ties to effective step size and preconditioning.

### Project structure

| Path | Description |
|------|-------------|
| `scripts/` | Entrypoints to run EoS experiments, e.g. `run_PP_adam_eos_python.py`, `run_EC_sgd.py`, `run_ZJ_shampoo.py` |
| `notebooks/` | Per-optimizer EoS notebooks for Shampoo, SGD, Adam, Muon |
| `src/` | Shared code for training, Hessian/sharpness, and data loading |
| `output/eos/` | Experiment outputs: CSVs and per-run subdirectories |
| `plots/` | Generated plots, e.g. `plots/adam_plots/` |

### Local Setup

Run the following code block to clone the Github repository and setup the 
virtual environment:
```bash
git clone https://github.com/mfjacobsen/Investigating-NN-Optimizers
cd Investigating-NN-Optimizers
conda env create -f environment.yml
conda activate inv-nn-opt-env
```

### DSMLP Setup
Run the following block in the terminal in DSMLP to clone the GitHub repository
and set-up the conda environment. The environment is installed in the scratch
directory since the size of pytorch exceeds the the default location's storage
quota.
``` bash
cd ~
mkdir -p /scratch/$USER/conda/envs
mkdir -p /scratch/$USER/conda/pkgs
mkdir -p /scratch/$USER/pip-cache

conda config --show envs_dirs | grep -q "/scratch/$USER/conda/envs" \
  || conda config --add envs_dirs "/scratch/$USER/conda/envs"
conda config --show pkgs_dirs | grep -q "/scratch/$USER/conda/pkgs" \
  || conda config --add pkgs_dirs "/scratch/$USER/conda/pkgs"

grep -qxF 'export PIP_CACHE_DIR=/scratch/$USER/pip-cache' ~/.bashrc \
  || echo 'export PIP_CACHE_DIR=/scratch/$USER/pip-cache' >> ~/.bashrc
source ~/.bashrc

mkdir -p ~/private
cd ~/private
if [ ! -d Investigating-NN-Optimizers ]; then
  git clone https://github.com/mfjacobsen/Investigating-NN-Optimizers
fi

cd ~/private/Investigating-NN-Optimizers

conda env create -f environment.yml
conda activate inv-nn-opt-env

python -m ipykernel install --user --name inv-nn-opt-env --display-name "Python (inv-nn-opt-env)"
```

For subsequent logins to DSMLP run:
```
conda activate inv-nn-opt-env
cd ~/private/Investigating-NN-Optimizers
```

To update the environment when dependencies change run:
```
conda activate base
cd ~/private/Investigating-NN-Optimizers
conda env update -n inv-nn-opt-env -f environment.yml --prune
```

### Running Experiments

From the repo root:

**Shampoo EoS:**
```bash
python scripts/run_ZJ_shampoo.py
```

**SGD EoS:**
```bash
python scripts/run_EC_sgd.py
```

**Adam EoS:**
```bash
python scripts/run_PP_adam_eos_python.py
```

Results go to `output/eos/` in subdirectories named by optimizer and author initials, e.g. `output/eos/shampoo_ZJ/`, `output/eos/sgd_EC/`, `output/eos/adam_PP/`. The CSVs record per epoch training loss, accuracy, sharpness as the Hessian max eigenvalue, gradient norm, and parameter update norm. Plots are written to `plots/`, e.g. `plots/adam_plots/`, and are also embedded in each notebook so they render on GitHub.

### Notebooks

| Notebook | Description |
|----------|-------------|
| `ZJ_shampoo_eos.ipynb` | Shampoo optimizer EoS investigation |
| `ZJ_muon_eos_batchsize.ipynb` | Muon optimizer minibatch size sweep |
| `EC_sgd.ipynb` | SGD EoS investigation with minibatch training |
| `PP_adam_eos.ipynb` | Adam optimizer EoS investigation |
| `PP_adam_eos_v2.ipynb` | Adam optimizer EoS investigation (updated) |
| `MJ_muon_eos.ipynb` | Muon optimizer EoS investigation |
| `MJ_sgd_eos.ipynb` | SGD EoS investigation (full batch) |
| `MJ_adam_eos.ipynb` | Adam optimizer EoS investigation |

### Further reading

Cohen, Jeremy, Simran Kaur, Yuanzhi Li, J. Zico Kolter, and Ameet Talwalkar. 2021. [*Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability*](https://arxiv.org/abs/2103.00065). CoRR abs/2103.00065.

Andreyev, Arseniy, and Pierfrancesco Beneventano. 2025. *Edge of Stochastic Stability: Revisiting the Edge of Stability for SGD.*

Cohen et al. 2024. *Adaptive Gradient Methods at the Edge of Stability.*

### Contributors

Notebooks and experiments are by project members; initials such as ZJ, EC, PP, MJ in notebook and output directory names indicate who led each part.
