# Investigating-NN-Optimizers

Experiments on how different optimizers behave when neural network training hits the edge of stability.

## What is this?

We train a small network on CIFAR-10 and compare optimizers: GD (full-batch), SGD (minibatch), Adam, Shampoo, and Muon. The goal is to see how each one behaves when the learning rate is pushed into the regime where training can go unstable or start oscillating. That regime is often called the *edge of stability*. GD is our baseline; the others let us compare different optimizer families. More background and the formal definition of sharpness and the stability threshold are in [ADAM_EOS_NOTEBOOK_SUMMARY.md](ADAM_EOS_NOTEBOOK_SUMMARY.md) and the papers below.

## Getting started

Clone the repo and create the conda environment:

```bash
git clone https://github.com/mfjacobsen/Investigating-NN-Optimizers
cd Investigating-NN-Optimizers
conda env create -f environment.yml
conda activate inv-nn-opt-env
```

## Running experiments

From the repo root you can run:

| Optimizer | Command |
|-----------|---------|
| Shampoo   | `python scripts/run_ZJ_shampoo.py` |
| SGD       | `python scripts/run_EC_sgd.py` |
| Adam      | `python scripts/run_PP_adam_eos_python.py` |

For full-batch GD, use the notebook `notebooks/EC_gd.ipynb`.

Results are written to `output/eos/` in subfolders named by optimizer and author initials (e.g. `gd_EC`, `sgd_EC`, `adam_PP`). Plots go to `plots/` and are also saved inside the notebooks.

## Notebooks

Each optimizer has one or more notebooks that reproduce or extend the script runs:

| Notebook | Content |
|----------|---------|
| `EC_gd.ipynb` | GD, full-batch |
| `EC_sgd.ipynb` | SGD, minibatch |
| `ZJ_shampoo_eos.ipynb` | Shampoo |
| `PP_adam_eos.ipynb`, `PP_adam_eos_v2.ipynb` | Adam |
| `MJ_sgd_eos.ipynb`, `MJ_muon_eos.ipynb`, `MJ_adam_eos.ipynb` | SGD, Muon, Adam |
| `ZJ_muon_eos_batchsize.ipynb` | Muon batch size sweep |

## Repo layout

- **scripts/** — Python scripts to run each optimizer.
- **notebooks/** — Jupyter notebooks for the same experiments and extra analysis.
- **src/** — Shared code (models, data loading, training, sharpness).
- **output/eos/** — Experiment outputs (CSVs and run subfolders).
- **plots/** — Generated figures.

See [EOS_EXPERIMENT_README.md](EOS_EXPERIMENT_README.md) for a more detailed description of the experiment setup and outputs.

## DSMLP setup

If you are on DSMLP, use the scratch directory for the environment so PyTorch fits under the quota. Run this once:

```bash
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

After that, for each new session:

```bash
conda activate inv-nn-opt-env
cd ~/private/Investigating-NN-Optimizers
```

To refresh the environment after dependency changes:

```bash
conda activate base
cd ~/private/Investigating-NN-Optimizers
conda env update -n inv-nn-opt-env -f environment.yml --prune
```

## Further reading

Cohen, Jeremy, Simran Kaur, Yuanzhi Li, J. Zico Kolter, and Ameet Talwalkar. 2021. [*Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability*](https://arxiv.org/abs/2103.00065). CoRR abs/2103.00065.

Andreyev, Arseniy, and Pierfrancesco Beneventano. 2025. *Edge of Stochastic Stability: Revisiting the Edge of Stability for SGD.*

Cohen et al. 2024. *Adaptive Gradient Methods at the Edge of Stability.*

## Contributors

Notebooks and experiments are by project members. Initials in notebook and output folder names (ZJ, EC, PP, MJ) indicate who led each part.
