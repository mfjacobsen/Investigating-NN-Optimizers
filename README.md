# Investigating-NN-Optimizers

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

Results are saved to `output/eos/` in subdirectories named by optimizer and author initials (e.g. `output/eos/shampoo_ZJ/`, `output/eos/sgd_EC/`, `output/eos/adam_PP/`).