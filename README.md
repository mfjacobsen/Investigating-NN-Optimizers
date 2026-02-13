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

### Using functions.py
Load a 5k image subset of CIFAR-10 with 1k test images.
```python
X, y, X_test, y_test = functions.load_cifar_10()
```

Define model architecture for a fully-connected neural network. 2 hidden layers of
200 neurons each is large enough to investigate phenomenon while small enough to 
iterate quickly. This architecture should remain constant across models.
```python
input_size = X.shape[1] * X.shape[2] * X.shape[3]
num_hidden_layers = 2
hidden_layer_size = 200
```

Define the learning rate, max epochs, and accuracy of the model. Training stops
when training accuracy reaches the accuracy variable or epochs reaches max_epochs. 
Sharpness is computed every 1% of the way through max_epochs (if max epochs is 
2,000 then sharpness is computed every 20 epochs.) Balance max_epochs and accuracy
to ensure an appropriate number of sharpness computations exists in the output data
for plotting.
```python
learning_rate = 0.01
max_epochs = 20000
accuracy = 0.99
```

Initialize the model, criteria, and optimizer.
```python
model = models.FullyConnectedNet(
        input_size=input_size,
        num_hidden_layers=num_hidden_layers,
        hidden_layer_size=hidden_layer_size,
        num_labels=10,
        activation=nn.Tanh
    )
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
```

Train the model and record the training data. It's critical to ensure that you save
to the correct output directory. All of the file utility functions default to the 
main output directory and include the metadata and output data csv file names. When working with
output files, you only need to specify the subdirectory. For example, if you want 
to save your output files in output/eos/sgd_EC, then output_dir="eos/sgd_EC". If 
you want to load the output files from output/eos/adam_PP, then output_dir = "eos/sgd_PP":
```python
output_dir = "eos/muon_MJ"
functions.train_model(model, optimizer, criterion, epochs, accuracy, X, y, X_test, y_test, output_dir)
```

Load output data for plotting:
```python
md, out = functions.load_output_files(output_dir)
```

Another useful function for deleting model data from your output files if they
are getting too large, or want to trim specific models from the file.
```python
model_ids = [1,4,5]
delete_model_data(model_ids, output_dir)
```
