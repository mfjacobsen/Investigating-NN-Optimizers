from . import seed
from . import models
import src.functions as functions

import torch
import torch.nn as nn

# Set device and generator from seed module
device = seed.device
generator = seed.generator

# Load CIFAR-10 data
X, y, X_test, y_test = functions.load_cifar_10()

output_dir_sgd = "eos/sgd_MJ"
output_dir_sgd_mom = "eos/sgd_momentum_MJ"
output_dir_rmsprop = "eos/rmsprop_MJ"
output_dir_adam = "eos/adam_MJ"

# Define model parameters
input_size = X.shape[1] * X.shape[2] * X.shape[3]
num_hidden_layers = 2
hidden_layer_size = 200

# Gradient descent and MSE loss
epochs = 20000
learning_rates = [0.13, 0.07, 0.04]
accuracy = 0.99

for learning_rate in learning_rates:

    model = models.FullyConnectedNet(
        input_size=input_size,
        num_hidden_layers=num_hidden_layers,
        hidden_layer_size=hidden_layer_size,
        num_labels=10,
        activation=nn.Tanh
    )
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    functions.train_model(model, optimizer, criterion, epochs, accuracy, X, y, X_test, y_test, output_dir_sgd)

# Gradient descent and Cross-Entropy loss
epochs = 4000
learning_rates  = [0.032, 0.025, 0.02]
accuracy = 0.999
for learning_rate in learning_rates:

    model = models.FullyConnectedNet(
        input_size=input_size,
        num_hidden_layers=num_hidden_layers,
        hidden_layer_size=hidden_layer_size,
        num_labels=10,
        activation=nn.Tanh
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    functions.train_model(model, optimizer, criterion, epochs, accuracy, X, y, X_test, y_test, output_dir_sgd)



# Gradient descent with momentum and MSE loss
epochs = 8000
learning_rates = [0.14, 0.12, 0.1]
momentum = 0.9
accuracy = 0.9999

for learning_rate in learning_rates:

    model = models.FullyConnectedNet(
        input_size=input_size,
        num_hidden_layers=num_hidden_layers,
        hidden_layer_size=hidden_layer_size,
        num_labels=10,
        activation=nn.Tanh
    )
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    functions.train_model(model, optimizer, criterion, epochs, accuracy, X, y, X_test, y_test, output_dir_sgd_mom)

# Gradient descent with momentum and Cross-Entropy loss
epochs = 500
learning_rates  = [0.10, 0.06, 0.04]
momentum = 0.9
accuracy = 0.9999

for learning_rate in learning_rates:

    model = models.FullyConnectedNet(
        input_size=input_size,
        num_hidden_layers=num_hidden_layers,
        hidden_layer_size=hidden_layer_size,
        num_labels=10,
        activation=nn.Tanh
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    functions.train_model(model, optimizer, criterion, epochs, accuracy, X, y, X_test, y_test, output_dir_sgd_mom)




# RMSprop and MSE loss
epochs = 4000
learning_rates = [0.00005, 0.00002, 0.00001]
accuracy = 0.99

for learning_rate in learning_rates:

    model = models.FullyConnectedNet(
        input_size=input_size,
        num_hidden_layers=num_hidden_layers,
        hidden_layer_size=hidden_layer_size,
        num_labels=10,
        activation=nn.Tanh
    )
    criterion = nn.MSELoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)

    functions.train_model(model, optimizer, criterion, epochs, accuracy, X, y, X_test, y_test, output_dir_rmsprop)

# Train model with RMSprop and Cross-Entropy loss
epochs = 2000
learning_rates = [0.00005, 0.00002, 0.00001]
accuracy = 0.9999

for learning_rate in learning_rates:

    model = models.FullyConnectedNet(
        input_size=input_size,
        num_hidden_layers=num_hidden_layers,
        hidden_layer_size=hidden_layer_size,
        num_labels=10,
        activation=nn.Tanh
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)

    functions.train_model(model, optimizer, criterion, epochs, accuracy, X, y, X_test, y_test, output_dir_rmsprop)



# Define model parameters
input_size = X.shape[1] * X.shape[2] * X.shape[3]
num_hidden_layers = 2
hidden_layer_size = 200

# Gradient descent and MSE loss
epochs = 1500
learning_rates = [6e-4, 2e-4, 1e-4]
accuracy = 1.1

for learning_rate in learning_rates:

    model = models.FullyConnectedNet(
        input_size=input_size,
        num_hidden_layers=num_hidden_layers,
        hidden_layer_size=hidden_layer_size,
        num_labels=10,
        activation=nn.Tanh
    )
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    functions.train_model(model, optimizer, criterion, epochs, accuracy, X, y, X_test, y_test, output_dir_adam)

# Define model parameters
input_size = X.shape[1] * X.shape[2] * X.shape[3]
num_hidden_layers = 2
hidden_layer_size = 200

# Gradient descent and Cross-Entropy loss
epochs = 300
learning_rates = [6e-3, 2e-3, 1e-3]
accuracy = 1.1

for learning_rate in learning_rates:

    model = models.FullyConnectedNet(
        input_size=input_size,
        num_hidden_layers=num_hidden_layers,
        hidden_layer_size=hidden_layer_size,
        num_labels=10,
        activation=nn.Tanh
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    functions.train_model(model, optimizer, criterion, epochs, accuracy, X, y, X_test, y_test, output_dir_adam)

functions.generate_gd_quadratic_plot()

model_ids_mse = [1, 2, 3]
model_ids_ce = [4, 5, 6]

md, out = functions.load_output_files(output_dir_sgd)
functions.plot_sgd_fcnn_data(md, out, model_ids_mse, model_ids_ce, save=True)

md, out = functions.load_output_files(output_dir_sgd_mom)
functions.plot_sgdm_fcnn_data(md, out, model_ids_mse, model_ids_ce, save=True)

md, out = functions.load_output_files(output_dir_rmsprop)
functions.plot_rmsprop_fcnn_data(md, out, model_ids_mse, model_ids_ce, save=True)

md, out = functions.load_output_files(output_dir_adam)
functions.plot_adam_fcnn_data(md, out, model_ids_mse, model_ids_ce, save=True)