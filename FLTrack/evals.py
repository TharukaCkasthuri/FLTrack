"""
Copyright (C) [2023] [Tharuka Kasthuriarachchige]

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

Paper: [Clients Behavior Monitoring in Federated Learning via Eccentricity Analysis]
Published in: [IEEE International Conference on Evolving and Adaptive Intelligent Systems,
IEEE EAIS 2024 (23–24 May 2024, Madrid, Spain), 2024]
"""

import torch
import time
import collections

import numpy as np

from typing import Callable
from torch.autograd import grad

from sklearn.metrics import mean_squared_error, mean_absolute_error


def evaluate(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
) -> tuple:
    """
    Evaluate the model with validation dataset.Returns the average loss, mean squared error and mean absolute error.

    Parameters:
    -------------
    model: torch.nn.Module object;
        Model to be evaluated.
    dataloader: torch.utils.data.DataLoader object;
        Validation dataset.
    loss_fn: torch.nn.Module object;
        Loss function.

    Returns:
    -------------
    loss: float;
      Average loss.
    mse: float;
        Average mean squared error.
    mae: float;
        Average mean absolute error.

    """
    model.eval()
    loss, mse, mae = [], [], []

    for _, (x, y) in enumerate(dataloader):
        predicts = model(x)
        batch_loss = loss_fn(predicts, y)

        loss.append(batch_loss.item())
        batch_mse = mean_squared_error(y, np.squeeze(predicts.detach().numpy()))
        mse.append(batch_mse)
        batch_mae = mean_absolute_error(y, np.squeeze(predicts.detach().numpy()))
        mae.append(batch_mae)

    return sum(loss) / len(loss), sum(mse) / len(mse), sum(mae) / len(mae)


def evaluate_mae_with_confidence(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    num_bootstrap_samples: int = 250,
    confidence_level: float = 0.95,
) -> tuple:
    """
    Evaluate the model with validation dataset and calculate confidence intervals. Returns the average mean absolute error and confidence intervals.

    Parameters:
    -------------
    model: torch.nn.Module object;
        Model to be evaluated.
    dataloader: torch.utils.data.DataLoader object;
        Validation dataset.
    num_bootstrap_samples: int;
        Number of bootstrap samples.
    confidence_level: float;
        Confidence level. Default is 0.95.

    Returns:
    -------------
    avg_mae: float;
        Average mean absolute error.
    (lower_mae, upper_mae): tuple;
        Lower and upper bounds of the confidence interval for mean absolute error.
    """
    model.eval()
    mae_values = []

    for _, (x, y) in enumerate(dataloader):
        predicts = model(x)
        batch_mae = mean_absolute_error(y, predicts.detach().cpu().numpy())
        mae_values.append(batch_mae)

    avg_mae = np.mean(mae_values)

    bootstrap_mae = []

    for _ in range(num_bootstrap_samples):
        bootstrap_sample_indices = np.random.choice(
            len(dataloader.dataset), size=len(dataloader.dataset), replace=True
        )
        bootstrap_dataloader = torch.utils.data.DataLoader(
            dataloader.dataset,
            batch_size=dataloader.batch_size,
            sampler=torch.utils.data.SubsetRandomSampler(bootstrap_sample_indices),
        )

        mae_values = []

        for _, (x, y) in enumerate(bootstrap_dataloader):
            predicts = model(x)
            batch_mae = mean_absolute_error(y, predicts.detach().cpu().numpy())
            mae_values.append(batch_mae)

        bootstrap_mae.append(np.mean(mae_values))

    # Calculate confidence intervals
    confidence_interval = (1 - confidence_level) / 2
    sorted_mae = np.sort(bootstrap_mae)

    lower_mae = sorted_mae[int(confidence_interval * num_bootstrap_samples)]
    upper_mae = sorted_mae[int((1 - confidence_interval) * num_bootstrap_samples)]

    bootstrap_mae_std = np.std(bootstrap_mae)

    return avg_mae, (lower_mae, upper_mae), bootstrap_mae_std


def influence_with_mae(
    model: torch.nn.Module,
    influenced_model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
) -> float:
    """
    Calculate the influence of the model on the influenced model for the given validation set based on the difference of the mae.

    Parameters:
    -------------
    model: torch.nn.Module object;
        Model trained with all the clients.
    influenced_model: torch.nn.Module object;
        Model trained without a specific client.
    data_loader: torch.utils.data.DataLoader object;
        Validation dataset.

    Returns:
    -------------
    influence: float;
        Influence of the client on the federation.
    """

    model.eval()
    influenced_model.eval()

    influence = 0.0

    for _, (x, y) in enumerate(data_loader):
        model_pred = model(x)
        influenced_model_pred = influenced_model(x)

        batch_mae_model = mean_absolute_error(
            y, np.squeeze(model_pred.detach().numpy())
        )
        batch_mae_influenced_model = mean_absolute_error(
            y, np.squeeze(influenced_model_pred.detach().numpy())
        )

        influence += abs(batch_mae_model - batch_mae_influenced_model)

    return influence


def influence(
    model: torch.nn.Module,
    influenced_model: torch.nn.Module,
    val_set: torch.utils.data.DataLoader,
) -> float:
    """
    Calculate the influence of the model on the influenced model for the given validation set based on the prediction difference.

    Parameters:
    -------------
    model: torch.nn.Module object;
        Model trained with all the clients.
    influenced_model: torch.nn.Module object;
        Model trained without a specific client.
    data_loader: torch.utils.data.DataLoader object;
        Validation dataset.

    Returns:
    -------------
    influence: float;
        Influence of the model on the influenced model
    """

    model.eval()
    influenced_model.eval()

    total_influence = 0

    for x, y in val_set:
        abs_value = abs(model(x) - influenced_model(x))
        # abs_value = abs(mean_absolute_error(model(x).detach().numpy(),y) -  mean_absolute_error(influenced_model(x).detach().numpy(),y))
        total_influence += abs_value
    influence = total_influence / len(val_set)

    return influence


def calculate_hessian(model, loss_fn, data_loader) -> tuple:
    """
    Calculate the Hessian matrix of the model for the given validation set.

    Parameters:
    -------------
    model: torch.nn.Module object; model to be evaluated
    loss_fn: torch.nn.Module object; loss function
    data_loader: torch.utils.data.DataLoader object; validation dataset

    Returns:
    -------------
    hessian_matrix: torch.tensor object; Hessian matrix
    time: float; time taken to calculate the Hessian matrix
    """
    model.eval()

    total_loss = 0
    for _, (x, y) in enumerate(data_loader):
        predicts = model(x)
        y = y.view(-1, 1)
        batch_loss = loss_fn(predicts, y)
        total_loss += batch_loss
    avg_loss = total_loss / len(data_loader)

    start = time.time()

    # Calculate Jacobian w.r.t. model parameters
    grads = torch.autograd.grad(avg_loss, model.parameters(), create_graph=True)

    hessian_matrix = []

    for grad in grads:
        hessian_row = []
        for grad_element in grad.view(-1):
            # Compute second-order gradient
            hessian_element = torch.autograd.grad(
                grad_element, model.parameters(), retain_graph=True
            )
            hessian_row.append(hessian_element)
        hessian_matrix.append(hessian_row)

    hessian_matrix = torch.stack(hessian_matrix).squeeze()

    end = time.time()
    print("Calculation time of hessian matrix", (end - start) / 60)

    return grads, hessian_matrix, end - start


def calculate_hessian_flattened(model, loss_fn, data_loader) -> tuple:
    """
    Calculate the Hessian matrix of the model for the given validation set. And flatten the Jacobian matrix.

    Parameters:
    -------------
    model: torch.nn.Module object; model to be evaluated
    loss_fn: torch.nn.Module object; loss function
    data_loader: torch.utils.data.DataLoader object; validation dataset

    Returns:
    -------------
    hessian_matrix: torch.tensor object; Hessian matrix
    time: float; time taken to calculate the Hessian matrix
    """
    model.eval()

    total_loss = 0
    for _, (x, y) in enumerate(data_loader):
        predicts = model(x)
        y = y.view(-1, 1)
        batch_loss = loss_fn(predicts, y)
        total_loss += batch_loss
    avg_loss = total_loss / len(data_loader)

    # Allocate Hessian size
    num_param = sum(p.numel() for p in model.parameters())

    hessian_matrix = torch.zeros((num_param, num_param))

    start = time.time()
    # Calculate Jacobian w.r.t. model parameters
    J = grad(avg_loss, list(model.parameters()), create_graph=True)
    J = torch.cat([e.flatten() for e in J])  # flatten

    print("Calculation time of Gradients", time.time() - start)

    restart = time.time()
    for i in range(num_param):
        result = torch.autograd.grad(J[i], list(model.parameters()), retain_graph=True)
        hessian_matrix[i] = torch.cat([r.flatten() for r in result])  # flatten

    print("Calculation time of Hessian", (time.time() - start) / 60)

    return hessian_matrix


def calculate_hessian_flattened_optimized(
    model, loss_fn, data_loader, batch_size=64, device="cpu"
):
    model.eval()
    model.to(device)

    total_loss = 0
    J_batch = []

    start = time.time()

    for batch_idx, (x, y) in enumerate(data_loader):
        x, y = x.to(device), y.to(device)
        predicts = model(x)
        batch_loss = loss_fn(predicts, y)
        total_loss += batch_loss

        if (batch_idx + 1) % batch_size == 0:
            # Calculate Jacobian w.r.t. model parameters for the batch
            J_batch.extend(grad(batch_loss, model.parameters(), create_graph=True))

    J = torch.cat([e.flatten() for e in J_batch])

    print("Calculation time of Gradients: {:.2f} seconds".format(time.time() - start))

    num_param = sum(p.numel() for p in model.parameters())
    hessian_matrix = torch.zeros((num_param, num_param), device=device)

    start = time.time()

    for i in range(num_param):
        result = torch.autograd.grad(J[i], model.parameters(), retain_graph=True)
        hessian_matrix[i] = torch.cat([r.flatten() for r in result])

    print("Calculation time of Hessian: {:.2f} seconds".format(time.time() - start))

    return hessian_matrix


def calculate_hessian_gnewtons(model, loss_fn, data_loader) -> tuple:
    """
    Calculate the Hessian matrix of the model for the given dataset using gauss-newton approximation method.

    Parameters:
    -------------
    model: torch.nn.Module object; model to be evaluated
    loss_fn: torch.nn.Module object; loss function
    data_loader: torch.utils.data.DataLoader object; validation dataset

    Returns:
    -------------
    hessian_matrix: torch.tensor object; Hessian matrix
    """
    model.eval()

    total_loss = 0
    for _, (x, y) in enumerate(data_loader):
        predicts = model(x)
        y = y.view(-1, 1)
        batch_loss = loss_fn(predicts, y)
        total_loss += batch_loss
    avg_loss = total_loss / len(data_loader)

    start = time.time()
    # Calculate Jacobian w.r.t. model parameters
    J = grad(avg_loss, list(model.parameters()), create_graph=True)
    J = torch.cat([e.flatten() for e in J])  # flatten
    J_transpose = J.view(-1, 1)  # Reshape for transpose operation

    print("Calculation time of Gradients", time.time() - start)

    hessian_matrix = torch.mm(J_transpose, J.view(1, -1))

    print("Calculation time of Hessian", (time.time() - start) / 60)

    return hessian_matrix


def layer_importance(model, loss_fn, data_loader) -> dict:
    """
    Calculate the layer-wise importance scores for a neural network model using a data loader.

    Parameters:
    -----------
    model: torch.nn.Module
        The neural network model.
    loss_fn: callable
        The loss function used for training the model.
    data_loader: torch.utils.data.DataLoader
        A DataLoader object containing input-target pairs.

    Returns:
    --------
    layer_importance_scores: dict
        A dictionary containing the importance scores for each layer in the model.
    """
    model.eval()
    layer_importance_scores = {}
    total_importance_score = 0.0

    # Iterate through the layers of the model
    for layer_name, layer in model.track_layers.items():
        total_layer_importance = 0.0

        for batch_inputs, batch_targets in data_loader:
            model.zero_grad()

            outputs = model(batch_inputs)
            batch_loss = loss_fn(outputs, batch_targets)
            batch_loss.backward(retain_graph=True)

            importance_score = 0.0
            for param in layer.parameters():
                importance_score += torch.sum(torch.abs(param.grad)).item()

            total_layer_importance += importance_score

        avg_importance_score = total_layer_importance / len(data_loader)
        layer_importance_scores[layer_name] = avg_importance_score
        total_importance_score += avg_importance_score

    # Scale the importance scores to percentages out of 100
    for layer_name in layer_importance_scores:
        layer_importance_scores[layer_name] = (
            layer_importance_scores[layer_name] / total_importance_score
        ) * 100

    return layer_importance_scores


def layer_importance_bias(model, loss_fn, data_loader) -> dict:
    """
    Calculate the layer-wise importance scores for a neural network model using a data loader.

    Parameters:
    -----------
    model: torch.nn.Module
        The neural network model.
    loss_fn: callable
        The loss function used for training the model.
    data_loader: torch.utils.data.DataLoader
        A DataLoader object containing input-target pairs.

    Returns:
    --------
    layer_importance_scores: dict
        A dictionary containing the importance scores for each layer in the model as percentages.
    """
    model.eval()
    layer_importance_scores = {}
    total_importance_score = 0.0

    for layer_name, layer in model.named_parameters():
        total_layer_importance = 0.0

        for batch_inputs, batch_targets in data_loader:
            model.zero_grad()

            outputs = model(batch_inputs)
            batch_loss = loss_fn(outputs, batch_targets)
            batch_loss.backward(retain_graph=True)

            # Importance score
            importance_score = torch.sum(torch.abs(layer.grad)).item()
            total_layer_importance += importance_score

        avg_layer_importance = total_layer_importance / len(data_loader)
        layer_importance_scores[layer_name] = avg_layer_importance
        total_importance_score += avg_layer_importance

    # Scale the importance scores to percentages out of 100
    for layer_name in layer_importance_scores:
        layer_importance_scores[layer_name] = (
            layer_importance_scores[layer_name] / total_importance_score
        ) * 100

    return layer_importance_scores


def calculate_contribution(local_models, model_layers):
    # Initialize a dictionary to store the contributions
    contributions = {}

    # Get the state dicts for all local models
    state_dicts = [model.state_dict() for model in local_models]

    for key in model_layers:
        weight_contributions = torch.stack(
            [item[str(key) + ".weight"] for item in state_dicts]
        )
        bias_contributions = torch.stack(
            [item[str(key) + ".bias"] for item in state_dicts]
        )

        # Calculate the mean contributions for weight and bias
        mean_weight_contribution = weight_contributions.mean(dim=0)
        mean_bias_contribution = bias_contributions.mean(dim=0)

        # Store the contributions in a dictionary
        contributions[key] = {
            "weight": mean_weight_contribution,
            "bias": mean_bias_contribution,
        }

    return contributions


# Example usage:
# local_models = [local_model1, local_model2, ...]  # List of local models
# model_layers = ['layer1', 'layer2', ...]  # List of layers to consider
# contributions = calculate_contribution(local_models, model_layers)
