"""
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.

If you use this software in your research, please consider citing our paper:
[Author(s). "Title of Your Paper." Journal or Conference Name, Year]
"""

import torch
import time
import collections

import numpy as np

from typing import Callable
from torch.autograd import grad

from sklearn.metrics import mean_squared_error, mean_absolute_error
from FLTrack.utils import get_all_possible_pairs


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


def pairwise_euclidean_distance(x: torch.tensor, y: torch.tensor) -> torch.tensor:
    """
    Calculate the pairwise Euclidean distance between two tensors. using torch.cdist

    Parameters:
    -------------
    x: torch.tensor object;
        Tensor 1.
    y: torch.tensor object;
        Tensor 2.

    Returns:
    -------------
    distance: torch.tensor object;
        Pairwise Euclidean distance between two tensors.
    """

    distance = torch.cdist(x, y, p=2)

    return distance


def euclidean_distance(x: torch.tensor, y: torch.tensor) -> torch.tensor:
    """Calculate the Euclidean distance between two matrices.

    Parameters:
    -------------
    x: torch.tensor object;
        Tensor 1.
    y: torch.tensor object;
        Tensor 2.

    Returns:
    -------------
    distance: torch.tensor object;
        Euclidean distance between two matrices.
    """

    # Calculate the squared Euclidean distance element-wise
    squared_distance = torch.pow(x - y, 2).sum(dim=1)
    distance = torch.sqrt(squared_distance)

    return distance


def manhattan_distance(x: torch.tensor, y: torch.tensor) -> torch.tensor:
    """Calculate the Manhattan distance between two matrices.

    Parameters:
    -------------
    x: torch.tensor object;
        Tensor 1.
    y: torch.tensor object;
        Tensor 2.

    Returns:
    -------------
    distance: torch.tensor object;
        Manhattan distance between two matrices
    """

    # Calculate the absolute difference element-wise
    absolute_difference = torch.abs(x - y)
    # Sum the absolute differences along the appropriate dimension
    distance = absolute_difference.sum(dim=1)

    return distance


def accumulated_proximity(
    x: torch.tensor, y: torch.tensor, distance_matrix: Callable
) -> torch.tensor:
    """
    Calculate the accumulated proximity between two tensors. These tensors expected to be the same size.

    Parameters:
    -------------
    x: torch.tensor object;
        Tensor 1.
    y: torch.tensor object;
        Tensor 2.

    Returns:
    -------------
    proximity: torch.tensor object;
        Accumulated proximity between two tensors.
    """

    distances = distance_matrix(x, y)
    proximity = torch.sum(distances)

    return proximity


def hessian_eccentricity(hess_matrix_dict: dict, distance_matrix: Callable) -> dict:
    """
    Calculate the eccentricity between models (model is represented as dictinary of Hessian matrices).

    Parameters:
    -------------
    hess_matrix_dict: dict;
        Dictionary of Hessian matrices.
    distance_matrix: Callable;
        Distance matrix.

    Returns:
    -------------
    eccentricity_dict: dict;
        Eccentricity between models.
    """
    full_proximity = 0.0
    prox_dict = {}

    # Calculate proximities and cache results
    prox_cache = {}

    for i in hess_matrix_dict:
        total_prox = 0.0

        for j in hess_matrix_dict:
            if (i, j) in prox_cache:
                prox = prox_cache[(i, j)]
            else:
                prox = accumulated_proximity(
                    hess_matrix_dict[i], hess_matrix_dict[j], distance_matrix
                )
                prox_cache[(i, j)] = prox

            total_prox += prox

        prox_dict[i] = total_prox
        full_proximity += total_prox

    eccentricity_dict = {
        key: round((value / full_proximity).item(), 4)
        for key, value in prox_dict.items()
    }

    return eccentricity_dict


def layerwise_proximity(
    x: collections.OrderedDict,
    y: collections.OrderedDict,
    critarian: str,
    distance_matrix: Callable,
) -> torch.tensor:
    """
    Calculate the layer-wise proximity between state dictionaries based on a critarian (layer bias or weights).

    Parameters:
    -------------
    x: collections.OrderedDict;
        State dictionary 1.
    y: collections.OrderedDict;
        State dictionary 2.
    critarian: str;
        Critarian to be evaluated, basically the layer name and specifying the weight or bias.
    distance_matrix: Callable;
        Distance matrix.

    Returns:
    -------------
    proximity: torch.tensor object;
        Layer-wise proximity between state dictionaries.
    """
    if critarian.split(".")[-1] == "bias":
        proximity = accumulated_proximity(
            x[critarian].view(1, -1), y[critarian].view(1, -1), distance_matrix
        )
    else:
        proximity = accumulated_proximity(x[critarian], y[critarian], distance_matrix)

    return proximity


def layerwise_eccentricity(state_dicts: dict, criterian: str, distance_matrix) -> dict:
    """
    Calculate the layer-wise eccentricity between models (dictinary of state dicts) based on a critarian (layer bias or weights).

    Parameters:
    -------------
    state_dicts: dict;
        Dictionary of state dicts.
    criterian: str;
        Critarian to be evaluated, basically the layer name and specifying the weight or bias.
    distance_matrix: Callable;
        Distance matrix.

    Returns:
    -------------
    eccentricity_dict: dict;
        Layer-wise eccentricity between models.
    """

    total_proximity = 0.0
    prox_dict = {}
    for i in state_dicts:
        item_prox = 0.0
        for j in state_dicts:
            prox = layerwise_proximity(
                state_dicts[i], state_dicts[j], criterian, distance_matrix
            )
            item_prox += prox

        prox_dict[i] = item_prox.item()
        total_proximity += item_prox

    eccentricity_dict = {
        key: (value / total_proximity).item() for key, value in prox_dict.items()
    }

    return eccentricity_dict


def full_accumulated_proximity(
    clients: list, distance_matrix: Callable
) -> torch.tensor:
    """
    Calculate the full accumulated proximity between all clients.

    Parameters:
    -------------
    clients: list;
        List of clients.

    Returns:
    -------------
    total_proximity: torch.tensor object;
        Full accumulated proximity between all clients
    """
    matrix_dict = {
        key: {"iso_matrix": torch.load("hessians/iso/" + str(key) + ".pth")}
        for key in clients
    }

    total_proximity = 0
    # possible_pairs = get_all_possible_pairs(clients)
    for l in clients:
        for i in clients:
            acc_prox = accumulated_proximity(
                matrix_dict[i]["iso_matrix"],
                matrix_dict[l]["iso_matrix"],
                distance_matrix,
            )
            total_proximity += acc_prox

    return total_proximity


def layerwise_full_accumulated_proximity(
    clients: list, criterian: str, distance_matrix: Callable
) -> tuple:
    """
    Calculate the layer-wise full accumulated proximity between all clients.

    Parameters:
    -------------
    clients: list; list of clients
    criterian: str; criterian to be evaluated, basically the layer name.
    distance_matrix: Callable; distance matrix

    Returns:
    -------------
    total_weight_proximity: float; total weight proximity
    total_bias_proximity: float; total bias proximity
    """
    model_dict = {
        key: torch.load("checkpt/isolated/batch64_client_" + str(key) + ".pth")
        for key in clients
    }

    possible_pairs = get_all_possible_pairs(clients)

    total_weight_proximity = 0.0
    total_bias_proximity = 0.0

    for pair in possible_pairs:
        weight_prox = accumulated_proximity(
            model_dict[pair[0]][criterian + str(".weight")],
            model_dict[pair[1]][criterian + str(".weight")],
            distance_matrix,
        )

        bias_prox = accumulated_proximity(
            model_dict[pair[0]][criterian + str(".bias")].view(1, -1),
            model_dict[pair[1]][criterian + str(".bias")].view(1, -1),
            distance_matrix,
        )

        total_weight_proximity += weight_prox
        total_bias_proximity += bias_prox

    return total_weight_proximity, total_bias_proximity


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

    return hessian_matrix, (time.time() - start) / 60


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
