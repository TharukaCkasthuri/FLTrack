import torch
import time
import collections

import numpy as np

from typing import Callable
from torch.autograd import grad

from sklearn.metrics import mean_squared_error, mean_absolute_error
from utils import get_all_possible_pairs


def evaluate(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
) -> tuple:
    """
    Evaluate the model with validation dataset.Returns the average loss, mean squared error and mean absolute error.

    Parameters:
    -------------
    model: torch.nn.Module object; model to be evaluated
    dataloader: torch.utils.data.DataLoader object; validation dataset
    loss_fn: torch.nn.Module object; loss function

    Returns:
    -------------
    loss: float; average loss
    mse: float; average mean squared error
    mae: float; average mean absolute error

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
):
    """
    Evaluate the model with validation dataset and calculate confidence intervals.Returns the average mean absolute error and confidence intervals.

    Parameters:
    -------------
    model: torch.nn.Module object; model to be evaluated
    dataloader: torch.utils.data.DataLoader object; validation dataset
    num_bootstrap_samples: int; number of bootstrap samples
    confidence_level: float; confidence level

    Returns:
    -------------
    avg_mae: float; average mean absolute error
    (lower_mae, upper_mae): tuple; lower and upper bounds of the confidence interval for mean absolute error
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


def influence_with_mae(model, influenced_model, data_loader) -> tuple:
    """
    Calculate the influence of the model on the influenced model for the given validation set.

    Parameters:
    -------------
    model: torch.nn.Module object; model to be evaluated
    influenced_model: torch.nn.Module object; model to be evaluated
    data_loader: torch.utils.data.DataLoader object; validation dataset

    Returns:
    -------------
    model_predictions: list; predictions of the model
    influence_model_predictions: list; predictions of the influenced model
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


def influence(model, influenced_model, val_set) -> float:
    """
    Calculate the influence of the model on the influenced model for the given validation set.

    Parameters:
    -------------
    model: torch.nn.Module object; model to be evaluated
    influenced_model: torch.nn.Module object; model to be evaluated
    val_set: torch.utils.data.DataLoader object; validation dataset

    Returns:
    -------------
    influence: float; influence of the model on the influenced model
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
    x: torch.tensor object; tensor 1
    y: torch.tensor object; tensor 2

    Returns:
    -------------
    distance: torch.tensor object; pairwise Euclidean distance between two tensors
    """

    distance = torch.cdist(x, y, p=2)

    return distance


def euclidean_distance(x: torch.tensor, y: torch.tensor) -> torch.tensor:
    """Calculate the Euclidean distance between two matrices.

    Parameters:
    -------------
    x: torch.tensor object; tensor 1
    y: torch.tensor object; tensor 2

    Returns:
    -------------
    distance: torch.tensor object; Euclidean distance between two matrices
    """

    # Calculate the squared Euclidean distance element-wise
    squared_distance = torch.pow(x - y, 2).sum(dim=1)
    distance = torch.sqrt(squared_distance)

    return distance


def manhattan_distance(x: torch.tensor, y: torch.tensor) -> torch.tensor:
    """Calculate the Manhattan distance between two matrices.

    Parameters:
    -------------
    x: torch.tensor object; tensor 1
    y: torch.tensor object; tensor 2

    Returns:
    -------------
    distance: torch.tensor object; Manhattan distance between two matrices
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
    Calculate the accumulated proximity between two tensors.

    Parameters:
    -------------
    x: torch.tensor object; tensor 1
    y: torch.tensor object; tensor 2

    Returns:
    -------------
    proximity: torch.tensor object; accumulated proximity between two tensors
    """

    distances = distance_matrix(x, y)
    proximity = torch.sum(distances)

    return proximity


def layerwise_proximity(
    x: collections.OrderedDict,
    y: collections.OrderedDict,
    critarian: str,
    distance_matrix,
):
    """
    Calculate the layer-wise proximity between state dictionaries based on a critarian (layer bias or weights).

    Parameters:
    -------------
    x: collections.OrderedDict; state dictionary 1
    y: collections.OrderedDict; state dictionary 2
    critarian: str; critarian to be evaluated, basically the layer name and specifying the weight or bias.
    distance_matrix: Callable; distance matrix

    Returns:
    -------------
    proximity: torch.tensor object; layer-wise proximity between state dictionaries
    """
    if critarian.split(".")[-1] == "bias":
        proximity = accumulated_proximity(
            x[critarian].view(1, -1), y[critarian].view(1, -1), distance_matrix
        )
    else:
        proximity = accumulated_proximity(x[critarian], y[critarian], distance_matrix)

    return proximity


def full_accumulated_proximity(
    clients: list, distance_matrix: Callable
) -> torch.tensor:
    """
    Calculate the full accumulated proximity between all clients.

    Parameters:
    -------------
    clients: list; list of clients

    Returns:
    -------------
    total_proximity: torch.tensor object; full accumulated proximity between all clients
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
