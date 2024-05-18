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
import itertools

import numpy as np

from typing import Callable

def get_all_possible_pairs(client_ids: list) -> list:
    """
    Returns all possible pairs of client ids.

    Parameters:
    --------
    client_ids: list; list of client ids

    Returns:
    --------
    pairs: list; list of all possible pairs of client ids
    """

    pairs = list(itertools.combinations(client_ids, 2))

    return pairs

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
        print(total_prox)
        prox_dict[i] = total_prox
        full_proximity += total_prox

    eccentricity_dict = {}

    for key, value in prox_dict.items():
        eccentricity_dict[key] = round((value / full_proximity).item(), 4)

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
    if len(x[critarian].shape) == 1:
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
