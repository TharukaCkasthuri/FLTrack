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

import torch.nn as nn
import torch.nn.init as init


class ShallowNN(nn.Module):
    """
    Shallow Neural Network model with three layers.

    Parameters:
    ------------
    feats: int; number of features
    """

    def __init__(self, feats)-> None:
        super(ShallowNN, self).__init__()
        self.layer_1 = nn.Linear(feats, 32)
        self.bn_1 = nn.BatchNorm1d(32)
        self.relu_1 = nn.ReLU()
        self.layer_2 = nn.Linear(32, 16)
        self.relu_2 = nn.ReLU()
        self.layer_3 = nn.Linear(16, 1)
        self.track_layers = {
            "layer_1": self.layer_1,
            "bn_1": self.bn_1,
            "layer_2": self.layer_2,
            "layer_3": self.layer_3,
        }

        # Initialize weights using He initialization
        for layer in [self.layer_1, self.layer_2, self.layer_3]:
            if isinstance(layer, nn.Linear):
                init.kaiming_uniform_(layer.weight, mode="fan_in", nonlinearity="relu")
                init.constant_(layer.bias, 0.0)

    def forward(self, inputs)-> torch.tensor:
        """
        Forward pass.

        Parameters:
        ------------
        inputs: torch.tensor object; input data

        Returns:
        ------------
        x: torch.tensor object; output data
        """
        x = self.relu_1(self.bn_1(self.layer_1(inputs)))
        x = self.relu_2(self.layer_2(x))
        x = self.layer_3(x)
        return x
