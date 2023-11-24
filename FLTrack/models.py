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

Paper: [Title of Your Paper]
Published in: [Journal/Conference Name]
"""

import torch


class ShallowNN(torch.nn.Module):
    """
    Shallow Nueral Network model with three layers.

    Parameters:
    ------------
    feats: int; number of features
    """

    def __init__(self, feats):
        super(ShallowNN, self).__init__()
        self.layer_1 = torch.nn.Linear(feats, 64)
        self.relu_1 = torch.nn.ReLU()
        self.layer_2 = torch.nn.Linear(64, 32)
        self.relu_2 = torch.nn.ReLU()
        self.layer_3 = torch.nn.Linear(32, 1)
        self.track_layers = {
            "layer_1": self.layer_1,
            "layer_2": self.layer_2,
            "layer_3": self.layer_3,
        }

    def forward(self, inputs):
        """
        Forward pass.

        Parameters:
        ------------
        inputs: torch.tensor object; input data

        Returns:
        ------------
        x: torch.tensor object; output data
        """
        x = self.relu_1(self.layer_1(inputs))
        x = self.relu_2(self.layer_2(x))
        x = self.layer_3(x)
        return x


### When the feats = 197,
