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

from hparams import HParams


model_hparams = {
    "c1": HParams(
        batch_size=256,
        num_epochs=250,
        learning_rate=0.001,
        weight_decay=0.01,
    ),
    "c2": HParams(
        batch_size=256,
        num_epochs=250,
        learning_rate=0.01,
        weight_decay=0.001,
    ),
    "c3": HParams(
        batch_size=256,
        num_epochs=250,
        learning_rate=0.01,
        weight_decay=0.001,
    ),
    "c4": HParams(
        batch_size=256,
        num_epochs=250,
        learning_rate=0.01,
        weight_decay=0.001,
    ),
    "c5": HParams(
        batch_size=256,
        num_epochs=250,
        learning_rate=0.01,
        weight_decay=0.001,
    ),
    "c6": HParams(
        batch_size=256,
        num_epochs=250,
        learning_rate=0.01,
        weight_decay=0.001,
    ),
    "c7": HParams(
        batch_size=256,
        num_epochs=250,
        learning_rate=0.01,
        weight_decay=0.0001,
    ),
    "c8": HParams(
        batch_size=256,
        num_epochs=250,
        learning_rate=0.01,
        weight_decay=0.001,
    ),
    "c9": HParams(
        batch_size=256,
        num_epochs=250,
        learning_rate=0.01,
        weight_decay=0.0001,
    ),
    "c10": HParams(
        batch_size=256,
        num_epochs=250,
        learning_rate=0.001,
        weight_decay=0.01,
    ),
    "c11": HParams(
        batch_size=256,
        num_epochs=250,
        learning_rate=0.01,
        weight_decay=0.001,
    ),
    "c12": HParams(
        batch_size=256,
        num_epochs=250,
        learning_rate=0.01,
        weight_decay=0.001,
    ),
    "c13": HParams(
        batch_size=256,
        num_epochs=250,
        learning_rate=0.01,
        weight_decay=0.001,
    ),
    "c14": HParams(
        batch_size=256,
        num_epochs=250,
        learning_rate=0.01,
        weight_decay=0.0001,
    ),
    "c15": HParams(
        batch_size=256,
        num_epochs=250,
        learning_rate=0.01,
        weight_decay=0.0001,
    ),
    "c16": HParams(
        batch_size=256,
        num_epochs=250,
        learning_rate=0.01,
        weight_decay=0.0001,
    ),
    "c17": HParams(
        batch_size=256,
        num_epochs=250,
        learning_rate=0.01,
        weight_decay=0.0001,
    ),
    "c18": HParams(
        batch_size=256,
        num_epochs=250,
        learning_rate=0.001,
        weight_decay=0.0001,
    ),
    "c19": HParams(
        batch_size=256,
        num_epochs=250,
        learning_rate=0.01,
        weight_decay=0.001,
    ),
    "c20": HParams(
        batch_size=256,
        num_epochs=250,
        learning_rate=0.01,
        weight_decay=0.0001,
    ),
    "c21": HParams(
        batch_size=256,
        num_epochs=250,
        learning_rate=0.01,
        weight_decay=0.001,
    ),
    "c22": HParams(
        batch_size=256,
        num_epochs=250,
        learning_rate=0.01,
        weight_decay=0.0001,
    ),
    "c23": HParams(
        batch_size=256,
        num_epochs=250,
        learning_rate=0.001,
        weight_decay=0.0001,
    ),
    "c24": HParams(
        batch_size=256,
        num_epochs=250,
        learning_rate=0.001,
        weight_decay=0.001,
    ),
}
