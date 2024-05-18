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

from tqdm import tqdm
from torch.utils.data import DataLoader


class Client:
    """
    Client class for federated learning.

    Parameters:
    ------------
    client_id: str; client id
    train_dataset: torch.utils.data.Dataset object; training dataset
    test_dataset: torch.utils.data.Dataset object; validation dataset
    batch_size: int; batch size
    """

    def __init__(
        self,
        client_id: str,
        train_dataset: object,
        test_dataset: object,
        batch_size: int,
        learning_rate: float,
        weight_decay: float,
        local_model: object = None,
    ) -> None:

        self.client_id: str = client_id
        self.batch_size = batch_size

        self.traindl = DataLoader(
            train_dataset, batch_size, shuffle=True, drop_last=True
        )
        self.valdl = DataLoader(test_dataset, batch_size, shuffle=True, drop_last=True)
        self.optimizer = torch.optim.SGD(
            local_model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.local_model = local_model

    def set_model(self, model_weights) -> None:
        """
        Set the model for the client.

        Parameters:
        ------------
        model: torch.nn.Module object; model
        """
        self.local_model.load_state_dict(model_weights)

    def get_model(self) -> object:
        """
        Get the model of the client.

        Parameters:
        ------------
        None

        Returns:
        ------------
        model: torch.nn.Module object; model
        """
        return self.local_model

    def train(self, loss_fn, epochs, global_round=None) -> tuple:
        """
        Training the model.

        Parameters:
        ------------
        model: torch.nn.Module object; model to be trained
        loss_fn: torch.nn.Module object; loss function
        optimizer: torch.optim object; optimizer
        epoch: int; epoch number

        Returns:
        ------------
        model: torch.nn.Module object; trained model
        loss_avg: float; average loss
        """

        if global_round:
            print(
                f"Global Round: {global_round} \tClient: {self.client_id} Started it's local training"
            )
        else:
            print(f"Client: {self.client_id} Started it's local training")

        train_losses = []
        for epoch in range(epochs):
            print("\n")
            batch_loss = []
            for batch_idx, (x, y) in enumerate(self.traindl):
                outputs = self.local_model(x)
                y = y.view(-1, 1)
                loss = loss_fn(outputs, y)
                self.local_model.zero_grad()
                loss.backward()
                self.optimizer.step()

                # if batch_idx % 50 == 0:
                #    print(
                #        f"Epoch: {epoch + 1} \tClient ID: {self.client_id} \t[{batch_idx * len(x)}/{len(self.traindl.dataset)} ({100.0 * batch_idx / len(self.traindl):.0f}%)] \tLoss: {loss.item():.6f}"
                #    )

                batch_loss.append(loss.item())

            loss_avg = sum(batch_loss) / len(batch_loss)
            train_losses.append(loss_avg)

            print(
                f"Client: {self.client_id} \tEpoch: {epoch + 1} \tAverage Training Loss: {loss_avg}"
            )

        validation_loss = self.eval(loss_fn)
        print(f"Client: {self.client_id} \tValidation Loss: {validation_loss}")

        return self.local_model, train_losses, validation_loss

    def eval(self, loss_fn) -> float:
        """
        Evaluate the model with validation dataset.

        Parameters:
        ------------
        model: torch.nn.Module object; model to be evaluated
        loss_fn: torch.nn.Module object; loss function

        Returns:
        ------------
        loss_avg: float; average loss
        """
        batch_loss = []
        for _, (x, y) in enumerate(self.valdl):
            outputs = self.local_model(x)
            y = y.view(-1, 1)
            loss = loss_fn(outputs, y)
            batch_loss.append(loss.item())
        loss_avg = sum(batch_loss) / len(batch_loss)

        return loss_avg
