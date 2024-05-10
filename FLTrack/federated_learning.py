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

import time
import torch
import argparse
import os

import pandas as pd
from tqdm import tqdm

from clients import Client
from utils import get_device
from evals import evaluate
from models import ShallowNN
from params import model_hparams

from torch.utils.tensorboard import SummaryWriter


class Federation:
    """
    Class for federated learning.

    Parameters:
    ------------
    checkpt_path: str;
        Path to save the model
    features: int;
        Number of features in the input data.
    loss_fn: torch.nn.Module object;
        The loss function used for training.
    batch_size: int;
        The batch size for training.
    learning_rate: float;
        The learning rate for the optimizer.
    rounds : int
        The number of training rounds in federated learning.
    epochs_per_round : int
        The number of training epochs per round.

    Methods:
    ----------
    __init__(self, checkpt_path: str, features: int, loss_fn, batch_size, learning_rate, rounds, epochs_per_round):
        Initializes a Federation instance with the specified parameters.

    set_clients(self, client_ids: list) -> None:
        Sets up the clients for federated learning.

    train(self, model, summery=False) -> tuple:
        Trains the model using federated learning.

    save_stats(self, model, training_stat: list) -> None:
        Saves the training statistics and the model.
    """

    def __init__(
        self,
        checkpt_path: str,
        features: int,
        loss_fn: torch.nn.Module,
        global_rounds: int,
        local_rounds: int,
        save_ckpt: bool = False,
        log_summary: bool = False,
    ) -> None:
        self.checkpt_path = checkpt_path
        self.features = features
        self.loss_fn = loss_fn
        self.global_rounds = global_rounds
        self.local_rounds = local_rounds
        self.save_ckpt = save_ckpt
        self.log_summary = log_summary

        # self.writer = SummaryWriter(comment="_fed_train_batch" + str(batch_size))

    def set_clients(self, client_ids: list) -> None:
        """
        Setting up the clients for federated learning.

        Parameters:
        ----------------
        client_ids: list;
            List of client ids

        Returns:
        ----------------
        None
        """
        self.client_ids = client_ids
        self.clients = [
            Client(
                id,
                torch.load(f"../trainpt/{id}.pt"),
                torch.load(f"../testpt/{id}.pt"),
                model_hparams[f"{id}"]["batch_size"],
                model_hparams[f"{id}"]["learning_rate"],
                model_hparams[f"{id}"]["weight_decay"],
                local_model=ShallowNN(self.features),
            )
            for id in client_ids
        ]
        self.client_dict = {
            client_id: {"training_loss": [], "validation_loss": []}
            for client_id in client_ids
        }

    def train(
        self,
        model: torch.nn.Module,
        log_summary: bool = False,
    ) -> tuple:
        """
        Training the model.

        Parameters:
        ----------------
        model:torch.nn.Module;
            Model to be trained. The model should have a track_layers attribute which is a dictionary of the layers to be trained.
        summery: bool;
            Whether to save the training stats and the model. Default is False.

        Returns:
        ----------------
        global_model:torch.nn.Module;
            Trained model
        training_stats: list;
            Training stattistics as a list of dictionaries
        """

        # initiate global model
        global_model = model
        global_model.train()

        global_weights = global_model.state_dict()
        model_layers = global_model.track_layers.keys()

        for round in tqdm(range(self.global_rounds)):
            local_models = []

            print(f"\n | Global Training Round : {round+1} |\n")

            global_model.load_state_dict(global_weights)

            for client in self.clients:
                client_id = client.client_id

                # loading the global model weights to client model
                client.set_model(global_weights)

                # training the client model for n' epochs
                client_model, train_loss, validation_loss = client.train(
                    self.loss_fn, self.local_rounds, round
                )

                local_models.append(client_model)

                if self.save_ckpt:
                    local_path = f"{self.checkpt_path}/global_{round+1}/clients/client_model_{client_id}.pth"
                    self.save_models(client_model, local_path)

                self.client_dict[client_id]["training_loss"].append(train_loss)
                self.client_dict[client_id]["validation_loss"].append(validation_loss)

            # update global model parameters here
            state_dicts = [model.state_dict() for model in local_models]
            for key in model_layers:
                global_model.track_layers[key].weight.data = torch.stack(
                    [item[str(key) + ".weight"] for item in state_dicts]
                ).mean(dim=0)
                global_model.track_layers[key].bias.data = torch.stack(
                    [item[str(key) + ".bias"] for item in state_dicts]
                ).mean(dim=0)
                # info here - https://discuss.pytorch.org/t/how-to-change-weights-and-bias-nn-module-layers/93065/2
            global_weights = global_model.state_dict()

            if self.save_ckpt:
                global_path = f"{self.checkpt_path}/global_{round+1}/global_model.pth"
                self.save_models(global_model, global_path)

        # self.writer.flush()
        # self.writer.close()

        return global_model

    def save_stats(self):
        for client_id, data in self.client_dict.items():
            file_path = f"training_stats/fedl/fl_stats_epoch{self.global_rounds}_{self.local_rounds}/client_{client_id}.csv"
            df = pd.DataFrame(data)
            if os.path.exists(file_path):
                df.to_csv(file_path, index=False)
            else:
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                df.to_csv(file_path, index=False)

    def save_models(self, model: torch.nn.Module, ckptpath: str) -> None:
        """
        Saving the training stats and the model.

        Parameters:
        ----------------
        model:
            Trained model.
        ckptpath: str;
            Path to save the model and the training stats. Default is None.

        Returns:
        ----------------
        None
        """

        if os.path.exists(ckptpath):
            torch.save(
                model.state_dict(),
                ckptpath,
            )
        else:
            os.makedirs(os.path.dirname(ckptpath), exist_ok=True)
            torch.save(
                model.state_dict(),
                ckptpath,
            )


if __name__ == "__main__":
    device = get_device()
    parser = argparse.ArgumentParser(description="Federated training parameters")
    parser.add_argument("--loss_function", type=str, default="L1Loss")
    parser.add_argument("--log_summary", action="store_true")
    parser.add_argument("--global_rounds", type=int, default=25)
    parser.add_argument("--local_rounds", type=int, default=10)
    parser.add_argument("--save_ckpt", action="store_true")
    args = parser.parse_args()

    features = 169

    # Hyper Parameters
    loss_fn = getattr(torch.nn, args.loss_function)()
    log_summary = args.log_summary
    global_rounds = args.global_rounds
    local_rounds = args.local_rounds
    epochs = global_rounds * local_rounds
    save_ckpt = args.save_ckpt

    checkpt_path = f"checkpt/fedl/selected_/epoch_{epochs}/{global_rounds}_rounds_{local_rounds}_epochs_per_round/"

    federation = Federation(
        checkpt_path,
        features,
        loss_fn,
        global_rounds,
        local_rounds,
        save_ckpt,
        log_summary,
    )

    # client_ids = [f"c{i}" for i in range(1, 25)]
    client_ids = [
        "c5",
        "c9",
        "c10",
        "c12",
        "c15",
        "c16",
        "c17",
        "c19",
        "c22",
        "c23",
        "c24",
    ]

    print("Federation with clients " + ", ".join(client_ids))

    start = time.time()
    federation.set_clients(client_ids=client_ids)
    model = ShallowNN(169)
    trained_model = federation.train(model)
    federation.save_stats()
    model_path = f"{checkpt_path}/global_model.pth"
    federation.save_models(trained_model.eval(), model_path)

    # federation.save_stats(trained_model, training_stats)

    print("Federation with clients " + ", ".join(client_ids))
    print(
        "Approximate time taken to train",
        str(round((time.time() - start) / 60, 2)) + " minutes",
    )
