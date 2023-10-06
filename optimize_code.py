import time
import torch
import argparse
import os

import pandas as pd
from tqdm import tqdm

from utils import Client
from utils import get_device
from evals import evaluate
from models import ShallowNN

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader


class Federation:
    """
    Class for federated learning.

    Parameters:
    ------------
    checkpt_path: str; path to save the model
    features: int; number of features
    loss_fn: torch.nn.Module object; loss function
    batch_size: int; batch size
    learning_rate: float; learning rate
    """

    def __init__(
        self,
        checkpt_path: str,
        features: int,
        loss_fn,
        batch_size,
        learning_rate,
        rounds,
        epochs_per_round,
    ):
        self.checkpt_path = checkpt_path
        self.features = features
        self.loss_fn = loss_fn
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.training_rounds = rounds
        self.epochs_per_round = epochs_per_round

        self.writer = SummaryWriter(comment="_fed_train_batch" + str(batch_size))

    def set_clients(self, client_ids) -> None:
        """
        Setting up the clients for federated learning.

        Parameters:
        ----------------
        client_ids: list; list of client ids

        Returns:
        ----------------
        None
        """
        self.client_ids = client_ids
        self.clients = [
            Client(
                id,
                torch.load("trainpt/" + id + ".pt"),
                torch.load("testpt/" + id + ".pt"),
                self.batch_size,
            )
            for id in self.client_ids
        ]

        self.client_model_dict = {}
        for i in self.client_ids:
            self.client_model_dict[i] = ShallowNN(self.features)

        test_dataset = torch.utils.data.ConcatDataset(
            [torch.load("testpt/" + id + ".pt") for id in self.client_ids]
        )
        self.test_dataloader = DataLoader(test_dataset, self.batch_size, shuffle=True)

    def train(
        self,
        model,
        summery=False,
    ) -> tuple:
        """
        Training the model.

        Parameters:
        ----------------
        model: model to be trained
        summery: bool; whether to save the training stats and the model

        Returns:
        ----------------
        global_model: trained model
        training_stats: list; training stats
        """

        # initiate global model
        global_model = model(self.features)
        # global_model.to(device)
        global_model.train()

        global_weights = global_model.state_dict()
        model_layers = global_model.track_layers.keys()

        training_stats = []
        for round in tqdm(range(self.training_rounds)):
            local_models, local_loss = [], []

            print(f"\n | Global Training Round : {round+1} |\n")

            for client in self.clients:
                client_id = client.client_id
                training_stat_dict = {"client_id": client_id, "training_round": round}

                # loading the global model weights to client model
                self.client_model_dict[client_id].load_state_dict(global_weights)

                # setting up the optimizer
                optimizer = torch.optim.SGD(
                    self.client_model_dict[client_id].parameters(),
                    lr=self.learning_rate,
                )

                # training the client model
                this_client_state_dict, training_loss = client.train(
                    self.client_model_dict[client_id],
                    self.loss_fn,
                    optimizer,
                    self.epochs_per_round,
                )

                local_loss.append(training_loss)
                local_models.append(this_client_state_dict)

                training_stat_dict["fed_train"] = (
                    sum(local_loss[-self.epochs_per_round :]) / self.epochs_per_round
                )
                validation_loss = client.eval(
                    self.client_model_dict[client_id], self.loss_fn
                )
                training_stat_dict["fed_val"] = validation_loss

                training_stats.append(training_stat_dict)

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

            if summery:
                global_training_loss = sum(local_loss) / len(local_loss)
                global_validation_loss, _, _ = evaluate(
                    global_model, self.test_dataloader, loss_fn
                )

                self.writer.add_scalars(
                    "Global Model - Federated Learning",
                    {
                        "Training Loss": global_training_loss,
                        "Validation Loss": global_validation_loss,
                    },
                    round,
                )

        self.writer.flush()
        self.writer.close()

        return global_model, training_stats

    def save_stats(self, model, training_stat) -> None:
        """
        Saving the training stats and the model.

        Parameters:
        ----------------
        model: trained model
        training_stat: list; training stats
        path_det: str; path to save the model and the training stats

        Returns:
        ----------------
        None
        """

        path = f"losses/_federated_stats_epoch{self.training_rounds}_{self.epochs_per_round}.csv"

        if os.path.exists(path):
            pd.DataFrame.from_dict(training_stat).to_csv(
                path,
                index=False,
            )
        else:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            pd.DataFrame.from_dict(training_stat).to_csv(
                path,
                index=False,
            )

        if os.path.exists(self.checkpt_path):
            torch.save(
                model.state_dict(),
                f"{self.checkpt_path}_fedl_global_{self.training_rounds}_{self.epochs_per_round}.pth",
            )
        else:
            os.makedirs(os.path.dirname(self.checkpt_path), exist_ok=True)
            torch.save(
                model.state_dict(),
                f"{self.checkpt_path}_fedl_global_{self.training_rounds}_{self.epochs_per_round}.pth",
            )


if __name__ == "__main__":
    device = get_device()
    parser = argparse.ArgumentParser(description="Federated training parameters")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.005)
    parser.add_argument("--loss_function", type=str, default="L1Loss")
    parser.add_argument("--log_summary", action="store_true")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--epochs_per_round", type=int, default=3)

    args = parser.parse_args()

    checkpt_path = (
        f"checkpt/{args.rounds}_rounds_{args.epochs_per_round}_epochs_per_round/"
    )

    features = 197

    # Hyper Parameters
    loss_fn = getattr(torch.nn, args.loss_function)()
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    log_summary = args.log_summary
    rounds = args.rounds
    epochs_per_round = args.epochs_per_round

    federation = Federation(
        checkpt_path,
        features,
        loss_fn,
        batch_size,
        learning_rate,
        rounds,
        epochs_per_round,
    )

    client_ids = [f"{i}_{j}" for i in range(4) for j in range(6)]

    print("Federation with clients " + ", ".join(client_ids))

    start = time.time()
    federation.set_clients(client_ids=client_ids)
    trained_model, training_stats = federation.train(ShallowNN)
    federation.save_stats(trained_model, training_stats)
    print("Federation with clients " + ", ".join(client_ids))
    print(
        "Approximate time taken to train",
        str(round((time.time() - start) / 60, 2)) + " minutes",
    )
