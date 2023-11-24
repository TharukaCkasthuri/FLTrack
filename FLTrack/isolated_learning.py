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

import os
import torch
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

from FLTrack.utils import Client
from FLTrack.utils import get_device
from models import ShallowNN

from torch.utils.tensorboard import SummaryWriter

device = get_device()

parser = argparse.ArgumentParser(description="Isolated client training parameters")
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--learning_rate", type=float, default=0.005)
parser.add_argument("--loss_function", type=str, default="L1Loss")
parser.add_argument("--log_summary", action="store_true")

args = parser.parse_args()

features = 197

# Hyper Parameters
loss_fn = getattr(torch.nn, args.loss_function)()
batch_size = args.batch_size
epochs = args.epochs
learning_rate = args.learning_rate
log_summary = args.log_summary

# Args
checkpt_path = f"checkpt/epoch_{epochs}/isolated/"

client_ids = [f"{i}_{j}" for i in range(4) for j in range(6)]

clients = [
    Client(
        id,
        torch.load(f"trainpt/{id}.pt"),
        torch.load(f"testpt/{id}.pt"),
        batch_size,
    )
    for id in client_ids
]

if log_summary:
    with SummaryWriter(comment=f"_iso_train_batch_{batch_size}") as writer:
        for client in clients:
            client_id = client.client_id
            client_model = ShallowNN(features)
            train_losses = []
            validation_losses = []

            optimizer = torch.optim.SGD(client_model.parameters(), lr=learning_rate)

            for epoch in tqdm(range(epochs)):
                client_model, train_loss = client.train(
                    client_model, loss_fn, optimizer, epoch
                )
                print("Train loss:", train_loss)
                validation_loss = client.eval(client_model, loss_fn)
                print("Validation loss:", validation_loss, "\n")
                train_losses.append(train_loss)
                validation_losses.append(validation_loss)
                writer.add_scalars(
                    str(client_id),
                    {"Training Loss": train_loss, "Validation Loss": validation_loss},
                    epoch,
                )

            model_path = f"{checkpt_path}batch{batch_size}_client_{client_id}.pth"

            saving_model = client_model.eval()

            if os.path.exists(model_path):
                torch.save(saving_model.state_dict(), model_path)
            else:
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                torch.save(saving_model.state_dict(), model_path)

                loss_df = pd.DataFrame(
                    np.column_stack([train_losses, validation_losses]),
                    columns=["iso_train", "iso_val"],
                )
                loss_df.to_csv(
                    f"losses/batch{batch_size}_client_{client_id}.csv",
                    index=False,
                )

        writer.flush()
        writer.close()

else:
    for client in clients:
        client_id = client.client_id
        client_model = ShallowNN(features)
        train_losses = []
        validation_losses = []

        optimizer = torch.optim.SGD(client_model.parameters(), lr=learning_rate)

        for epoch in tqdm(range(epochs)):
            client_model, train_loss = client.train(
                client_model, loss_fn, optimizer, epoch
            )
            print("Train loss:", train_loss)

        model_path = f"{checkpt_path}batch{batch_size}_client_{client_id}.pth"

        saving_model = client_model.eval()

        if os.path.exists(model_path):
            torch.save(saving_model.state_dict(), model_path)
        else:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(saving_model.state_dict(), model_path)
