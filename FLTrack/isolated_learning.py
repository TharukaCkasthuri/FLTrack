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

from clients import Client
from utils import get_device
from models import ShallowNN
from params import model_hparams

from torch.utils.tensorboard import SummaryWriter

device = get_device()

parser = argparse.ArgumentParser(description="Isolated client training parameters")
parser.add_argument("--loss_function", type=str, default="L1Loss")
parser.add_argument("--log_summary", action="store_true")


# best lr for 1 is 0.01, with 256 batch size , and weight_decay = 0.01
# best lr for 6 is 0.00001, with 256 batch size , and weight_decay = 0.00001
# best lr for 18,20 is 0.01 with 256 batch size weight_decay = 0.0001
# best lr for 3 is 0.01 with 256 batch size weight_decay = 0.0001
# best lr for 2,4,5,6,7,8,9,11,12,13,14,15,16,17,19, 21, 22,23,24 is 0.01 with 256 batch size weight_decay = 0.0001
# best lr for 10 is 0.001 with 256 batch size weight_decay = 0.01

# epoch 250

# best lr for 7,9,14,15,16,17, 22,23,24 is 0.01 with 256 batch size weight_decay = 0.0001
# best lr for 1 is 0.01, with 256 batch size , and weight_decay = 0.01
# best lr 2, 4, 5 for lr 0.01 weight_decay = 0.001
# best lr for 3 is 0.01 with 256 batch size weight_decay = 0.001
# best lr for 6,8,11,12,13,19,21 is 0.01 with 256 batch size weight_decay = 0.001
# best lr for 10 is 0.001 with 256 batch size weight_decay = 0.01
# best lr for 18,20 is 0.01 with 256 batch size weight_decay = 0.0001

args = parser.parse_args()

features = 169

# Hyper Parameters
loss_fn = getattr(torch.nn, args.loss_function)()
log_summary = args.log_summary

client_ids = [f"c{i}" for i in range(1, 25)]

#client_ids = ["c18"]

clients = [
    Client(
        id,
        torch.load(f"../trainpt/{id}.pt"),
        torch.load(f"../testpt/{id}.pt"),
        model_hparams[f"{id}"]["batch_size"],
        model_hparams[f"{id}"]["learning_rate"],
        model_hparams[f"{id}"]["weight_decay"],
        local_model=ShallowNN(features),
    )
    for id in client_ids
]


for client in clients:
    client_id = client.client_id
    model = ShallowNN(features)
    train_losses = []
    validation_losses = []

    client_model, train_loss, validation_loss = client.train(
        loss_fn, model_hparams[f"{client_id}"]["num_epochs"]
    )

    print("\n")
    train_losses.append(train_loss)
    validation_losses.append(validation_loss)

    print(f"Client {client_id} has trained successfully", validation_loss)

    model_path = f"checkpt/isolated/epoch_{model_hparams[f"{client_id}"]["num_epochs"]}/batch{client.batch_size}_client_{client_id}.pth"
    print(model_path)

    saving_model = client_model  # .eval()

    if os.path.exists(model_path):
        torch.save(saving_model.state_dict(), model_path)
    else:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(saving_model.state_dict(), model_path)
