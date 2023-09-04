import time
import torch
import argparse

import pandas as pd

from utils import get_device

from models import ShallowNN
from federated_learning import Federation

if __name__ == "__main__":
    device = get_device()
    parser = argparse.ArgumentParser(description="Federated training parameters")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--learning_rate", type=float, default=0.005)
    args = parser.parse_args()

    checkpt_path = "checkpt/"
    features = 197

    # Hyper Parameters
    loss_fn = torch.nn.L1Loss()
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.learning_rate

    fed = Federation(checkpt_path, features, loss_fn, batch_size, epochs, learning_rate)

    skips_list = [
        "0_0",
        "0_1",
        "0_2",
        "0_3",
        "0_4",
        "0_5",
        "1_0",
        "1_1",
        "1_2",
        "1_3",
        "1_4",
        "1_5",
        "2_0",
        "2_1",
        "2_2",
        "2_3",
        "2_4",
        "2_5",
        "3_0",
        "3_1",
        "3_2",
        "3_3",
        "3_4",
        "3_5",
    ]

    times = []
    for item in skips_list:
        client_ids = [
            "0_0",
            "0_1",
            "0_2",
            "0_3",
            "0_4",
            "0_5",
            "1_0",
            "1_1",
            "1_2",
            "1_3",
            "1_4",
            "1_5",
            "2_0",
            "2_1",
            "2_2",
            "2_3",
            "2_4",
            "2_5",
            "3_0",
            "3_1",
            "3_2",
            "3_3",
            "3_4",
            "3_5",
        ]
        start = time.time()
        try:
            client_ids.remove(item)
        except:
            raise ValueError
        print("Federation with clients " + ", ".join(client_ids))
        fed.set_clients(client_ids=client_ids)
        trained_model, training_stats = fed.train(ShallowNN)
        fed.save_stats(trained_model, training_stats, path_det="influence/" + str(item))
        times.append((time.time() - start))

print(
    "Approximate time taken to train",
    str(round(sum(times) / len(times), 2)) + " minutes",
)
