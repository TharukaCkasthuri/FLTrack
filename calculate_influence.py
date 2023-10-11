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
    parser.add_argument("--learning_rate", type=float, default=0.005)
    parser.add_argument("--loss_function", type=str, default="L1Loss")
    parser.add_argument("--log_summary", action="store_true")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--epochs_per_round", type=int, default=25)
    args = parser.parse_args()

    features = 197

    # Hyper Parameters
    loss_fn = getattr(torch.nn, args.loss_function)()
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    log_summary = args.log_summary
    rounds = args.rounds
    epochs_per_round = args.epochs_per_round
    epochs = rounds * epochs_per_round

    skips_list = [f"{i}_{j}" for i in range(4) for j in range(6)]

    times = []
    for item in skips_list:
        client_ids = [f"{i}_{j}" for i in range(4) for j in range(6)]

        checkpt_path = f"checkpt/epoch_{epochs}/influence/{rounds}_rounds_{epochs_per_round}_epochs_per_round/{item}"

        fed = Federation(
            checkpt_path,
            features,
            loss_fn,
            batch_size,
            learning_rate,
            rounds,
            epochs_per_round,
        )

        start = time.time()

        try:
            client_ids.remove(item)
        except:
            raise ValueError
        print(f"Federation with clients {', '.join(client_ids)}")
        fed.set_clients(client_ids=client_ids)
        trained_model, training_stats = fed.train(ShallowNN)
        fed.save_stats(trained_model, training_stats)
        times.append((time.time() - start))

average_time = sum(times) / len(times)
print(f"Approximate time taken to train: {average_time:.2f} minutes")
