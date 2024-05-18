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

import pandas as pd

from utils import get_device

from models import ShallowNN
from federated_learning import Federation

if __name__ == "__main__":
    device = get_device()
    parser = argparse.ArgumentParser(
        description="Federated training parameters for Influence calculation"
    )
    parser.add_argument("--loss_function", type=str, default="L1Loss", help="Loss function")
    parser.add_argument("--log_summary", action="store_true", help="Log summary, pass to log")
    parser.add_argument("--global_rounds", type=int, default=25, help="Number of global rounds")
    parser.add_argument("--local_rounds", type=int, default=10)
    parser.add_argument("--save_ckpt", action="store_true", help="Save checkpoint, pass to save checkpoint")
    args = parser.parse_args()

    features = 169

    # Parameters
    loss_fn = getattr(torch.nn, args.loss_function)()
    log_summary = args.log_summary
    global_rounds = args.global_rounds
    local_rounds = args.local_rounds
    epochs = global_rounds * local_rounds
    save_ckpt = args.save_ckpt

    skips_list = [f"c{i}" for i in range(1, 25)]

    times = []
    for item in skips_list:
        client_ids = [f"c{i}" for i in range(1, 25)]

        checkpt_path = f"checkpt/influence/{item}"

        federation = Federation(
            checkpt_path,
            features,
            loss_fn,
            global_rounds,
            local_rounds,
            save_ckpt,
            log_summary,
        )

        # Start training
        start = time.time()

        try:
            client_ids.remove(item)
        except:
            raise ValueError
        
        model = ShallowNN(169)
        print(f"Federation with clients {', '.join(client_ids)}")
        federation.set_clients(client_ids=client_ids)
        trained_model = federation.train(model)
        model_path = f"{checkpt_path}/global_model.pth"
        federation.save_models(trained_model.eval(), model_path)
        times.append((time.time() - start))

average_time = sum(times) / len(times)
print(f"Approximate time taken to train all influenced models: {average_time:.2f} minutes")
