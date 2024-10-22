# FLTrack

This is the code repository that was developed for the paper "Clients Behavior Monitoring in Federated Learning via Eccentricity Analysis".

## Run baseline training

Here all the training data from all the clients will be shuffled and trained the model, this is just for the demonstration, We are not going to use any insight from this for the hyperparameter tuning of the federated learning. Since all about federated learning is privacy-preserving.

```bash
python baseline.py
```

## Run isolated training

This script is to train the models for each client with their locally available data without federation.

**Params:**
- `batch_size` (int): Batch size used during training.
- `epochs` (int): Number of training epochs.
- `learning_rate` (float): Learning rate for the optimization algorithm.
- `loss_function` (str): The loss function used to measure the model's performance.
- `log_summary` (bool): If set, the script will generate a summary log. 
 
**Example**
```bash
python isolated_learning.py --batch_size 64 --epochs 50 --learning_rate 0.005 --loss_function 'L1Loss' --log_summary
```
## Run Federated Learning

The script is for federated learning to train machine learning models across multiple clients. It includes parameters for configuring the federated training process.

```bash
python FLTrack/federated_learning.py --save_ckpt
```

**Params:**


- `loss_function` (string): The loss function used to measure the model's performance. (Default: "L1Loss")
- `log_summary` (bool): If set, the script will generate a summary log.
- `global_rounds` (int): Number of federated training rounds. (Default: 25)
- `local_rounds` (int): Number of training epochs per federated round. (Default: 10)
- `save_ckpt` (bool): If set, the script will save checkpoints during training.


If you are using this code please cite this paper;

```
@inproceedings{Kasthuriarachchige2024,
  title={Clients Behavior Monitoring in Federated Learning via Eccentricity Analysis},
  author={Tharuka Kasthuriarachchige and Selim Ickin and Shahrooz Abghari and Veselka Boeva},
  booktitle={IEEE International Conference on Evolving and Adaptive Intelligent Systems, IEEE EAIS 2024 (23–24 May 2024, Madrid, Spain)},
  pages={},
  year={2024}
}
```