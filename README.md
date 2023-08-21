# FedCE

This is the code repository that was developed for the FedCE - Federated Learning Contribution Evaluation.

## Run baseline training

Here all the training data from all the clients will be shuffled and trained the model, this is just for the demonstration, We are not going to use any insight from this for the hyperparameter tuning of the federated learning. Since all about federated learning is privacy-preserving.

```bash
python baseline.py
```

## Run isolated training

```bash
python isolated_learning.py
```

## Run Federated Learning

```bash
python federated_learning.py
```

## Loading the tensorboard

```bash
tensorboard --logdir='runs'
```