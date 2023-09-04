import torch
import time

import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from torch.autograd import grad

from sklearn.metrics import mean_squared_error, mean_absolute_error


def load_file(file_path) -> pd.DataFrame:
    """
    Loads the pickle file into a dataframe.

    Parameters:
    --------
    file_path:str

    Returns:
    --------
    df: Pandas Dataframe object
    """
    tmp = pd.read_pickle(file_path)
    df = pd.merge(
        left=tmp["dataset"]["X"],
        right=tmp["dataset"]["Y"],
        how="left",
        left_index=True,
        right_index=True,
    )
    df.drop(columns=["TimeStamp", "WritesAvg"], inplace=True)
    df.rename(columns={"ReadsAvg": "label"}, inplace=True)
    df.reset_index(inplace=True, drop=True)

    return df


def get_device() -> torch.device:
    """
    Returns the device to be used for training.

    Parameters:
    --------
    None

    Returns:
    --------
    device: torch.device object
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model, dataloader, loss_fn) -> tuple:
    """
    Evaluate the model with validation dataset.

    Parameters:
    -------------
    model: torch.nn.Module object; model to be evaluated
    dataloader: torch.utils.data.DataLoader object; validation dataset
    loss_fn: torch.nn.Module object; loss function

    Returns:
    -------------
    loss: float; average loss
    mse: float; average mean squared error
    mae: float; average mean absolute error

    """
    model.eval()
    loss, mse, mae = [], [], []

    for _, (x, y) in enumerate(dataloader):
        predicts = model(x)
        batch_loss = loss_fn(predicts, y)

        loss.append(batch_loss.item())
        batch_mse = mean_squared_error(y, np.squeeze(predicts.detach().numpy()))
        mse.append(batch_mse)
        batch_mae = mean_absolute_error(y, np.squeeze(predicts.detach().numpy()))
        mae.append(batch_mae)

    return sum(loss) / len(loss), sum(mse) / len(mse), sum(mae) / len(mae)


def evaluate_mae_with_confidence(
    model, dataloader, num_bootstrap_samples=250, confidence_level=0.95
):
    """
    Evaluate the model with validation dataset and calculate confidence intervals.

    Parameters:
    -------------
    model: torch.nn.Module object; model to be evaluated
    dataloader: torch.utils.data.DataLoader object; validation dataset
    num_bootstrap_samples: int; number of bootstrap samples
    confidence_level: float; confidence level

    Returns:
    -------------
    avg_mae: float; average mean absolute error
    (lower_mae, upper_mae): tuple; lower and upper bounds of the confidence interval for mean absolute error
    """
    model.eval()
    mae_values = []

    for _, (x, y) in enumerate(dataloader):
        predicts = model(x)
        batch_mae = mean_absolute_error(y, predicts.detach().cpu().numpy())
        mae_values.append(batch_mae)

    avg_mae = np.mean(mae_values)

    bootstrap_mae = []

    for _ in range(num_bootstrap_samples):
        bootstrap_sample_indices = np.random.choice(
            len(dataloader.dataset), size=len(dataloader.dataset), replace=True
        )
        bootstrap_dataloader = torch.utils.data.DataLoader(
            dataloader.dataset,
            batch_size=dataloader.batch_size,
            sampler=torch.utils.data.SubsetRandomSampler(bootstrap_sample_indices),
        )

        mae_values = []

        for _, (x, y) in enumerate(bootstrap_dataloader):
            predicts = model(x)
            batch_mae = mean_absolute_error(y, predicts.detach().cpu().numpy())
            mae_values.append(batch_mae)

        bootstrap_mae.append(np.mean(mae_values))

    # Calculate confidence intervals
    confidence_interval = (1 - confidence_level) / 2
    sorted_mae = np.sort(bootstrap_mae)

    lower_mae = sorted_mae[int(confidence_interval * num_bootstrap_samples)]
    upper_mae = sorted_mae[int((1 - confidence_interval) * num_bootstrap_samples)]

    bootstrap_mae_std = np.std(bootstrap_mae)

    return avg_mae, (lower_mae, upper_mae), bootstrap_mae_std


def influence(model, influenced_model, data_loader) -> tuple:
    model.eval()
    influenced_model.eval()
    model_pred = []
    influenced_model_pred = []

    for _, (x, y) in enumerate(data_loader):
        model_pred.append(model(x))
        influenced_model_pred.append(influenced_model(x))

    return model_pred, influenced_model_pred


def calculate_hessian(model, loss_fn, data_loader) -> tuple:
    """
    Calculate the Hessian matrix of the model for the given validation set.

    Parameters:
    -------------
    model: torch.nn.Module object; model to be evaluated
    loss_fn: torch.nn.Module object; loss function
    data_loader: torch.utils.data.DataLoader object; validation dataset

    Returns:
    -------------
    hessian_matrix: torch.tensor object; Hessian matrix
    time: float; time taken to calculate the Hessian matrix

    """
    model.eval()

    total_loss = 0
    for _, (x, y) in enumerate(data_loader):
        predicts = model(x)
        batch_loss = loss_fn(predicts, y)
        total_loss += batch_loss
    avg_loss = total_loss / len(data_loader)

    start = time.time()

    # Calculate Jacobian w.r.t. model parameters
    grads = torch.autograd.grad(avg_loss, list(model.parameters()), create_graph=True)

    hessian_matrix = []

    for grad in grads:
        hessian_row = []
        for grad_element in grad.view(-1):
            # Compute second-order gradient
            hessian_element = torch.autograd.grad(
                grad_element, model.parameters(), retain_graph=True
            )
            hessian_row.append(hessian_element)
        hessian_matrix.append(hessian_row)

    hessian_matrix = torch.stack(hessian_matrix).squeeze()

    end = time.time()
    print("Calculation time of hessian matrix", (end - start) / 60)

    return grads, hessian_matrix, end - start


def calculate_hessian_flattened(model, loss_fn, data_loader) -> tuple:
    """
    Calculate the Hessian matrix of the model for the given validation set. And flatten the Jacobian matrix.

    Parameters:
    -------------
    model: torch.nn.Module object; model to be evaluated
    loss_fn: torch.nn.Module object; loss function
    data_loader: torch.utils.data.DataLoader object; validation dataset

    Returns:
    -------------
    hessian_matrix: torch.tensor object; Hessian matrix
    time: float; time taken to calculate the Hessian matrix
    """
    model.eval()

    total_loss = 0
    for _, (x, y) in enumerate(data_loader):
        predicts = model(x)
        batch_loss = loss_fn(predicts, y)
        total_loss += batch_loss
    avg_loss = total_loss / len(data_loader)

    # Allocate Hessian size
    num_param = sum(p.numel() for p in model.parameters())

    hessian_matrix = torch.zeros((num_param, num_param))

    start = time.time()
    # Calculate Jacobian w.r.t. model parameters
    J = grad(avg_loss, list(model.parameters()), create_graph=True)
    J = torch.cat([e.flatten() for e in J])  # flatten

    print("Calculation time of Gradients", time.time() - start)

    restart = time.time()
    for i in range(num_param):
        result = torch.autograd.grad(J[i], list(model.parameters()), retain_graph=True)
        hessian_matrix[i] = torch.cat([r.flatten() for r in result])  # flatten

    print("Calculation time of Hessian", (time.time() - start) / 60)

    return hessian_matrix, (time.time() - start) / 60


class CustomDataSet(Dataset):

    """
    Custom dataset class for the training and validation dataset.
    """

    def __init__(self, x, y) -> None:
        self.x_train = torch.tensor(x.values, dtype=torch.float32)
        self.y_train = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Parameters:
        --------
        None

        Returns:
        --------
        length: int; length of the dataset
        """
        return len(self.y_train)

    def __getitem__(self, idx) -> tuple:
        """
        Returns the item at the given index.

        Parameters:
        ------------
        idx: int; index of the item

        Returns:
        ------------
        x_train: torch.tensor object; input data
        y_train: torch.tensor object; label
        """
        return self.x_train[idx], self.y_train[idx]


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
    ) -> None:
        self.client_id = client_id
        self.train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
        self.test_dataloader = DataLoader(test_dataset, batch_size, shuffle=True)

    def train(self, model, loss_fn, optimizer, epoch=0) -> tuple:
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
        batch_loss = []

        for batch_idx, (x, y) in enumerate(self.train_dataloader):
            outputs = model(x)
            loss = loss_fn(outputs, y)
            model.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 200 == 0:
                print(
                    "Epoch: {} \tClient ID:{} \t[{}/{} ({:.0f}%)] \tLoss: {:.6f}".format(
                        epoch + 1,
                        str(self.client_id),
                        batch_idx * len(x),
                        len(self.train_dataloader.dataset),
                        100.0 * batch_idx / len(self.train_dataloader),
                        loss.item(),
                    )
                )

            batch_loss.append(loss.item())

        loss_avg = sum(batch_loss) / len(batch_loss)

        return model, loss_avg

    def eval(self, model, loss_fn) -> float:
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
        for _, (x, y) in enumerate(self.test_dataloader):
            outputs = model(x)
            loss = loss_fn(outputs, y)
            batch_loss.append(loss.item())
        loss_avg = sum(batch_loss) / len(batch_loss)

        return loss_avg
