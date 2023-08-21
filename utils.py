import torch
import time

import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from torch.autograd import grad

from sklearn.metrics import mean_squared_error, mean_absolute_error

def load_file(file_path)->pd.DataFrame:
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
    df = pd.merge(left=tmp["dataset"]["X"], right=tmp["dataset"]
                  ["Y"], how="left", left_index=True, right_index=True)
    df.drop(columns=["TimeStamp", "WritesAvg"], inplace=True)
    df.rename(columns={"ReadsAvg": "label"}, inplace=True)
    df.reset_index(inplace=True, drop=True)

    return df


def get_device()->torch.device:
    """
    Returns the device to be used for training.

    Parameters:
    --------
    None

    Returns:
    --------
    device: torch.device object
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def evaluate(model, dataloader, loss_fn)->tuple:
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
        batch_loss = loss_fn(predicts,y)
        
        loss.append(batch_loss.item())
        batch_mse = mean_squared_error(y, np.squeeze(predicts.detach().numpy()))
        mse.append(batch_mse)
        batch_mae = mean_absolute_error(y, np.squeeze(predicts.detach().numpy()))
        mae.append(batch_mae)
    
    return sum(loss)/len(loss), sum(mse)/len(mse), sum(mae)/len(mae)


def calculate_hessian(model, loss_fn, data_loader)->tuple:
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
        batch_loss = loss_fn(predicts,y)
        total_loss += batch_loss
    avg_loss = total_loss / len(data_loader)
    print(avg_loss)
    
    # Allocate Hessian size
    num_param = sum(p.numel() for p in model.parameters())
    hessian_matrix = torch.zeros((num_param, num_param))
    
    start = time.time()

    # Calculate Jacobian w.r.t. model parameters
    J = grad(avg_loss, list(model.parameters()), create_graph=True)
    J = torch.cat([e.flatten() for e in J]) # flatten
    
    for i in range(num_param):
        result = torch.autograd.grad(J[i], list(model.parameters()), retain_graph=True)
        hessian_matrix[i] = torch.cat([r.flatten() for r in result]) # flatten
        
    return hessian_matrix, time.time() - start


class CustomDataSet(Dataset):

    """
    Custom dataset class for the training and validation dataset.
    """

    def __init__(self, x, y)->None:

        self.x_train = torch.tensor(x.values, dtype=torch.float32)
        self.y_train = torch.tensor(y, dtype=torch.float32)

    def __len__(self)->int:
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

    def __getitem__(self, idx)->tuple:
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

    def __init__(self, client_id:str, train_dataset:object, test_dataset:object, batch_size:int)->None:
        self.client_id = client_id
        self.train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
        self.test_dataloader = DataLoader(test_dataset, batch_size, shuffle=True)

    def train(self, model, loss_fn, optimizer, epoch=0)->tuple:
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

            if batch_idx % 50 == 0:
                print('Epoch: {} \tClient ID:{} \t[{}/{} ({:.0f}%)] \tLoss: {:.6f}'.format(
                    epoch+1,str(self.client_id) ,batch_idx * len(x), len(self.train_dataloader.dataset),
                    100. * batch_idx / len(self.train_dataloader), loss.item()))

            batch_loss.append(loss.item())

        loss_avg = sum(batch_loss)/len(batch_loss)

        return model, loss_avg
    

    def eval(self, model, loss_fn)->float:
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
        for _, (x,y) in enumerate(self.test_dataloader):
            outputs = model(x)
            loss = loss_fn(outputs,y)
            batch_loss.append(loss.item())
        loss_avg =  sum(batch_loss)/len(batch_loss)

        return loss_avg
