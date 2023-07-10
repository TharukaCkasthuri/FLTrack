import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

def load_file(file_path):
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


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CustomDataSet(Dataset):

    def __init__(self, x, y):

        self.x_train = torch.tensor(x.values, dtype=torch.float32)
        self.y_train = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx]


class Client:
    def __init__(self, client_id:str, train_dataset:object, test_dataset:object, batch_size:int):
        self.client_id = client_id
        self.train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
        self.test_dataloader = DataLoader(test_dataset, batch_size, shuffle=True)

    def set_model(self, model):
        self.model = model

    def get_model(self):
        return self.model

    def train(self, model, loss_fn, optimizer, epoch=0):

        batch_loss = []

        for batch_idx, (x, y) in enumerate(self.train_dataloader):
            outputs = model(x)
            loss = loss_fn(outputs, y)
            model.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 50 == 0:
                print('Epoch: {} \tClient ID:{} \t[{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch+1,str(self.client_id) ,batch_idx * len(x), len(self.train_dataloader.dataset),
                    100. * batch_idx / len(self.train_dataloader), loss.item()))

            batch_loss.append(loss.item())

        loss_avg = sum(batch_loss)/len(batch_loss)
        print('\nTrain loss:', loss_avg)

        return model, loss_avg
    
    def save_model(self, path):
        saving_model = self.model.eval()
        torch.save(saving_model.state_dict(), path)

    def eval(self, model):
        pass
