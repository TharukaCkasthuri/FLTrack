import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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
    def __init__(self, client_id, dataset):
        self.client_id = client_id
        self.dataset = dataset

        X = dataset.drop(columns=["label"])
        y = dataset["label"].values

        scaler_x = StandardScaler()
        scaler_y = StandardScaler()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.33, random_state=42)

        self.X_train = pd.DataFrame(scaler_x.fit_transform(
            X_train.values), index=X_train.index, columns=X_train.columns)
        
        self.X_test = pd.DataFrame(scaler_x.transform(
            X_test.values), index=X_test.index, columns=X_test.columns)
        
        self.y_train = scaler_y.fit_transform(y_train.reshape(-1, 1))
        self.y_test = scaler_y.transform(y_test.reshape(-1, 1))
        

    def get_x_train(self):
        return self.X_train

    def get_y_train(self):
        return self.y_train

    def get_x_test(self):
        return self.X_test

    def get_y_test(self):
        return self.y_test

    def get_dataset_size(self):
        return len(self.dataset)

    def get_client_id(self):
        return self.client_id

    def set_model(self, model):
        self.model = model

    def get_model(self):
        return self.model

    def train(self, model, loss_fn, optimizer, batch_size, epoch=0):

        #take this out of the function
        dataset = CustomDataSet(self.X_train, self.y_train)
        client_dl = DataLoader(dataset, batch_size, shuffle=True)

        batch_loss = []

        for batch_idx, (x, y) in enumerate(client_dl):
            outputs = model(x)
            loss = loss_fn(outputs, y)
            model.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 50 == 0:
                print('Epoch: {} \tClient ID:{} \t[{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch+1,str(self.client_id) ,batch_idx * len(x), len(client_dl.dataset),
                    100. * batch_idx / len(client_dl), loss.item()))

            batch_loss.append(loss.item())

        loss_avg = sum(batch_loss)/len(batch_loss)
        print('\nTrain loss:', loss_avg)

        return model, loss_avg
    
    def save_model(self, path):
        saving_model = self.model.eval()
        torch.save(saving_model.state_dict(), path)

    def eval(self, model):
        pass
