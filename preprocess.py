import os
import torch

import pandas as pd
import numpy as np

from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from utils import CustomDataSet

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

def build_dataset(dataframe, client_id):
    """
    Split the dataframe into train and test, saving as a pytorch dataset.

    Parameters:
    --------
    dataframe: Pandas dataframe object
    client_id: String, Unique id

    Returns:
    --------
    None
    """

    for c in dataframe.columns:
        dataframe[c] = dataframe[c].apply(lambda a: np.ma.log(a))
        print(str(c) + "done")

    X = dataframe.drop(columns=["label"])
    y = dataframe["label"].values

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)

    X_train = pd.DataFrame(scaler_x.fit_transform(
        X_train.values), index=X_train.index, columns=X_train.columns)
    
    X_test = pd.DataFrame(scaler_x.transform(
        X_test.values), index=X_test.index, columns=X_test.columns)
    
    y_train = scaler_y.fit_transform(y_train.reshape(-1, 1))
    y_test = scaler_y.transform(y_test.reshape(-1, 1))

    train_dataset = CustomDataSet(X_train, y_train)
    test_dataset = CustomDataSet(X_test, y_test)

    torch.save(train_dataset, "./trainpt/" +str(client_id) +".pt")
    torch.save(test_dataset, "./testpt/" +str(client_id) +".pt")


def main():
    data_path = "../kv_data/kv/"
    files = os.listdir(data_path)
    ids = [file.split(".")[0] for file in files]
    files_path = [os.path.join(data_path, file) for file in files]

    for id, pickle_file in zip(ids,files_path):
        build_dataset(load_file(pickle_file), id)

if __name__ == "__main__":
    main()