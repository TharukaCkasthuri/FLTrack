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

Paper: [Title of Your Paper]
Published in: [Journal/Conference Name]
"""

import torch
import itertools

import pandas as pd

from torch.utils.data import Dataset, DataLoader


def load_file(file_path: str) -> pd.DataFrame:
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


def get_all_possible_pairs(client_ids: list) -> list:
    """
    Returns all possible pairs of client ids.

    Parameters:
    --------
    client_ids: list; list of client ids

    Returns:
    --------
    pairs: list; list of all possible pairs of client ids
    """

    pairs = list(itertools.combinations(client_ids, 2))

    return pairs


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

    def get_subset(self, start_index, end_index):
        """
        Returns a subset of the dataset based on the specified range.

        Parameters:
        ------------
        start_index: int; starting index of the subset
        end_index: int; ending index of the subset

        Returns:
        ------------
        subset_x: torch.tensor object; subset of input data
        subset_y: torch.tensor object; subset of labels
        """
        subset_x = self.x_train[start_index:end_index]
        subset_y = self.y_train[start_index:end_index]
        return subset_x, subset_y
