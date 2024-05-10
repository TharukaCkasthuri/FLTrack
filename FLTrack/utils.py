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
import numpy as np

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


def remove_outliers(dataframe: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Removes the outliers from the given column.

    Parameters:
    --------
    dataframe: pd.DataFrame object
    column: str; column name

    Returns:
    --------
    new_df: pd.DataFrame object
    """

    percentile25 = dataframe[column].quantile(0.25)
    percentile75 = dataframe[column].quantile(0.75)
    iqr = percentile75 - percentile25

    upper_limit = percentile75 + 1.5 * iqr
    lower_limit = percentile25 - 1.5 * iqr

    new_df = dataframe[dataframe["label"] < upper_limit]
    new_df = new_df[new_df["label"] > lower_limit]

    return new_df


def drop_columns(df: pd.DataFrame) -> None:
    """
    Drops the columns with the same values.

    Parameters:
    --------
    df: pd.DataFrame object

    Returns:
    --------
    None
    """

    for i in df.columns:
        if df[i].max() == df[i].min():
            df.drop(i, axis=1, inplace=True)


def update_distribution(
    original_df: pd.DataFrame, adj_mean: float, adj_std: float, samples: int
) -> pd.DataFrame:
    """
    Updates the distribution of the given dataframe. With weighted sampling based on the z-scores.

    Parameters:
    --------
    original_df: pd.DataFrame object
    adj_mean: float; adjusted mean
    adj_std: float; adjusted standard deviation

    Returns:
    --------
    updated_df: pd.DataFrame object
    """
    original_mean = original_df["label"].mean()
    original_std = original_df["label"].std()
    desired_mean = original_mean + adj_mean
    desired_std = original_std * adj_std
    z_scores = (original_df["label"] - original_mean) / original_std
    weights = np.exp(
        -0.5 * ((z_scores - (desired_mean - original_mean) / desired_std) ** 2)
    )
    updated_df = original_df.sample(n=samples, replace=False, weights=weights)

    return updated_df


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
