"""
utils.py is a module for helper functions in the oma_tracking package.
-------------------------------------------------------------------------
Author: Maximillian Weil
"""
import pandas as pd
from typing import Union


def check_columns(cols: list, data: pd.DataFrame) -> bool:
    """
    Check if all elements of a list are in the columns of a dataframe.

    Args:
        columns (list): The list of columns to be checked
        data (pd.DataFrame): The dataframe to be checked against

    Returns:
        bool: True if all elements of the list are in the columns of the dataframe, False otherwise
    """
    return all(col in data.columns for col in cols)


def validate_data(cols: list, modal_data: pd.DataFrame) -> None:
    """Validate the modal data to contain all the required columns.

    Args:
        modal_data (pd.DataFrame): The modal data to be validated.

    Raises:
         ValueError: If the modal data does not contain all the required columns.
    """
    if not check_columns(cols, modal_data):
        raise ValueError("The modal data does not contain all the required columns.")
        

def data_selection(
    modal_data: pd.DataFrame,
    cols: list,
    min_size: float,
    max_damping: float,
    frequency_range: Union[tuple[float, float], None] = None
    ) -> pd.DataFrame:
    """Select the data to be used for clustering.

    Args:
        modal_data (pd.DataFrame): The modal data to be used for clustering.
        cols (list): The columns of the dataframe that should be used for clustering.
        min_size (float): The minimum size of a mode to be considered for clustering.
        max_damping (float): The maximum damping of a mode to be considered for clustering.
        frequency_range (Union[tuple[float, float], None], optional): The frequency range to be used for clustering.
            Defaults to None.

    Raises:
        ValueError: _description_
        KeyError: _description_

    Returns:
        pd.DataFrame: _description_
    """    
    # feature selection
    selected_data = modal_data.copy()
    # Data selection based on frequency_range
    frequency_col = 'mean_frequency'
    if frequency_col not in modal_data.columns:
        frequency_col = 'frequency'
        if frequency_col not in modal_data.columns:
            raise ValueError("No frequency data found in dataframe. Columns 'mean_frequency' or 'frequency' required.")
    if frequency_range is not None:
        modal_data = \
            modal_data.copy().loc[
                (modal_data[frequency_col] >= frequency_range[0]) &
                (modal_data[frequency_col] <= frequency_range[1])
            ]
    # remove clusters with small size and very high damping as these are non-physical
    selected_data = selected_data[selected_data['size'] > min_size]
    if 'damping' in selected_data.columns:
        selected_data = selected_data[selected_data['damping'] < max_damping]
    elif 'mean_damping' in selected_data.columns:
        selected_data = selected_data[selected_data['mean_damping'] < max_damping]
    else:
        raise KeyError(
            "The modal data does not contain the column 'damping' or 'mean_damping'."
        )
    return selected_data


def column_multiplier(
    modal_data: pd.DataFrame,
    cols: list,
    multipliers: dict[str,float],
    index_divider: Union[float, None] = None
    ) -> pd.DataFrame:
    """Multiply the columns of a dataframe with the corresponding multiplier.

    Args:
        modal_data (pd.DataFrame): The modal data to be used for clustering.
        cols (list): The columns of the dataframe that should be used for clustering.
        multipliers (dict[str,float]): The multipliers for the columns.
        index_divider (Union[float, None], optional): The divider for the index dimension.
            Defaults to None.

    Returns:
        pd.DataFrame: The transformed dataframe through the multipliers.
    """    
    multiplied_data = modal_data.copy()
    # Remove timestamps as index to allow for time gaps in monitoring
    multiplied_data.reset_index(inplace=True)
    multiplied_data = multiplied_data[cols]
    for key in multipliers:
        if 'damping' in key:
            multiplied_data[key] = (multiplied_data[key] + 1)
        multiplied_data[key] = multiplied_data[key] * multipliers[key]
    # Include the index dimension to the clustering if index_divider is not None
    if index_divider is not None:
        multiplied_data["time_diff"] = (
            multiplied_data.index.astype(float) - multiplied_data.index.values[0].astype(float)
        ) / index_divider
    return multiplied_data


def predict_clusters(
    data: pd.DataFrame,
    min_cluster_size: int = 1000,
    target_column: str = "labels"
    ) -> pd.DataFrame:
    """Filter clusters based on their size and remove too small clusters (added to the noise).

    Args:
        data (pd.DataFrame): Data with cluster labels.
        min_cluster_size (int, optional): Minimum number of samples in a cluster. Defaults to 1000.
        target_column (str, optional): The column name for labels. Defaults to 'labels'.

    Returns:
        pd.DataFrame: Data with clusters having more than min_cluster_size samples.
    """
    labels_to_keep = []
    for label in data[target_column].unique():
        count = len(data[data[target_column] == label])
        if count > min_cluster_size:
            labels_to_keep.append(label)

    filtered_data = data[data[target_column].isin(labels_to_keep)]
    # Replace all labels that aren't in lbls with -1 (as noise)
    filtered_data.loc[~filtered_data["labels"].isin(labels_to_keep), "labels"] = -1
    # Reset the modes bigger than 0 to start from 0 keeping the previous order but filling the missing gaps
    # and keep the noise as -1
    non_noise_clusters = filtered_data[filtered_data["labels"] >= 0]
    codes, uniques = pd.factorize(non_noise_clusters["labels"])
    non_noise_clusters_factorized = non_noise_clusters.copy()
    non_noise_clusters_factorized["labels"] = codes
    filtered_data.loc[filtered_data["labels"] >= 0, :] = non_noise_clusters_factorized
    return filtered_data