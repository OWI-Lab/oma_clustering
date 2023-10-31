"""
oma_clustering_dbscan.py is a module for clustering modes using HDBSCAN algorithm.
For more information, see https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html
-------------------------------------------------------------------------
Author: Maximillian Weil
"""
from dataclasses import dataclass, field
from typing import Union
#import hdbscan
import pandas as pd
from sklearn.cluster import HDBSCAN
from oma_clustering.utils import data_selection, column_multiplier, check_columns


@dataclass()
class ModeClusterer_HDBSCAN:
    """ModeClusterer is a class for clustering mode parameters using HDBSCAN algorithm.

    Attributes:
        freq_multiplier (float): The multiplier for the frequency feature.
        damping_multiplier (float): The multiplier for the damping feature.
        size_multiplier (float): The multiplier for the size feature.
        cols (list): The columns of the dataframe that should be used for clustering.
        min_modal_size (float): The minimum size of a mode to be considered for clustering.
        max_modal_damping (float): The maximum damping of a mode to be considered for clustering.
        dbsc (DBSCAN): The DBSCAN object that stores the result of the clustering.
        dbscan_data (pd.DataFrame): The dataframe that is used for clustering.
    """
    min_cluster_size: int = 100
    min_samples: Union[int, None] = None
    multipliers: dict[str,float] = \
        field(default_factory=lambda: {"frequency": 40, "size": 0.5, "damping": 1})
    index_divider: Union[float, None] = None
    cols: list[str] = \
        field(default_factory=lambda: ["frequency", "size", "damping"])
    min_size: float = 5.0
    max_damping: float = 5.0

    def __post_init__(self):
        self.hdbsc = \
            HDBSCAN(
                min_cluster_size = self.min_cluster_size,
                min_samples = self.min_samples
            )
        self.hdbscan_data: pd.DataFrame = pd.DataFrame()
        

    def fit(
        self,
        modal_data: pd.DataFrame,
        frequency_range: Union[tuple[float, float], None] = None,
        **kwargs
        ):
        """Fit the modal_data to the HDBSCAN algorithm
        for the time period between start_time and end_time.

        Args:
            modal_data (pd.DataFrame): The modal data to be fitted.
        """
        if not check_columns(self.cols, modal_data):
            raise ValueError(
                "The modal data does not contain all the required columns."
            )
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
        hdbscan_data = \
            data_selection(
                modal_data,
                self.cols,
                self.min_size,
                self.max_damping
            )

        ## feature construction
        multiplied_dbscan_data = \
            column_multiplier(
                hdbscan_data,
                self.cols,
                self.multipliers,
                self.index_divider
            )

        self.hdbsc = \
            HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                min_samples=self.min_samples,
                **kwargs
            ).fit(multiplied_dbscan_data[self.cols])
        hdbscan_data["labels"] = self.hdbsc.labels_
        self.hdbscan_data = hdbscan_data

    def predict(self, min_cluster_size: int = 500) -> pd.DataFrame:
        """Predict the clusters of the fitted data
        that have more clusters than the min_cluster_size.

        Args:
            min_cluster_size (int, optional): The minimum number of modes in a cluster.
                Defaults to 500.

        Returns:
            pd.DataFrame: The dataframe with the predicted clusters.
        """

        clustered_data = self.hdbscan_data.copy()
        # Remove clusters with less than min_cluster_size samples
        lbls = []
        for label in self.hdbscan_data["labels"].unique():
            cnt = len(self.hdbscan_data[self.hdbscan_data["labels"] == label])
            if cnt > min_cluster_size:
                lbls.append(label)

        clustered_data = self.hdbscan_data[self.hdbscan_data["labels"].isin(lbls)][
            self.hdbscan_data[self.hdbscan_data["labels"].isin(lbls)]["labels"] >= 0
        ]
        # Reset the modes to start from 0
        codes, uniques = pd.factorize(clustered_data["labels"])
        clustered_data["labels"] = codes
        return clustered_data
    
    def predict_with_noise(self, min_cluster_size: int = 500) -> pd.DataFrame:
        """Predict the clusters of the fitted data
        that have more clusters than the min_cluster_size.
        Keep the noise as -1.

        Args:
            min_cluster_size (int, optional): The minimum number of modes in a cluster.
                Defaults to 500.

        Returns:
            pd.DataFrame: The dataframe with the predicted clusters and the noise as -1.
        """

        clustered_data = self.hdbscan_data.copy()
        # Remove clusters with less than min_cluster_size samples
        lbls = []
        for label in self.hdbscan_data["labels"].unique():
            cnt = len(self.hdbscan_data[self.hdbscan_data["labels"] == label])
            if cnt > min_cluster_size:
                lbls.append(label)

        # Replace all labels that aren't in lbls with -1
        clustered_data.loc[~clustered_data["labels"].isin(lbls), "labels"] = -1

        # Reset the modes to start from -1
        codes, uniques = pd.factorize(clustered_data["labels"])
        clustered_data["labels"] = codes
        return clustered_data