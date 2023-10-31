"""
oma_clustering_dbscan.py is a module for clustering modes using HDBSCAN algorithm.
For more information, see https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html
-------------------------------------------------------------------------
Author: Maximillian Weil
"""
# TODO: Add automatic hyperparameter tuner
from dataclasses import dataclass, field
from typing import Union
import pandas as pd
from sklearn.cluster import HDBSCAN  # Importing HDBSCAN instead of DBSCAN
from oma_clustering.utils import data_selection, column_multiplier, validate_data, predict_clusters

@dataclass
class ModeClusterer_HDBSCAN:  # Renamed class to reflect the use of HDBSCAN
    """ModeClusterer is a class for clustering modal parameters using the HDBSCAN algorithm.

    Attributes:  # Attributes remain mostly the same, but some may be specific to HDBSCAN
        min_cluster_size (int): The minimum number of samples in a group for that group to be considered a cluster.
        min_samples (int, optional): The number of samples in a neighborhood for a point to be considered as a core point.
        metric (str): The metric to use for distance computation.
        ...
        hdbsc (HDBSCAN): The HDBSCAN object that stores the result of the clustering.
        hdbscan_data (pd.DataFrame): The dataframe that is used for clustering.
    """

    min_cluster_size: int = 5
    min_samples: Union[int, None] = None
    metric: str = 'euclidean'
    # ... other HDBSCAN specific parameters
    multipliers: dict[str, float] = field(default_factory=lambda: {"frequency": 40, "size": 0.5, "damping": 1})
    index_divider: Union[float, None] = None
    cols: list[str] = field(default_factory=lambda: ["frequency", "size", "damping"])
    min_size: float = 5.0
    max_damping: float = 5.0

    def __post_init__(self):
        self.hdbsc = HDBSCAN(min_cluster_size=self.min_cluster_size, min_samples=self.min_samples, metric=self.metric)  # HDBSCAN initialization
        self.hdbscan_data: pd.DataFrame = pd.DataFrame()
    
    def select_data(
        self,
        modal_data: pd.DataFrame,
        frequency_range: Union[tuple[float, float], None]
        ) -> pd.DataFrame:
        """Select the data to be used for clustering.
        Set the selected_data as the class dbscan_data.

        Args:
            modal_data (pd.DataFrame): The modal data to be used for clustering.
            frequency_range (Union[tuple[float, float], None]): The frequency range to be used for clustering.

        Raises:
            ValueError: If no frequency data is found in the dataframe.

        Returns:
            pd.DataFrame: The data to be used for clustering.
        """
        # select the data to be used for clustering
        hdbscan_data = \
            data_selection(
                modal_data,
                self.cols,
                self.min_size,
                self.max_damping,
                frequency_range
            )
        self.hdbscan_data = hdbscan_data
        return hdbscan_data
    
    def fit_hdbscan(self, processed_data: pd.DataFrame) -> None:
        """Fit the processed_data to the HDBSCAN algorithm.

        Args:
            processed_data (pd.DataFrame): _description_
        """        
        self.hdbsc.fit(processed_data)
        self.hdbscan_data["labels"] = self.hdbsc.labels_

    # Modified fit method
    def fit(
        self,
        modal_data: pd.DataFrame,
        frequency_range: Union[tuple[float, float], None] = None
        ):
        """Fit the modal_data to the DBSCAN algorithm

        Args:
            modal_data (pd.DataFrame): The modal data to be fitted.
            frequency_range (Union[tuple[float, float], None], optional): The frequency range to be used for clustering.
                Defaults to None.
        """
        validate_data(self.cols, modal_data)
        hdbscan_data = self.select_data(modal_data, frequency_range)
        self.processed_data = column_multiplier(
                hdbscan_data,
                self.cols,
                self.multipliers,
                self.index_divider
            )
        self.fit_hdbscan(self.processed_data)

    def predict(self, min_cluster_size: int = 1000) -> pd.DataFrame:
        """Predict the clusters of the fitted data
        that have more clusters than the min_cluster_size.

        Args:
            min_cluster_size (int, optional): The minimum number of modes in a cluster.
                Defaults to 1000.

        Returns:
            pd.DataFrame: The dataframe with the predicted clusters.
        """
        return predict_clusters(self.hdbscan_data, min_cluster_size)
