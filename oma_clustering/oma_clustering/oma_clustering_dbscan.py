"""
oma_clustering_dbscan.py is a module for clustering modes using DBSCAN algorithm.
For more information, see https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
-------------------------------------------------------------------------
Author: Maximillian Weil
"""
#TODO: Add automatic hyperparmater tuner
from dataclasses import dataclass, field
from typing import Union
import pandas as pd
from sklearn.cluster import DBSCAN
from oma_clustering.utils import data_selection, column_multiplier, validate_data, predict_clusters


@dataclass
class ModeClusterer_DBSCAN:
    """ModeClusterer is a class for clustering modal parameters using the DBSCAN algorithm.

    Attributes:
        eps (float): The maximum distance between two samples
            for them to be considered as in the same neighborhood.
        min_samples (int): The number of samples (or total weight) in a neighborhood,
            for a point to be considered as a core point. This includes the point itself.
        freq_multiplier (float): The multiplier for the frequency feature.
        damping_multiplier (float): The multiplier for the damping feature.
        size_multiplier (float): The multiplier for the size feature.
        cols (list): The columns of the dataframe that should be used for clustering.
        min_modal_size (float): The minimum size of a mode to be considered for clustering.
        max_modal_damping (float): The maximum damping of a mode to be considered for clustering.
        dbsc (DBSCAN): The DBSCAN object that stores the result of the clustering.
        dbscan_data (pd.DataFrame): The dataframe that is used for clustering.
    """

    eps: float = 5
    min_samples: int = 100
    multipliers: dict[str,float] = \
        field(default_factory=lambda: {"frequency": 40, "size": 0.5, "damping": 1})
    index_divider: Union[float, None] = None
    cols: list[str] = \
        field(default_factory=lambda: ["frequency", "size", "damping"])
    min_size: float = 5.0
    max_damping: float = 5.0

    def __post_init__(self):
        self.dbsc = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        self.dbscan_data: pd.DataFrame = pd.DataFrame()
    
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
        dbscan_data = \
            data_selection(
                modal_data,
                self.cols,
                self.min_size,
                self.max_damping,
                frequency_range
            )
        self.dbscan_data = dbscan_data
        return dbscan_data
    
    def fit_dbscan(self, processed_data: pd.DataFrame) -> None:
        """Fit the processed_data to the DBSCAN algorithm.

        Args:
            processed_data (pd.DataFrame): _description_
        """        
        self.dbsc.fit(processed_data)
        self.dbscan_data["labels"] = self.dbsc.labels_

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
        dbscan_data = self.select_data(modal_data, frequency_range)
        self.processed_data = column_multiplier(
                dbscan_data,
                self.cols,
                self.multipliers,
                self.index_divider
            )
        self.fit_dbscan(self.processed_data)

    def predict(self, min_cluster_size: int = 1000) -> pd.DataFrame:
        """Predict the clusters of the fitted data
        that have more clusters than the min_cluster_size.

        Args:
            min_cluster_size (int, optional): The minimum number of modes in a cluster.
                Defaults to 500.

        Returns:
            pd.DataFrame: The dataframe with the predicted clusters.
        """
        return predict_clusters(self.dbscan_data, min_cluster_size)