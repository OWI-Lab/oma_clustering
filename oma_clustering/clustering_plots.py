"""
clustering_plots.py is a module for helper functions
when plotting the resulting clustering of the modal parameters.
-------------------------------------------------------------------------
Author: Maximillian Weil
"""

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

def generate_colormap(nr_labels):
    """
    Generates a colormap based on the first `nr_labels` colors of 'tab20'
    and repeats colors if `nr_labels` > 20.

    Parameters:
        nr_labels (int): The number of unique labels or colors required.

    Returns:
        ListedColormap: A colormap with `nr_labels` colors.
    """

    tab20 = plt.get_cmap('tab20') # type: ignore
    
    if nr_labels <= 20:
        return ListedColormap(tab20(range(nr_labels)))

    else:
        original_colors = tab20(np.arange(20))
        n_repeats, remainder = divmod(nr_labels, 20)

        repeated_colors = np.tile(original_colors, (n_repeats, 1))
        remaining_colors = original_colors[:remainder]

        all_colors = np.vstack((repeated_colors, remaining_colors))
        
        return ListedColormap(all_colors) # type: ignore