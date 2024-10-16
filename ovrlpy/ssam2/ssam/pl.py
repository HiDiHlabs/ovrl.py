# ssam2 plotting library, compatible with the squidpy plot handling. 

import matplotlib.pyplot as plt
# impor linear color map:
import matplotlib.colors as colors


def celltype_map(adata,color=None,ax=None):
    """
    Plots a cell type map of the data.
    """

    celltype_map = adata.uns['ssam']['ct_map_filtered']


    if ax is None:
        fig, ax = plt.subplots(figsize=(10,10))
    else:
        fig = ax.get_figure()

    # plot_classes = 