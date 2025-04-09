from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.linalg import norm
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from umap import UMAP

from ._kde import find_local_maxima, kde_2d

UMAP_2D_PARAMS: dict[str, Any] = {"n_components": 2, "n_neighbors": 20, "min_dist": 0}
"""Default 2D-UMAP parameters"""

UMAP_RGB_PARAMS: dict[str, Any] = {"n_components": 3, "n_neighbors": 10, "min_dist": 0}
"""Default RGB-UMAP parameters"""


def _determine_localmax_and_sample(
    distribution, min_distance: int = 3, min_expression: float = 5
):
    """
    Returns a list of local maxima in a kde of the data frame.

    Parameters
    ----------
    distribution : np.ndarray
        A 2D array of the distribution.
    min_distance : int, optional
        The minimum distance between local maxima.
    min_expression : float, optional
        The minimum expression level to include in the histogram.

    Returns
    -------
    rois_x
        x coordinates of local maxima.
    rois_y
        y coordinates of local maxima.
    values
        values at local maxima.
    """

    rois = find_local_maxima(distribution, min_distance, min_expression)

    rois_x = rois[:, 0]
    rois_y = rois[:, 1]

    return rois_x, rois_y, distribution[rois_x, rois_y]


## These functions are going to be separated into a package of their own at some point:

# define a 45-degree 3D rotation matrix
_ROTATION_MATRIX = np.array(
    [
        [0.500, 0.500, -0.707],
        [-0.146, 0.854, 0.500],
        [0.854, -0.146, 0.500],
    ]
)


def _fill_color_axes(rgb, pca: PCA, *, fit: bool = False) -> np.ndarray:
    # rotate the transformed data 45° in all the dimensions
    if fit:
        pca.fit(rgb)
    return np.dot(pca.transform(rgb), _ROTATION_MATRIX)


# normalize array
def _min_to_max(arr: np.ndarray):
    arr_min = arr.min(0, keepdims=True)
    arr_max = arr.max(0, keepdims=True)
    arr = arr - arr_min
    arr /= arr_max - arr_min
    return arr


# define a function that fits expression data to into the umap embeddings
def _transform_embeddings(expression, pca: PCA, embedder_2d: UMAP, embedder_3d: UMAP):
    factors = pca.transform(expression)

    embedding = embedder_2d.transform(factors)
    embedding_color = embedder_3d.transform(factors / norm(factors, axis=1)[..., None])

    return embedding, embedding_color


# define a function that subsamples spots around x,y given a window size
def _spatial_subset_mask(
    coordinates: pd.DataFrame, x: float, y: float, window_size: float = 5
):
    return (
        (coordinates["x"] > x - window_size)
        & (coordinates["x"] < x + window_size)
        & (coordinates["y"] > y - window_size)
        & (coordinates["y"] < y + window_size)
    )


# define a function that returns the k nearest neighbors of x,y
def _create_knn_graph(coords, k: int = 10):
    nbrs = NearestNeighbors(n_neighbors=k, algorithm="ball_tree").fit(coords)
    distances, indices = nbrs.kneighbors(coords)
    return distances, indices


# get a kernel-weighted average of the expression values of the k nearest neighbors of x,y
def _knn_expression(
    gene_idx: NDArray[np.integer],
    distances: np.ndarray,
    neighbor_indices: np.ndarray,
    gene_list: Iterable,
    bandwidth: float = 2.5,
) -> pd.DataFrame:
    weights = (1 / ((2 * np.pi) ** (3 / 2) * bandwidth**3)) * np.exp(
        -(distances**2) / (2 * bandwidth**2)
    )

    return pd.DataFrame(
        {
            gene: ((gene_idx[neighbor_indices] == i) * weights).sum(axis=1)
            for i, gene in enumerate(gene_list)
        }
    )


def _compute_embedding_vectors(
    df: np.ndarray, mask: np.ndarray, factor: np.ndarray, **kwargs
):
    if len(df) < 2:
        return None, None

    # TODO: what happens if equal?
    top = df[df[:, 2] > df[:, 3], :2]
    bottom = df[df[:, 2] < df[:, 3], :2]

    if len(top) == 0:
        signal_top = None
    else:
        signal_top = kde_2d(top, size=mask.shape, **kwargs)[mask]
        signal_top = signal_top[:, None] * factor[None, :]
    if len(bottom) == 0:
        signal_bottom = None
    else:
        signal_bottom = kde_2d(bottom, size=mask.shape, **kwargs)[mask]
        signal_bottom = signal_bottom[:, None] * factor[None, :]

    return signal_top, signal_bottom
