from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
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
    values: np.ndarray, min_distance: int = 3, min_value: float = 5
):
    """
    Returns a list of local maxima and their corresponding values.

    Parameters
    ----------
    values : np.ndarray
        A 2D array of values.
    min_distance : int, optional
        The minimum distance between local maxima.
    min_value : float, optional
        The minimum value to consider values as maxima.

    Returns
    -------
    rois_x
        x coordinates of local maxima.
    rois_y
        y coordinates of local maxima.
    values
        values at local maxima.
    """
    rois = find_local_maxima(values, min_distance, min_value)

    rois_x = rois[:, 0]
    rois_y = rois[:, 1]

    return rois_x, rois_y, values[rois_x, rois_y]


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
    """rotate the transformed data 45° in all dimensions"""
    if fit:
        pca.fit(rgb)
    return np.dot(pca.transform(rgb), _ROTATION_MATRIX)


def _minmax_scaling(x: np.ndarray):
    """scale features (rows) to unit range"""
    x_min = x.min(axis=0)
    x_max = x.max(axis=0)
    return (x - x_min) / (x_max - x_min)


def _transform_embeddings(expression, pca: PCA, embedder_2d: UMAP, embedder_3d: UMAP):
    """fit the expression data into the umap embeddings after PCA transformation"""
    factors = pca.transform(expression)

    embedding = embedder_2d.transform(factors)
    embedding_color = embedder_3d.transform(factors / norm(factors, axis=1)[..., None])

    return embedding, embedding_color


def _create_knn_graph(coords, k: int = 10):
    """k nearest neighbors distances and indices"""
    nbrs = NearestNeighbors(n_neighbors=k, algorithm="ball_tree").fit(coords)
    distances, indices = nbrs.kneighbors(coords)
    return distances, indices


def _knn_expression(
    gene_idx: NDArray[np.integer],
    distances: np.ndarray,
    neighbor_indices: np.ndarray,
    gene_list: Iterable,
    bandwidth: float = 2.5,
) -> pd.DataFrame:
    """kernel-weighted average of the expression values of the k nearest neighbors"""
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
    df: pl.DataFrame, mask: np.ndarray, factor: np.ndarray, **kwargs
):
    """
    calculate top and bottom embedding

    Parameters
    ----------
    df : polars.DataFrame
        DataFrame of x, y, z and z_delim coordinates
    mask : numpy.ndarray
        binary mask for which pixels to calculate embedding
    factor : numpy.ndarray
        embedding weights
    """
    if len(df) < 2:
        return None, None

    # TODO: what happens if equal?
    top = df.filter(pl.col("z") > pl.col("z_delim")).select(["x", "y"])
    bottom = df.filter(pl.col("z") < pl.col("z_delim")).select(["x", "y"])

    if len(top) == 0:
        signal_top = None
    else:
        signal_top = kde_2d(
            top["x"].to_numpy(), top["y"].to_numpy(), size=mask.shape, **kwargs
        )[mask]
        signal_top = signal_top[:, None] * factor[None, :]

    if len(bottom) == 0:
        signal_bottom = None
    else:
        signal_bottom = kde_2d(
            bottom["x"].to_numpy(), bottom["y"].to_numpy(), size=mask.shape, **kwargs
        )[mask]
        signal_bottom = signal_bottom[:, None] * factor[None, :]

    return signal_top, signal_bottom


def _cosine_similarity(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    norm_ = np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1)
    norm_[norm_ == 0] = np.inf
    return np.sum(x * y, axis=1) / norm_
