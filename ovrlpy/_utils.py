import warnings
from collections.abc import Iterable
from queue import Empty, SimpleQueue
from typing import Any, Literal

import numpy as np
import polars as pl
from scipy.linalg import norm
from scipy.sparse import csr_array
from sklearn.decomposition import PCA
from sklearn.neighbors import radius_neighbors_graph
from umap import UMAP

from ._kde import find_local_maxima, kde_2d_discrete

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


def _transform_embeddings(expression, pca: PCA, umap_2d: UMAP, umap_rgb: UMAP):
    """fit the expression data into the umap embeddings after PCA transformation"""
    factors = pca.transform(expression)

    embedding = umap_2d.transform(factors)
    embedding_color = umap_rgb.transform(factors / norm(factors, axis=1, keepdims=True))

    return embedding, embedding_color


def _gaussian_weighted_neighbors(
    coords, radius: float, bandwidth: float, *, n_workers: int | None = None
) -> csr_array:
    """gaussian-weighted radius-based neighbors graph"""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names")
        neighbors = radius_neighbors_graph(
            coords, radius, mode="distance", include_self=True, n_jobs=n_workers
        )
    neighbors = csr_array(neighbors)
    neighbors.data = (1 / ((2 * np.pi) ** (3 / 2) * bandwidth**3)) * np.exp(
        -(neighbors.data**2) / (2 * bandwidth**2)
    )
    return neighbors


def _weighted_average_expression(
    genes: pl.Series,
    weights: np.ndarray | csr_array,
    gene_list: Iterable[str],
    normalize: bool = True,
) -> pl.DataFrame:
    """weighted average of the expression values of the neighbors"""

    assert weights.shape[0] == weights.shape[1] == len(genes)

    # drop name which makes having correct names after to_dummies easier
    genes = genes.rename("")
    # first cast to string for correct comparison
    genes = genes.cast(pl.String).cast(pl.Enum(gene_list), strict=False)
    # one-hot encoded genes
    genes_idx = genes.to_dummies(separator="").drop("null", strict=False)

    expression = pl.DataFrame(
        {gene: weights @ genes_idx[gene].to_numpy() for gene in genes_idx.columns}
    )

    if normalize:
        l2_norms = norm(expression.to_numpy(), axis=1)
        expression = expression.with_columns(pl.all().truediv(l2_norms))

    expression = expression.with_columns(
        pl.lit(0.0).alias(gene) for gene in gene_list if gene not in genes_idx.columns
    ).select(gene_list)

    return expression


def _gene_embedding(
    df: pl.DataFrame,
    mask: np.ndarray,
    factor: np.ndarray,
    xy: tuple[str, str] = ("x_pixel", "y_pixel"),
    **kwargs,
):
    """
    calculate top and bottom embedding

    Parameters
    ----------
    df : polars.DataFrame
        DataFrame of x, y, z, and z_center coordinates
    mask : numpy.ndarray
        binary mask for which pixels to calculate embedding
    factor : numpy.ndarray
        embedding weights
    """
    if len(df) < 2:
        return None, None

    x, y = xy

    # TODO: what happens if equal?
    top = df.select(xy).filter(df["z"] > df["z_center"])
    bottom = df.select(xy).filter(df["z"] < df["z_center"])

    if len(top) == 0:
        signal_top = None
    else:
        signal_top = kde_2d_discrete(
            top[x].to_numpy(), top[y].to_numpy(), size=mask.shape, **kwargs
        )[mask]
        signal_top = signal_top[:, None] * factor[None, :]

    if len(bottom) == 0:
        signal_bottom = None
    else:
        signal_bottom = kde_2d_discrete(
            bottom[x].to_numpy(), bottom[y].to_numpy(), size=mask.shape, **kwargs
        )[mask]
        signal_bottom = signal_bottom[:, None] * factor[None, :]

    return signal_top, signal_bottom


def _calculate_embedding(
    genes: SimpleQueue[tuple[int, pl.DataFrame]],
    mask: np.ndarray[tuple[int, int], np.dtype[np.bool_]],
    components: np.ndarray[tuple[int, int], np.dtype[np.floating]],
    **kwargs,
) -> tuple[np.ndarray | Literal[0], np.ndarray | Literal[0]]:
    embedding_top: Literal[0] | np.ndarray = 0
    embedding_bottom: Literal[0] | np.ndarray = 0

    while True:
        try:
            i, gene = genes.get(block=False)
        except Empty:
            break

        top, bottom = _gene_embedding(gene, mask, components[:, i], **kwargs)
        if top is not None:
            embedding_top += top
        if bottom is not None:
            embedding_bottom += bottom

    return embedding_top, embedding_bottom


def _cosine_similarity(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    norm_ = norm(x, axis=1) * norm(y, axis=1)
    norm_[norm_ == 0] = np.inf
    return np.sum(x * y, axis=1) / norm_
