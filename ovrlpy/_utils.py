from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from math import ceil
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
from matplotlib.axes import Axes
from matplotlib_scalebar.scalebar import ScaleBar
from scipy.linalg import norm
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from ._patching import _patches, n_patches
from ._ssam2 import find_local_maxima, kde_2d

SCALEBAR_PARAMS: dict[str, Any] = {"dx": 1, "units": "um"}
"""Default scalebar parameters"""

UMAP_2D_PARAMS: dict[str, Any] = {
    "n_components": 2,
    "n_neighbors": 20,
    "min_dist": 0,
    "random_state": 42,
}
"""Default 2D-UMAP parameters"""

UMAP_RGB_PARAMS: dict[str, Any] = {
    "n_components": 3,
    "n_neighbors": 10,
    "min_dist": 0,
    "random_state": 42,
}
"""Default RGB-UMAP parameters"""


def _plot_scalebar(ax: Axes, dx: float = 1, units="um", **kwargs):
    ax.add_artist(ScaleBar(dx, units=units, **kwargs))


def _determine_localmax_and_sample(distribution, min_distance=3, min_expression=5):
    """
    Returns a list of local maxima in a kde of the data frame.

    Parameters
    ----------
    distribution : np.array
        A 2d array of the distribution.
    min_distance : int, optional
        The minimum distance between local maxima. The default is 3.
    min_expression : int, optional
        The minimum expression level to include in the histogram. The default is 5.

    Returns
    -------
    rois_x : list
        A list of x coordinates of local maxima.
    rois_y : list
        A list of y coordinates of local maxima.

    """

    rois = find_local_maxima(distribution, min_distance, min_expression)

    rois_x = rois[:, 0]
    rois_y = rois[:, 1]

    return rois_x, rois_y, distribution[rois_x, rois_y]


## These functions are going to be seperated into a package of their own at some point:

# define a 45-degree 3D rotation matrix
_ROTATION_MATRIX = np.array(
    [
        [0.500, 0.500, -0.707],
        [-0.146, 0.854, 0.500],
        [0.854, -0.146, 0.500],
    ]
)


def _fill_color_axes(rgb, dimred=None):
    if dimred is None:
        dimred = PCA(n_components=3).fit(rgb)

    facs = dimred.transform(rgb)

    # rotate the facs 45 in all the dimensions
    facs = np.dot(facs, _ROTATION_MATRIX)

    return facs, dimred


# normalize array:
def _min_to_max(arr, arr_min=None, arr_max=None):
    if arr_min is None:
        arr_min = arr.min(0, keepdims=True)
    if arr_max is None:
        arr_max = arr.max(0, keepdims=True)
    arr = arr - arr_min
    arr /= arr_max - arr_min
    return arr


# define a function that fits expression data to into the umap embeddings:
def _transform_embeddings(expression, pca, embedder_2d, embedder_3d):
    factors = pca.transform(expression)

    embedding = embedder_2d.transform(factors)
    embedding_color = embedder_3d.transform(factors / norm(factors, axis=1)[..., None])

    return embedding, embedding_color


# define a function that plots the embeddings, with celltype centers rendered as plt.texts on top:
def _plot_embeddings(
    embedding,
    embedding_color,
    celltype_centers,
    celltypes,
    rasterized=False,
    ax=None,
    scatter_kwargs={"alpha": 0.1, "marker": "."},
):
    colors = embedding_color.copy()  # np.clip(embedding_color.copy(),0,1)

    if ax is None:
        ax = plt.gca()

    ax.axis("off")

    alpha = scatter_kwargs.pop("alpha", 0.1)
    marker = scatter_kwargs.pop("marker", ".")

    ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=(colors),
        alpha=alpha,
        marker=marker,
        rasterized=rasterized,
        **scatter_kwargs,
    )

    text_artists = []
    for i, celltype in enumerate(celltypes):
        if not np.isnan(celltype_centers[i, 0]):
            t = ax.text(
                np.nan_to_num((celltype_centers[i, 0])),
                np.nan_to_num(celltype_centers[i, 1]),
                celltype,
                color="k",
                fontsize=12,
            )
            text_artists.append(t)

    _untangle_text(text_artists, ax)


def _untangle_text(text_artists, ax=None, max_iterations=10000):
    if ax is None:
        ax = plt.gca()
    inv = ax.transData.inverted()

    artist_coords = np.array(
        [text_artist.get_position() for text_artist in text_artists]
    )
    artist_coords = artist_coords + np.random.normal(0, 0.001, artist_coords.shape)
    artist_extents = [text_artist.get_window_extent() for text_artist in text_artists]
    artist_extents = np.array(
        [inv.transform(extent.get_points()) for extent in artist_extents]
    )
    artist_extents = artist_extents[:, 1] - artist_extents[:, 0]

    for i in range(max_iterations):
        relative_positions_x = (
            artist_coords[:, 0][:, None] - artist_coords[:, 0][None, :]
        )
        relative_positions_y = (
            artist_coords[:, 1][:, None] - artist_coords[:, 1][None, :]
        )

        relative_positions_x /= (
            0.1 + (artist_extents[:, 0][:, None] + artist_extents[:, 0][None, :]) / 2
        )
        relative_positions_y /= (
            0.1 + (artist_extents[:, 1][:, None] + artist_extents[:, 1][None, :]) / 2
        )

        # distances = np.sqrt(relative_positions_x**2+relative_positions_y**2)
        distances = np.abs(relative_positions_x) + np.abs(relative_positions_y)

        gaussian_repulsion = 1 * np.exp(-distances / 0.5)

        velocities_x = np.zeros_like(relative_positions_x)
        velocities_y = np.zeros_like(relative_positions_y)

        velocities_x[distances > 0] = (
            gaussian_repulsion[distances > 0]
            * relative_positions_x[distances > 0]
            / distances[distances > 0]
        )
        velocities_y[distances > 0] = (
            gaussian_repulsion[distances > 0]
            * relative_positions_y[distances > 0]
            / distances[distances > 0]
        )

        velocities_x[np.eye(velocities_x.shape[0], dtype=bool)] = 0
        velocities_y[np.eye(velocities_y.shape[0], dtype=bool)] = 0

        delta = np.stack([velocities_x, velocities_y], axis=1).mean(-1)
        # # delta = delta.clip(-0.1,0.1)
        artist_coords = artist_coords + delta * 0.1
        # artist_coords  = artist_coords*0.9 + initial_artist_coords*0.1

    for i, text_artist in enumerate(text_artists):
        text_artist.set_position(artist_coords[i, :])


# define a function that subsamples spots around x,y given a window size:
def _get_spatial_subsample_mask(coordinate_df, x, y, plot_window_size=5):
    return (
        (coordinate_df.x > x - plot_window_size)
        & (coordinate_df.x < x + plot_window_size)
        & (coordinate_df.y > y - plot_window_size)
        & (coordinate_df.y < y + plot_window_size)
    )


# define a function that returns the k nearest neighbors of x,y:
def _create_knn_graph(coords, k=10):
    nbrs = NearestNeighbors(n_neighbors=k, algorithm="ball_tree").fit(coords)
    distances, indices = nbrs.kneighbors(coords)
    return distances, indices


# get a kernel-weighted average of the expression values of the k nearest neighbors of x,y:
def _get_knn_expression(distances, neighbor_indices, genes, gene_labels, bandwidth=2.5):
    weights = (1 / ((2 * np.pi) ** (3 / 2) * bandwidth**3)) * np.exp(
        -(distances**2) / (2 * bandwidth**2)
    )
    local_expression = pd.DataFrame(
        index=genes, columns=np.arange(distances.shape[0])
    ).astype(float)

    for i, gene in enumerate(genes):
        weights_ = weights.copy()
        weights_[(gene_labels[neighbor_indices]) != i] = 0
        local_expression.loc[gene, :] = weights_.sum(1)

    return local_expression


def _create_histogram(
    df,
    genes=None,
    min_expression: float = 0,
    KDE_bandwidth=None,
    x_max=None,
    y_max=None,
):
    """
    Creates a 2d histogram of the data frame's [x,y] coordinates.

    Parameters
    ----------
    df : pd.DataFrame
        A dataframe of coordinates.
    genes : list, optional
        A list of genes to include in the histogram. The default is None.
    min_expression : float, optional
        The minimum expression level to include in the histogram.
    KDE_bandwidth : int, optional
        The bandwidth of the gaussian blur applied to the histogram.
    x_max :
        TODO
    y_max :
        TODO

    Returns
    -------
    hist : np.array
        A 2d array of the histogram.

    """
    if genes is None:
        genes = df["gene"].unique()

    if x_max is None:
        x_max = df["x_pixel"].max()
    if y_max is None:
        y_max = df["y_pixel"].max()

    df = df[df["gene"].isin(genes)].copy()

    hist, *_ = np.histogram2d(
        df["x_pixel"], df["y_pixel"], bins=[np.arange(x_max + 2), np.arange(y_max + 2)]
    )

    if KDE_bandwidth is not None:
        hist = gaussian_filter(hist, sigma=KDE_bandwidth)

    hist[hist < min_expression] = 0

    return hist


def _compute_embedding_vectors(df: pd.DataFrame, mask: np.ndarray, factor, **kwargs):
    if len(df) < 2:
        return None, None

    # TODO: what happens if equal?
    top = df[df[:, 2] > df[:, 3]]
    bottom = df[df[:, 2] < df[:, 3]]

    if len(top) == 0:
        signal_top = 0
    else:
        signal_top = kde_2d(top[:, :2], size=mask.shape, **kwargs)[mask]
        signal_top = signal_top[:, None] * factor[None]
    if len(bottom) == 0:
        signal_bottom = 0
    else:
        signal_bottom = kde_2d(bottom[:, :2], size=mask.shape, **kwargs)[mask]
        signal_bottom = signal_bottom[:, None] * factor[None]

    return signal_top, signal_bottom


def _compute_divergence_patched(
    df: pd.DataFrame,
    gene_list: Iterable,
    pca_component_matrix: np.ndarray,
    min_expression: float = 2,
    KDE_bandwidth: float = 1,
    patch_length: int = 500,
    n_workers: int = 8,
    dtype=np.float32,
):
    # the 4 comes from the default truncate in scipy.ndimage.gaussian_filter
    padding = int(ceil(4 * KDE_bandwidth))

    n_components = pca_component_matrix.shape[0]

    signal = kde_2d(df[["x", "y"]].values, bandwidth=KDE_bandwidth, dtype=dtype)

    cosine_similarity = np.zeros_like(signal)

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        for patch_df, offset, size in tqdm.tqdm(
            _patches(df, patch_length, padding, size=signal.shape),
            total=n_patches(patch_length, signal.shape),
        ):
            if len(patch_df) == 0:
                continue

            patch_signal = kde_2d(
                patch_df[["x", "y"]].values, bandwidth=KDE_bandwidth, dtype=dtype
            )

            patch_signal_mask = patch_signal > min_expression
            n_pixels = patch_signal_mask.sum()

            if n_pixels == 0:
                continue

            patch_embedding_top = np.zeros((n_pixels, n_components), dtype=dtype)
            patch_embedding_bottom = np.zeros((n_pixels, n_components), dtype=dtype)

            df_gene_grouped = patch_df.groupby("gene", observed=False).apply(
                lambda x: x[["x", "y", "z", "z_delim"]].values
            )

            fs = set(
                executor.submit(
                    _compute_embedding_vectors,
                    df_gene_grouped.loc[gene],
                    patch_signal_mask,
                    pca_component_matrix[:, i],
                    bandwidth=KDE_bandwidth,
                    dtype=dtype,
                )
                for i, gene in enumerate(gene_list)
            )

            for f in as_completed(fs):
                top_, bottom_ = f.result()

                if top_ is not None:
                    assert bottom_ is not None
                    patch_embedding_top += top_
                    patch_embedding_bottom += bottom_

                fs.remove(f)

            patch_norm_top = np.linalg.norm(patch_embedding_top, axis=1)
            patch_norm_bottom = np.linalg.norm(patch_embedding_bottom, axis=1)

            spatial_patch_cosine_similarity = np.zeros_like(patch_signal)
            spatial_patch_cosine_similarity[patch_signal_mask] = np.sum(
                patch_embedding_top * patch_embedding_bottom, axis=1
            ) / (patch_norm_top * patch_norm_bottom)
            # remove padding
            spatial_patch_cosine_similarity = spatial_patch_cosine_similarity[
                padding : padding + size[0], padding : padding + size[1]
            ]

            x_pad = offset[0] + padding
            y_pad = offset[1] + padding
            cosine_similarity[
                x_pad : x_pad + spatial_patch_cosine_similarity.shape[0],
                y_pad : y_pad + spatial_patch_cosine_similarity.shape[1],
            ] = spatial_patch_cosine_similarity

    return cosine_similarity, signal
