from concurrent.futures import ThreadPoolExecutor, as_completed

# create circular kernel:
# draw outlines around artist:
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
from scipy.ndimage import gaussian_filter, maximum_filter
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from .ssam2 import utils as ssam_utils


def draw_outline(ax, artist, lw=2, color="black"):
    _ = artist.set_path_effects(
        [PathEffects.withStroke(linewidth=lw, foreground=color), PathEffects.Normal()]
    )


def plot_scalebar(
    ax,
    x,
    y,
    length=100,
    fontsize=10,
    text="100um",
    color="k",
    text_offset=0,
    edge_color=None,
):
    plot_artist = ax.plot([x, x + length], [y, y], c=color, lw=2)
    text_artist = ax.text(
        x + length / 2,
        y + text_offset,
        text,
        fontsize=fontsize,
        ha="center",
        va="bottom",
        c=color,
    )

    if edge_color is not None:
        draw_outline(ax, plot_artist[0], lw=5, color=edge_color)
        draw_outline(ax, text_artist, lw=5, color=edge_color)

    return plot_artist, text_artist


def create_circular_kernel(r):
    """
    Creates a circular kernel of radius r.
    Parameters
    ----------
    r : int
        The radius of the kernel.

    Returns
    -------
    kernel : np.array
        A 2d array of the circular kernel.

    """

    span = np.linspace(-1, 1, r * 2)
    X, Y = np.meshgrid(span, span)
    return (X**2 + Y**2) ** 0.5 <= 1


def get_kl_divergence(p, q):
    # mask = (p!=0) * (q!=0)
    output = np.zeros(p.shape)
    # output[mask] = p[mask]*np.log(p[mask]/q[mask])
    output[:] = p[:] * np.log(p[:] / q[:])
    return output


def determine_localmax(distribution, min_distance=3, min_expression=5):
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
    localmax_kernel = create_circular_kernel(min_distance)
    localmax_projection = distribution == maximum_filter(
        distribution, footprint=localmax_kernel
    )

    rois_x, rois_y = np.where((distribution > min_expression) & localmax_projection)

    return rois_x, rois_y, distribution[rois_x, rois_y]


## These functions are going to be seperated into a package of their own at some point:


def fill_color_axes(rgb, dimred=None):
    if dimred is None:
        dimred = PCA(n_components=3).fit(rgb)

    facs = dimred.transform(rgb)

    # rotate the ica_facs 45 in all the dimensions:
    # define a 45-degree 3d rotation matrix
    # (0.500 | 0.500 | -0.707
    # -0.146 | 0.854 | 0.500
    # 0.854 | -0.146 | 0.500)
    rotation_matrix = np.array(
        [
            [0.500, 0.500, -0.707],
            [-0.146, 0.854, 0.500],
            [0.854, -0.146, 0.500],
        ]
    )

    # rotate the facs:
    facs = np.dot(facs, rotation_matrix)

    return facs, dimred


# normalize array:
def min_to_max(arr, arr_min=None, arr_max=None):
    if arr_min is None:
        arr_min = arr.min(0, keepdims=True)
    if arr_max is None:
        arr_max = arr.max(0, keepdims=True)
    arr = arr - arr_min
    arr /= arr_max - arr_min
    return arr


# define a function that fits expression data to into the umap embeddings:
def transform_embeddings(
    expression, pca, embedder_2d, embedder_3d, colors_min_max=[None, None]
):
    factors = pca.transform(expression)

    embedding = embedder_2d.transform(factors)
    embedding_color = embedder_3d.transform(factors)
    # embedding_color = embedder_3d.transform(embedding)

    # embedding_color = min_to_max(embedding_color,colors_min_max[0],colors_min_max[1])

    return embedding, embedding_color


# define a function that plots the embeddings, with celltype centers rendered as plt.texts on top:
def plot_embeddings(
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

    if "alpha" not in scatter_kwargs:
        alpha = 0.1
    else:
        alpha = scatter_kwargs["alpha"]
        scatter_kwargs.pop("alpha")

    if "marker" not in scatter_kwargs:
        marker = "."
    else:
        marker = scatter_kwargs["marker"]
        scatter_kwargs.pop("marker")

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
    for i in range(len(celltypes)):
        t = ax.text(
            np.nan_to_num((celltype_centers[i, 0])),
            np.nan_to_num(celltype_centers[i, 1]),
            celltypes[i],
            color="k",
            fontsize=12,
        )
        text_artists.append(t)

    untangle_text(text_artists, ax)


def untangle_text(text_artists, ax=None, max_iterations=10000):
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
def get_spatial_subsample_mask(coordinate_df, x, y, plot_window_size=5):
    return (
        (coordinate_df.x > x - plot_window_size)
        & (coordinate_df.x < x + plot_window_size)
        & (coordinate_df.y > y - plot_window_size)
        & (coordinate_df.y < y + plot_window_size)
    )


# define a function that returns the k nearest neighbors of x,y:
def create_knn_graph(coords, k=10):
    nbrs = NearestNeighbors(n_neighbors=k, algorithm="ball_tree").fit(coords)
    distances, indices = nbrs.kneighbors(coords)
    return distances, indices


# get a kernel-weighted average of the expression values of the k nearest neighbors of x,y:
def get_knn_expression(distances, neighbor_indices, genes, gene_labels, bandwidth=2.5):
    weights = np.exp(-distances / bandwidth)
    local_expression = pd.DataFrame(
        index=genes, columns=np.arange(distances.shape[0])
    ).astype(float)

    for i, gene in enumerate(genes):
        weights_ = weights.copy()
        weights_[(gene_labels[neighbor_indices]) != i] = 0
        local_expression.loc[gene, :] = weights_.sum(1)

    return local_expression


def create_histogram(
    df, genes=None, min_expression=0, KDE_bandwidth=None, x_max=None, y_max=None
):
    """
    Creates a 2d histogram of the data frame's [x,y] coordinates.
    Parameters
    ----------
    df : pd.DataFrame
        A dataframe of coordinates.
    genes : list, optional
        A list of genes to include in the histogram. The default is None.
    min_expression : int, optional
        The minimum expression level to include in the histogram. The default is 5.
    KDE_bandwidth : int, optional
        The bandwidth of the gaussian blur applied to the histogram. The default is 1.
    grid_size : int, optional
        The size of the grid. The default is 1.

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

    hist, xedges, yedges = np.histogram2d(
        df["x_pixel"], df["y_pixel"], bins=[np.arange(x_max + 2), np.arange(y_max + 2)]
    )

    if KDE_bandwidth is not None:
        hist = gaussian_filter(hist, sigma=KDE_bandwidth)

    hist[hist < min_expression] = 0

    return hist


def compute_divergence_embedded(
    df,
    genes,
    visualizer,
    KDE_bandwidth,
    min_expression,
    metric="cosine_similarity",
    pca_divergence=0.8,
):
    signal = create_histogram(
        df,
        genes,
        x_max=df.x_pixel.max(),
        y_max=df.y_pixel.max(),
        KDE_bandwidth=KDE_bandwidth,
    )

    genes = visualizer.genes

    pca = PCA(pca_divergence).fit(visualizer.localmax_celltyping_samples.T)

    mask = signal > min_expression

    df_top = df[df.z_delim < df.z]
    df_bot = df[df.z_delim > df.z]

    # dr_bottom = np.zeros((df_bottom.shape[0],df_bottom.shape[1], pca.components_.shape[0]))
    # dr_top = np.zeros((df_bottom.shape[0],df_bottom.shape[1], pca.components_.shape[0]))

    hists_top = np.zeros((mask.sum(), pca.components_.shape[0]))
    hists_bot = np.zeros((mask.sum(), pca.components_.shape[0]))

    for i, g in tqdm.tqdm(enumerate(visualizer.genes)):
        if np.abs(pca.components_.std(0)[i]) < 0.001:
            continue

        hist_top = np.dot(
            create_histogram(
                df_top,
                genes=[g],
                x_max=df.x_pixel.max(),
                y_max=df.y_pixel.max(),
                KDE_bandwidth=1.0,
            )[mask][:, None],
            pca.components_[None, :, i],
        )
        hist_bot = np.dot(
            create_histogram(
                df_bot,
                genes=[g],
                x_max=df.x_pixel.max(),
                y_max=df.y_pixel.max(),
                KDE_bandwidth=1.0,
            )[mask][:, None],
            pca.components_[None, :, i],
        )

        hists_top += hist_top
        hists_bot += hist_bot

    if metric == "cosine_similarity":
        # Cosine similarity:
        hists_top_norm = hists_top / np.linalg.norm(hists_top, axis=1)[:, None]
        hists_bot_norm = hists_bot / np.linalg.norm(hists_bot, axis=1)[:, None]

        cos_sim = (hists_top_norm * hists_bot_norm).sum(axis=1)

        distance_ = np.zeros_like(signal)
        distance_[mask] = cos_sim

        return distance_, signal

    elif metric == "kl_divergence":
        # KL divergence
        # D_KL(P||Q) = sum_i P(i) log(P(i)/Q(i))

        hists_top_p = hists_top[:]  # / np.linalg.norm(hists_top, axis=1)[:, None]
        hists_bot_p = hists_bot[:]  # / np.linalg.norm(hists_bot, axis=1)[:, None]

        hists_top_p[hists_top_p > 0] = 0
        hists_bot_p[hists_bot_p > 0] = 0

        hists_top_p = np.nan_to_num(hists_top_p / hists_top_p.sum(1)[:, None])
        hists_bot_p = np.nan_to_num(hists_bot_p / hists_bot_p.sum(1)[:, None])

        kl_divergence = np.zeros((hists_top_p.shape[0],))
        for i in range(hists_top_p.shape[0]):
            kl_divergence[i] = (
                hists_top_p[i] * np.nan_to_num(np.log(hists_top_p[i] / hists_bot_p[i]))
            ).sum() + (
                hists_bot_p[i] * np.nan_to_num(np.log(hists_bot_p[i] / hists_top_p[i]))
            ).sum()

        distance_ = np.zeros_like(signal)
        distance_[mask] = kl_divergence

        return distance_, signal

    elif metric == "euclidean_distance":
        distance_ = np.zeros_like(signal)
        distance_[mask] = np.linalg.norm(hists_top - hists_bot, axis=1)

        return distance_, signal

    elif metric == "correlation":

        def pearson_cross_correlation(a, b):
            out = np.zeros((a.shape[0]))

            a = a - a.mean(axis=1)[:, None]
            b = b - b.mean(axis=1)[:, None]

            for i in range(a.shape[0]):
                out[i] = np.dot(a[i], b[i]) / (
                    np.linalg.norm(a[i]) * np.linalg.norm(b[i])
                )

            return out

        distance_ = np.zeros_like(signal)
        distance_[mask] = pearson_cross_correlation(hists_top, hists_bot)

        return distance_, signal


def compute_embedding_vectors(subset_df, signal_mask, factor):
    # for i,g in tqdm.tqdm(enumerate(genes),total=len(genes)):

    if len(subset_df) < 2:
        return None, None

    subset_top = subset_df[subset_df[:, 2] > subset_df[:, 3]]
    subset_bottom = subset_df[subset_df[:, 2] < subset_df[:, 3]]

    if len(subset_top) == 0:
        signal_top = 0
    else:
        signal_top = ssam_utils.kde_2d(subset_top[:, :2], size=signal_mask.shape)[
            signal_mask
        ]
        signal_top = signal_top[:, None] * factor[None]
    if len(subset_bottom) == 0:
        signal_bottom = 0
    else:
        signal_bottom = ssam_utils.kde_2d(subset_bottom[:, :2], size=signal_mask.shape)[
            signal_mask
        ]
        signal_bottom = signal_bottom[:, None] * factor[None]

    return signal_top, signal_bottom


def compute_divergence_patched(
    df,
    gene_list,
    pca_component_matrix,
    min_expression=2,
    KDE_bandwidth=1,
    patch_length=1000,
    patch_padding=10,
    n_workers=5,
):
    n_components = pca_component_matrix.shape[0]

    signal = ssam_utils.kde_2d(df[["x", "y"]].values)

    cosine_similarity = np.zeros_like(signal)

    patch_count_x = signal.shape[0] // patch_length
    patch_count_y = signal.shape[1] // patch_length

    x_patches = np.linspace(0, signal.shape[0], patch_count_x + 1).astype(int)
    y_patches = np.linspace(0, signal.shape[1], patch_count_y + 1).astype(int)

    print((x_patches), (y_patches))

    with tqdm.tqdm(total=(len(x_patches) - 1) * (len(y_patches) - 1)) as pbar:
        for i in range(len(x_patches) - 1):
            for j in range(len(y_patches) - 1):
                x_ = x_patches[i] - patch_padding
                y_ = y_patches[j] - patch_padding
                _x = x_patches[i + 1] + patch_padding
                _y = y_patches[j + 1] + patch_padding

                patch_size_x = _x - x_ - patch_padding * 2
                patch_size_y = _y - y_ - patch_padding * 2

                # print(x_,y_,_x,_y)

                patch_df = df[
                    (df.x >= x_) & (df.x < _x) & (df.y >= y_) & (df.y < _y)
                ].copy()

                patch_df.x -= x_
                patch_df.y -= y_

                if len(patch_df) == 0:
                    pbar.update(1)
                    continue

                patch_signal = ssam_utils.kde_2d(patch_df[["x", "y"]].values)

                patch_signal_mask = patch_signal > min_expression

                if patch_signal_mask.sum() == 0:
                    pbar.update(1)
                    continue

                # plt.figure()
                # plt.imshow(patch_signal)

                patch_embedding_top = np.zeros((patch_signal_mask.sum(), n_components))
                patch_embedding_bottom = np.zeros(
                    (patch_signal_mask.sum(), n_components)
                )

                df_gene_grouped = patch_df.groupby("gene", observed=False).apply(
                    lambda x: x[["x", "y", "z", "z_delim"]].values
                )
                # df_gene_grouped = df_gene_grouped[df_gene_grouped.apply(lambda x: x.shape[0]>0)]

                with ThreadPoolExecutor(max_workers=n_workers) as executor:
                    # compute_embedding_vectors(df_gene_grouped.loc[gene_list[0]],patch_signal_mask,pca_component_matrix[:,0])
                    # return cosine_similarity, signal
                    fs = {
                        executor.submit(
                            compute_embedding_vectors,
                            df_gene_grouped.loc[gene],
                            patch_signal_mask,
                            pca_component_matrix[:, i],
                        ): gene
                        for i, gene in enumerate(gene_list)
                    }

                    for f in as_completed(fs):
                        # try:
                        top_, bottom_ = f.result()
                        # print(top_.shape)
                        if top_ is not None:
                            patch_embedding_top += top_
                            patch_embedding_bottom += bottom_

                        # except Exception as exc:
                        # print('%r generated an exception: %s' % (gene, exc))

                patch_norm_top = np.linalg.norm(patch_embedding_top, axis=1)
                patch_norm_bottom = np.linalg.norm(patch_embedding_bottom, axis=1)

                patch_cosine_similarity = np.sum(
                    patch_embedding_top * patch_embedding_bottom, axis=1
                ) / (patch_norm_top * patch_norm_bottom)
                spatial_patch_cosine_similarity = np.zeros_like(patch_signal)

                spatial_patch_cosine_similarity[patch_signal_mask] = (
                    patch_cosine_similarity
                )
                spatial_patch_cosine_similarity = spatial_patch_cosine_similarity[
                    patch_padding : patch_padding + patch_size_x,
                    patch_padding : patch_padding + patch_size_y,
                ]  # remove patch edges

                x_pad = x_ + patch_padding
                y_pad = y_ + patch_padding
                cosine_similarity[
                    x_pad : x_pad + spatial_patch_cosine_similarity.shape[0],
                    y_pad : y_pad + spatial_patch_cosine_similarity.shape[1],
                ] = spatial_patch_cosine_similarity

                pbar.update(1)

    return cosine_similarity, signal
