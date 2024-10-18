import warnings

import anndata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA

from ._ssam2 import sample_expression
from ._utils import (
    _compute_divergence_patched,
    _create_histogram,
    _create_knn_graph,
    _determine_localmax,
    _fill_color_axes,
    _get_kl_divergence,
    _get_knn_expression,
    _get_spatial_subsample_mask,
    _min_to_max,
    _plot_embeddings,
    _transform_embeddings,
)

# This is a package to detect overlapping cells in a 2d spatial transcriptomics sample.


def assign_xy(df, xy_columns=["x", "y"], grid_size=1):
    """
    Assigns an x,y coordinate to a pd.DataFrame of coordinates.
    Parameters
    ----------
    df : pd.DataFrame
        A dataframe of coordinates.
    xyz_columns : list, optional
        The names of the columns containing the x,y,z coordinates. The default is ['x','y', 'z'].
    grid_size : int, optional
        The size of the grid. The default is 1.

    Returns
    -------
    df : pd.DataFrame
        A dataframe with an x,y coordinate assigned to each row.

    """
    df["x_pixel"] = (df[xy_columns[0]] / grid_size).astype(int)
    df["y_pixel"] = (df[xy_columns[1]] / grid_size).astype(int)

    # assign each pixel a unique id
    df["n_pixel"] = df["x_pixel"] + df["y_pixel"] * df["x_pixel"].max()

    return df


def assign_z_median(df, z_column="z"):
    """
    Assigns a z coordinate to a pd.DataFrame of coordinates.
    Parameters
    ----------
    df : pd.DataFrame
        A dataframe of coordinates.
    z_column : str, optional
        The name of the column containing the z coordinate. The default is 'z'.

    Returns
    -------
    df : pd.DataFrame
        A dataframe with a z coordinate assigned to each row.

    """
    if "n_pixel" not in df.columns:
        print(
            "Please assign x,y coordinates to the dataframe first by running assign_xy(df)"
        )
    medians = df.groupby("n_pixel")[z_column].median()
    df["z_delim"] = medians[df.n_pixel].values

    return medians


def assign_z_mean_message_passing(df, z_column="z", delim_column="z_delim", rounds=3):
    """
    Assigns a z coordinate to a pd.DataFrame of coordinates.
    Parameters
    ----------
    df : pd.DataFrame
        A dataframe of coordinates.
    z_column : str, optional
        The name of the column containing the z coordinate. The default is 'z'.

    Returns
    -------
    df : pd.DataFrame
        A dataframe with a z coordinate assigned to each row.

    """
    if "n_pixel" not in df.columns:
        print(
            "Please assign x,y coordinates to the dataframe first by running assign_xy(df)"
        )

    means = df.groupby("n_pixel")[z_column].mean()
    df[delim_column] = means[df.n_pixel].values

    pixel_coordinate_df = (
        df[["n_pixel", "x_pixel", "y_pixel", delim_column]].groupby("n_pixel").max()
    )
    elevation_map = np.zeros((df.x_pixel.max() + 1, df.y_pixel.max() + 1))
    elevation_map.fill(np.nan)

    elevation_map[pixel_coordinate_df.x_pixel, pixel_coordinate_df.y_pixel] = (
        pixel_coordinate_df[delim_column]
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for r in range(rounds):
            elevation_map_ = (
                np.nanmean([elevation_map, np.roll(elevation_map, 1, axis=0)], axis=0)
                / 4
            )
            elevation_map_ += (
                np.nanmean([elevation_map, np.roll(elevation_map, 1, axis=1)], axis=0)
                / 4
            )
            elevation_map_ += (
                np.nanmean([elevation_map, np.roll(elevation_map, -1, axis=0)], axis=0)
                / 4
            )
            elevation_map_ += (
                np.nanmean([elevation_map, np.roll(elevation_map, -1, axis=1)], axis=0)
                / 4
            )

            elevation_map = elevation_map_

    df[delim_column] = elevation_map[df.x_pixel, df.y_pixel]

    return df[delim_column]


def assign_z_mean(df, z_column="z"):
    """
    Assigns a z coordinate to a pd.DataFrame of coordinates.
    Parameters
    ----------
    df : pd.DataFrame
        A dataframe of coordinates.
    z_column : str, optional
        The name of the column containing the z coordinate. The default is 'z'.

    Returns
    -------
    df : pd.DataFrame
        A dataframe with a z coordinate assigned to each row.

    """
    if "n_pixel" not in df.columns:
        print(
            "Please assign x,y coordinates to the dataframe first by running assign_xy(df)"
        )
    means = df.groupby("n_pixel")[z_column].mean()
    df["z_delim"] = means[df.n_pixel].values

    return means


def get_rois(df, genes=None, min_distance=10, KDE_bandwidth=1, min_expression=5):
    """
    Returns a list of local maxima in a kde of the data frame.
    Parameters
    ----------
    df : pd.DataFrame
        A dataframe of coordinates.
    min_distance : int
        The minimum distance between local maxima.

    Returns
    -------
    rois : list
        A list of local maxima in a kde of the data frame.

    """

    if genes is None:
        genes = sorted(df.gene.unique())

    hist = _create_histogram(
        df, genes=genes, min_expression=min_expression, KDE_bandwidth=KDE_bandwidth
    )

    rois_x, rois_y, _ = _determine_localmax(
        hist, min_distance=min_distance, min_expression=min_expression
    )

    return rois_x, rois_y


def get_expression_vectors_at_rois(
    df, rois_x, rois_y, genes=None, KDE_bandwidth=1, min_expression=0
):
    """
    Returns a matrix of gene expression vectors at each local maximum.
    Parameters
    ----------
    df : pd.DataFrame
        A dataframe of coordinates.
    kde_plot_window_size : int
    Returns
    -------
    """

    if genes is None:
        genes = sorted(df.gene.unique())

    rois_n_pixel = rois_x + rois_y * df.x_pixel.max()

    expressions = pd.DataFrame(index=genes, columns=rois_n_pixel, dtype=float)
    expressions[:] = 0

    for gene in genes:
        hist = _create_histogram(
            df, genes=[gene], min_expression=min_expression, KDE_bandwidth=KDE_bandwidth
        )

        expressions.loc[gene] = hist[rois_x, rois_y]

    return expressions


def compute_divergence(
    df,
    genes,
    KDE_bandwidth=1,
    threshold_fraction=0.5,
    min_distance=3,
    min_expression=5,
    density_weight=2,
    plot=False,
    return_maps=False,
    divergence_spatial_blur=2,
):
    """
    Computes the divergence between the top and bottom of the tissue sample.
    Parameters
    ----------
    df : pd.DataFrame
        A dataframe of coordinates.
    genes : list
        A list of genes to compute the divergence for.
    KDE_bandwidth : int
        The bandwidth of the KDE.
    threshold_fraction : float
        The fraction of the loss score's maximum, used as a cutoff value.
    min_distance : int
        The minimum distance between two retrieved regions of interest.
    plot : bool
        Whether to plot the KDE.
    Returns
    -------
    divergence : np.array
        A matrix of divergence values.
    """

    divergence, signal_histogram = compute_divergence_map(
        df, genes, KDE_bandwidth, min_expression
    )

    distance_map = divergence * signal_histogram**density_weight
    # gaussian filter on distance score:
    distance_map = gaussian_filter(distance_map, sigma=divergence_spatial_blur)
    distance_threshold = distance_map.max() * threshold_fraction

    rois_x, rois_y, distance_score = _determine_localmax(
        distance_map, min_distance, distance_threshold
    )

    if plot:
        plt.imshow(signal_histogram, cmap="Greens")
        alpha = np.nan_to_num(divergence)
        alpha = alpha - alpha.min()
        alpha = alpha / alpha.max()

        plt.imshow(divergence, cmap="Reds", alpha=alpha**0.5)
        # plt.scatter(rois_y, rois_x, c='b', marker='x')

    if return_maps:
        return (
            rois_x,
            rois_y,
            distance_score,
            distance_map,
            signal_histogram,
            divergence,
        )

    return rois_x, rois_y, distance_score


def compute_divergence_map(
    df,
    genes,
    KDE_bandwidth,
    min_expression,
):
    """
    Computes the divergence map between the top and bottom of the tissue sample.
    Parameters
    ----------
    df : pd.DataFrame
        A dataframe of coordinates.
    genes : list
        A list of genes to compute the divergence for.
    KDE_bandwidth : int
        The bandwidth of the KDE.
    Returns
    -------
    divergence : np.array
        A pixel map of divergence values.
    signal_histogram : np.array
        A pixel map of signal magnitude.
    """

    signal_histogram = _create_histogram(
        df, genes=genes, min_expression=min_expression, KDE_bandwidth=KDE_bandwidth
    )

    divergence = np.zeros_like(signal_histogram)

    df_top = df[df.z_delim < df.z]
    df_bottom = df[df.z_delim > df.z]

    x_max = df.x_pixel.max()
    y_max = df.y_pixel.max()

    for gene in genes:
        hist_top = _create_histogram(
            df_top,
            genes=[gene],
            min_expression=0,
            KDE_bandwidth=KDE_bandwidth,
            x_max=x_max,
            y_max=y_max,
        )
        hist_bottom = _create_histogram(
            df_bottom,
            genes=[gene],
            min_expression=0,
            KDE_bandwidth=KDE_bandwidth,
            x_max=x_max,
            y_max=y_max,
        )

        mask = (hist_top > 0) & (hist_bottom > 0) & (signal_histogram > 0)
        hist_top[mask] /= signal_histogram[mask]
        hist_bottom[mask] /= signal_histogram[mask]

        divergence[mask] += _get_kl_divergence(hist_top[mask], hist_bottom[mask])
        divergence[mask] += _get_kl_divergence(hist_bottom[mask], hist_top[mask])

    return divergence, signal_histogram


def find_overlaps(
    coordinate_df=None,
    adata=None,
    coordinates_key="spatial",
    genes_key="gene",
    genes=None,
    KDE_bandwidth=1.0,
    threshold_fraction=0.5,
    min_distance=10,
    min_expression=5,
    density_weight=2,
    return_maps=False,
):
    """
    Finds regions of overlap between the top and bottom of the tissue sample.
    Parameters
    ----------
    coordinate_df : pd.DataFrame
        A dataframe of coordinates.
    adata : anndata.AnnData
        An AnnData object containing the coordinates.
    coordinates_key : str
        The key in the AnnData object's uns attribute containing the coordinates.
    genes_key : str
        The key in the AnnData object's uns attribute containing the genes.
    genes : list
        A list of genes to compute the divergence for.
    KDE_bandwidth : float
        The bandwidth of the KDE.
    threshold_fraction : float
        The fraction of the divergence score's maximum, used as a cutoff value.

    """

    if (coordinate_df is None) and (adata is None):
        raise ValueError("Either adata or coordinate_df must be provided.")

    if coordinate_df is None:
        coordinate_df = adata.uns[coordinates_key]

    if genes is None:
        genes = sorted(coordinate_df[genes_key].unique())

    assign_xy(coordinate_df)
    assign_z_mean(coordinate_df)

    if return_maps:
        (
            rois_x,
            rois_y,
            distance_score,
            distance_map,
            signal_histogram,
            divergence_map,
        ) = compute_divergence(
            coordinate_df,
            genes,
            KDE_bandwidth=KDE_bandwidth,
            threshold_fraction=threshold_fraction,
            min_distance=min_distance,
            min_expression=min_expression,
            density_weight=density_weight,
            return_maps=return_maps,
        )

    else:
        rois_x, rois_y, distance_score = compute_divergence(
            coordinate_df,
            genes,
            KDE_bandwidth=KDE_bandwidth,
            threshold_fraction=threshold_fraction,
            min_distance=min_distance,
            min_expression=min_expression,
            density_weight=density_weight,
        )

    roi_df = pd.DataFrame({"x": rois_x, "y": rois_y, "divergence": distance_score})
    roi_df = roi_df.sort_values("divergence", ascending=False)

    if adata is not None:
        adata.uns["rois"] = roi_df
        return_val = adata.uns["rois"]
    else:
        return_val = roi_df

    if return_maps:
        return return_val, distance_map, signal_histogram, divergence_map
    else:
        return return_val


def plot_signal_integrity(
    integrity, signal, signal_threshold=2.0, cmap="BIH", plot_hist=True
):
    """
    Plots the determined signal integrity of the tissue sample in a signal integrity map.

    Args:
        integrity: Integrity map obtained from ovrlpy analysis
        signal: Signal map from ovrlpy analysis
        signal_threshold: Threshold below which the signal is faded out in the plot, to avoid displaying noisy areas with low predictive confidence.
        cmap: Colormap for display. Defaults to 'BIH'.
        plot_hist Whether to plot a histogram of integrity values alongside the map: Defaults to False.
    """

    figure_height = int(15 * integrity.shape[0] / integrity.shape[1]) + 1
    print(figure_height)

    with plt.style.context("dark_background"):
        if cmap == "BIH":
            cmap = LinearSegmentedColormap.from_list(
                "BIH",
                [
                    "#430541",
                    "mediumvioletred",
                    "violet",
                    "powderblue",
                    "powderblue",
                    "white",
                    "white",
                ][::-1],
            )

        if plot_hist:
            fig, ax = plt.subplots(
                1, 2, figsize=(20, figure_height), gridspec_kw={"width_ratios": [6, 1]}
            )
        else:
            fig, ax = plt.subplots(1, 1, figsize=(20, figure_height))
            ax = [ax]

        img = ax[0].imshow(
            integrity,
            cmap=cmap,
            alpha=((signal / signal_threshold).clip(0, 1) ** 2),
            vmin=0,
            vmax=1,
        )
        ax[0].invert_yaxis()
        ax[0].spines[["top", "right"]].set_visible(False)

        if plot_hist:
            vals, bins = np.histogram(
                integrity[signal > signal_threshold],
                bins=50,
                range=(0, 1),
                density=True,
            )
            colors = cmap(bins[1:-1])
            bars = ax[1].barh(bins[1:-1], vals[1:], height=0.01)
            for i, bar in enumerate(bars):
                bar.set_color(colors[i])
            ax[1].set_ylim(0, 1)
            ax[1].set_xticks([], [])
            ax[1].invert_xaxis()
            ax[1].yaxis.tick_right()
            ax[1].spines[["top", "bottom", "left"]].set_visible(False)
            ax[1].set_ylabel("signal integrity")

        else:
            fig.colorbar(img)

    return fig, ax


def detect_doublets(
    coherence,
    signal,
    min_distance=10,
    max_incoherence=0.4,
    signal_cutoff=3,
    coherence_sigma=None,
):
    """
    This function is used to find individual peaks of signal incoherence in the tissue map as an indicator of single occurrences overlapping cells.
    Args:
        coherence: Coherence map obtained from ovrlpy analysis
        signal: Signal map from ovrlpy analysis
        min_distance: Minimum distance between reported peaks
        max_incoherence: Maximum incoherence value for a peak to be considered
        signal_cutoff: Minimum signal value for a peak to be considered
        coherence_sigma: Optional sigma value for gaussian filtering of the coherence map, which leads to the detection of overlap regions with larger spatial extent.
    """

    # if signal_cutoff is None:
    #     signal_cutoff = signal.max()*0.6

    if coherence_sigma is not None:
        coherence = gaussian_filter(coherence, coherence_sigma)

    dist_x, dist_y, dist_t = _determine_localmax(
        (1 - coherence) * (signal > signal_cutoff),
        min_distance=min_distance,
        min_expression=max_incoherence,
    )

    arg_idcs = np.argsort(dist_t)[::-1]
    dist_x, dist_y = dist_x[arg_idcs], dist_y[arg_idcs]

    doublet_df = pd.DataFrame(
        {
            "x": dist_y,
            "y": dist_x,
            "integrity": dist_t,
            "signal": signal[dist_x, dist_y],
        }
    )

    return doublet_df


def determine_celltype_class_assignments(expression_samples, signature_matrix):
    expression_samples_ = expression_samples.copy().loc[signature_matrix.index]
    correlations = np.array(
        [
            np.corrcoef(expression_samples_.iloc[:, i], signature_matrix.values.T)[
                0, 1:
            ]
            for i in range(expression_samples.shape[1])
        ]
    )
    return np.argmax(correlations, -1)


class Visualizer:
    """
    A class to visualize spatial transcriptomics data.
    Contains a latent gene expression UMAP and RGB embedding.
    Parameters
    ----------
    KDE_bandwidth : float, optional
        The bandwidth of the KDE. Default is 1.5.
    celltyping_min_expression : int, optional
        Minimum expression level for cell typing. Default is 10.
    celltyping_min_distance : int, optional
        Minimum distance for cell typing. Default is 5.
    n_components_pca : float, optional
        Number of components for PCA. Default is 0.7.
    umap_kwargs : dict, optional
        Keyword arguments for 2D UMAP embedding. Default is {"n_components": 2, "min_dist": 0.0, "n_neighbors": 20, "random_state": None}.
    cumap_kwargs : dict, optional
        Keyword arguments for 3D UMAP embedding. Default is {"n_components": 3, "min_dist": 0.001, "n_neighbors": 50, "random_state": None}.
    Methods
    -------
    fit_ssam(coordinate_df=None, adata=None, genes=None, gene_key="gene", coordinates_key="spatial", signature_matrix=None, n_workers=32)
        Fits the visualizer to a spatial transcriptomics dataset using the SSAM algorithm.
    fit_signatures(signature_matrix=None)
        Fits the visualizer with a given signature matrix.
    subsample_df(x, y, coordinate_df=None, window_size=30)
        Subsamples the coordinate dataframe based on given x, y coordinates and window size.
    transform(coordinate_df)
        Transforms the coordinate dataframe to the embedding space.
    roi_df()
        Returns a pandas DataFrame containing the gene-count matrix of the fitted tissue's determined pseudo-cells.
    linear_transform(x, y, coordinate_df=None, window_size=30)
        Performs a linear transformation on the coordinate dataframe based on given x, y coordinates and window size.
    plot_instance(subsample, subsample_embedding, subsample_embedding_color, x, y, window_size=30, rasterized=True)
        Plots an instance of the visualized data.
    plot_umap(display_text=True, ax=None, rasterized=False, **kwargs)
        Plots the UMAP embedding.
    plot_tissue(rasterized=False, **kwargs)
        Plots the tissue embedding.
    plot_fit(rasterized=True)
        Plots the fitted model.
    save(path)
        Stores the visualizer and its attributes in an h5ad file.
    """

    def __init__(
        self,
        KDE_bandwidth=1.5,
        celltyping_min_expression=10,
        celltyping_min_distance=5,
        n_components_pca=0.7,
        umap_kwargs={
            "n_components": 2,
            "min_dist": 0.0,
            "n_neighbors": 20,
            "random_state": None,
        },
        cumap_kwargs={
            "n_components": 3,
            "min_dist": 0.001,
            "n_neighbors": 50,
            "random_state": None,
        },
    ) -> None:
        """ """
        self.KDE_bandwidth = KDE_bandwidth

        self.celltyping_min_expression = celltyping_min_expression
        self.celltyping_min_distance = celltyping_min_distance
        self.rois_celltyping_x, self.rois_celltyping_y = None, None
        self.localmax_celltyping_samples = None
        self.signatures = None
        self.celltype_centers = None
        self.celltype_class_assignments = None

        self.pca_2d = None
        self.embedder_2d = None
        self.pca_3d = None
        self.embedder_3d = None
        self.n_components_pca = n_components_pca
        self.umap_kwargs = umap_kwargs
        self.cumap_kwargs = cumap_kwargs

        self.cumap_kwargs["n_components"] = 3

        self.genes = None
        self.embedding = None
        self.colors = None
        self.colors_min_max = [None, None]

        self.coherence_map = None
        self.signal_map = None

    # def fit(
    #     self,
    #     coordinate_df=None,
    #     adata=None,
    #     genes=None,
    #     gene_key="gene",
    #     coordinates_key="spatial",
    #     signature_matrix=None,
    # ):
    #     """ """

    #     if (coordinate_df is None) and (adata is None):
    #         raise ValueError("Either adata or coordinate_df must be provided.")

    #     if coordinate_df is None:
    #         coordinate_df = adata.uns[coordinates_key]

    #     if genes is None:
    #         genes = sorted(coordinate_df[gene_key].unique())

    #     self.genes = genes

    #     if signature_matrix is None:
    #         signature_matrix = pd.DataFrame(
    #             np.eye(len(genes)), index=genes, columns=genes
    #         ).astype(float)
    #         signature_matrix[:] = np.eye(len(genes))

    #     self.signatures = signature_matrix

    #     self.rois_celltyping_x, self.rois_celltyping_y = get_rois(
    #         coordinate_df,
    #         genes=genes,
    #         min_distance=self.celltyping_min_distance,
    #         KDE_bandwidth=self.KDE_bandwidth,
    #         min_expression=self.celltyping_min_expression,
    #     )

    #     self.localmax_celltyping_samples = get_expression_vectors_at_rois(
    #         coordinate_df, self.rois_celltyping_x, self.rois_celltyping_y, genes
    #     )

    #     self.localmax_celltyping_samples = (
    #         self.localmax_celltyping_samples
    #         / (self.localmax_celltyping_samples.to_numpy() ** 2).sum(0, keepdims=True)
    #         ** 0.5
    #     )

    #     self.pca_2d = PCA(
    #         n_components=min(
    #             self.n_components_pca, self.localmax_celltyping_samples.shape[0] // 2
    #         )
    #     )
    #     factors = self.pca_2d.fit_transform(self.localmax_celltyping_samples.T)
    #     print(f"Modeling {factors.shape[0]} pseudo-celltypes")

    #     # init=np.concatenate([self.embedding,0.01*np.random.normal(size=(self.embedding.shape[0],1))],axis=1))
    #     # embedding_color = embedder_3d.fit_transform(factors)

    #     self.embedder_2d = umap.UMAP(**self.umap_kwargs)
    #     self.embedding = self.embedder_2d.fit_transform(factors)

    #     self.embedder_3d = umap.UMAP(**self.cumap_kwargs)
    #     # metric=discounted_euclidean_grad, metric_kwds={'discount':0.5},
    #     # metric=discounted_euclidean_grad, output_metric_kwds={'discount':1.0})

    #     embedding_color = self.embedder_3d.fit_transform(
    #         factors
    #     )  # np.tile(self.embedding,[1,2]))

    #     embedding_color, self.pca_3d = __fill_color_axes(embedding_color)

    #     color_min = embedding_color.min(0)
    #     color_max = embedding_color.max(0)

    #     self.colors = _min_to_max(embedding_color.copy())
    #     self.colors_min_max = [color_min, color_max]

    #     self.fit_signatures(signature_matrix)

    #     gene_assignments = determine_celltype_class_assignments(
    #         self.localmax_celltyping_samples,
    #         pd.DataFrame(np.eye(len(genes)), index=genes, columns=genes).astype(float),
    #     )

    #     # # determine the center of gravity of each celltype in the embedding:
    #     self.gene_centers = np.array(
    #         [
    #             np.median(self.embedding[gene_assignments == i, :], axis=0)
    #             for i in range(len(self.genes))
    #         ]
    #     )

    def fit_ssam(
        self,
        coordinate_df=None,
        adata=None,
        genes=None,
        gene_key="gene",
        coordinates_key="spatial",
        signature_matrix=None,
        n_workers=32,
    ):
        """
        Fits the visualizer to a spatial transcripts dataset using the SSAM algorithm.
        Parameters
        ----------
        coordinate_df : pd.DataFrame
            A dataframe of coordinates.
        adata : anndata.AnnData
            An AnnData object containing the coordinates.
        genes : list
            A list of genes to utilize in the model. None uses all genes.
        gene_key : str
            The key in the dataframe containing the gene names.
        coordinates_key : str
            The key in the dataframe containing the coordinates.
        signature_matrix : pd.DataFrame
            A matrix of celltypes x gene signatures to use to annotate the UMAP. None defaults to displaying individual genes.
        n_workers : int
            The number of workers to use in the SSAM algorithm
        """

        if (coordinate_df is None) and (adata is None):
            raise ValueError("Either adata or coordinate_df must be provided.")

        if coordinate_df is None:
            coordinate_df = adata.uns[coordinates_key]

        if genes is None:
            genes = sorted(coordinate_df[gene_key].unique())

        self.genes = genes

        if signature_matrix is None:
            signature_matrix = pd.DataFrame(
                np.eye(len(genes)), index=genes, columns=genes
            ).astype(float)
            signature_matrix[:] = np.eye(len(genes))

        self.signatures = signature_matrix

        adata_ssam = sample_expression(
            coordinate_df,
            gene_column=gene_key,
            minimum_expression=self.celltyping_min_expression,
            kde_bandwidth=self.KDE_bandwidth,
            n_workers=n_workers,
            min_pixel_distance=self.celltyping_min_distance,
        )

        self.rois_celltyping_x, self.rois_celltyping_y, _ = adata_ssam.obsm["spatial"].T

        self.localmax_celltyping_samples = pd.DataFrame(
            adata_ssam.X.T, columns=adata_ssam.obs_names, index=adata_ssam.var_names
        )
        self.pca_2d = PCA(
            n_components=min(
                self.n_components_pca, self.localmax_celltyping_samples.shape[0] // 2
            )
        )
        factors = self.pca_2d.fit_transform(self.localmax_celltyping_samples.T)

        print(f"Modeling {factors.shape[0]} pseudo-celltypes")

        self.embedder_2d = umap.UMAP(**self.umap_kwargs)
        self.embedding = self.embedder_2d.fit_transform(factors)

        self.embedder_3d = umap.UMAP(**self.cumap_kwargs)

        embedding_color = self.embedder_3d.fit_transform(factors)

        embedding_color, self.pca_3d = _fill_color_axes(embedding_color)

        color_min = embedding_color.min(0)
        color_max = embedding_color.max(0)

        self.colors = _min_to_max(embedding_color.copy())
        self.colors_min_max = [color_min, color_max]

        self.fit_signatures(signature_matrix)

        gene_assignments = determine_celltype_class_assignments(
            self.localmax_celltyping_samples,
            pd.DataFrame(np.eye(len(genes)), index=genes, columns=genes).astype(float),
        )

        # # determine the center of gravity of each celltype in the embedding:
        self.gene_centers = np.array(
            [
                np.median(self.embedding[gene_assignments == i, :], axis=0)
                for i in range(len(self.genes))
            ]
        )

    def fit_signatures(self, signature_matrix=None):
        """ """

        if signature_matrix is None:
            signature_matrix = pd.DataFrame(
                np.eye(len(self.genes)), index=self.genes, columns=self.genes
            ).astype(float)
            signature_matrix[:] = np.eye(len(self.genes))

        self.signatures = signature_matrix
        celltypes = sorted(signature_matrix.columns)

        self.celltype_class_assignments = determine_celltype_class_assignments(
            self.localmax_celltyping_samples, signature_matrix
        )

        # determine the center of gravity of each celltype in the embedding:
        self.celltype_centers = np.array(
            [
                np.median(
                    self.embedding[self.celltype_class_assignments == i, :], axis=0
                )
                for i in range(len(celltypes))
            ]
        )

    def subsample_df(self, x, y, coordinate_df=None, window_size=30):
        """ """

        subsample_mask = _get_spatial_subsample_mask(
            coordinate_df, x, y, plot_window_size=window_size
        )
        subsample = coordinate_df[subsample_mask]

        return subsample

    def transform(self, coordinate_df):
        """ """

        genes = self.genes

        subsample = coordinate_df

        distances, neighbor_indices = _create_knn_graph(
            subsample[["x", "y", "z"]].values, k=90
        )
        local_expression = _get_knn_expression(
            distances,
            neighbor_indices,
            genes,
            subsample.gene.cat.codes.values,
            bandwidth=self.KDE_bandwidth,
        )
        local_expression = local_expression / ((local_expression**2).sum(0) ** 0.5)
        subsample_embedding, subsample_embedding_color = _transform_embeddings(
            local_expression.T.values,
            self.pca_2d,
            embedder_2d=self.embedder_2d,
            embedder_3d=self.embedder_3d,
            colors_min_max=self.colors_min_max,
        )
        subsample_embedding_color, _ = _fill_color_axes(
            subsample_embedding_color, self.pca_3d
        )
        color_min, color_max = self.colors_min_max
        subsample_embedding_color = (subsample_embedding_color - color_min) / (
            color_max - color_min
        )
        subsample_embedding_color = np.clip(subsample_embedding_color, 0, 1)

        return subsample_embedding, subsample_embedding_color

    def roi_df(self):
        """
        Returns a pandas data frame containing the gene-count matrix of the fitted tissue's determined pseudo-cells.
        """
        roi_df = pd.DataFrame(
            {"x": self.rois_celltyping_x, "y": self.rois_celltyping_y}
        )

        if self.signal_map is not None:
            roi_df["signal"] = self.signal_map[roi_df.x, roi_df.y]

        if self.coherence_map is not None:
            roi_df["coherence"] = self.coherence_map[roi_df.x, roi_df.y]

        return roi_df

    def linear_transform(self, x, y, coordinate_df=None, window_size=30):
        """ """

        genes = self.genes

        subsample_mask = _get_spatial_subsample_mask(
            coordinate_df, x, y, plot_window_size=window_size
        )
        subsample = coordinate_df[subsample_mask]

        distances, neighbor_indices = _create_knn_graph(
            subsample[["x", "y", "z"]].values, k=90
        )
        local_expression = _get_knn_expression(
            distances,
            neighbor_indices,
            genes,
            subsample.gene.cat.codes.values,
            bandwidth=self.KDE_bandwidth,
        )
        local_expression = local_expression / ((local_expression**2).sum(0) ** 0.5)

        return subsample, local_expression

    def plot_instance(
        self,
        subsample,
        subsample_embedding,
        subsample_embedding_color,
        x,
        y,
        window_size=30,
        rasterized=True,
    ):
        """ """
        vertical_indices = subsample.z.argsort()
        subsample = subsample.sort_values("z")
        subsample_embedding = subsample_embedding[vertical_indices]
        subsample_embedding_color = subsample_embedding_color[vertical_indices]

        fig = plt.figure(figsize=(22, 12))

        gs = fig.add_gridspec(2, 3)

        ax1 = fig.add_subplot(gs[0, 2], projection="3d", label="3d_map")
        ax1.scatter(
            subsample.x,
            subsample.y,
            subsample.z,
            c=subsample_embedding_color,
            marker=".",
            alpha=0.5,
            rasterized=rasterized,
        )
        ax1.set_zlim(
            np.median(subsample.z) - window_size, np.median(subsample.z) + window_size
        )
        ax1.set_title("ROI celltype map, 3D")

        ax2 = fig.add_subplot(gs[0, 0], label="umap")
        ax2.scatter(
            self.embedding[:, 0],
            self.embedding[:, 1],
            c=self.colors,
            alpha=0.5,
            marker=".",
            s=1,
            rasterized=rasterized,
        )

        ax2.set_axis_off()
        # __plot_embeddings(subsample_embedding,subsample_embedding_color,self.celltype_centers,celltypes,rasterized=rasterized)
        ax2.set_title("UMAP")

        ax = fig.add_subplot(gs[0, 1], label="celltype_map")
        self.plot_tissue(rasterized=rasterized, s=1)
        ax.set_yticks([], [])

        artist = plt.Rectangle(
            (x - window_size, y - window_size),
            2 * window_size,
            2 * window_size,
            fill=False,
            edgecolor="k",
            linewidth=2,
        )
        ax.add_artist(artist)

        ax.set_title("celltype map")

        ax3 = fig.add_subplot(gs[1, 0], label="top_map")
        # plt.imshow((divergence*hist_sum).T,cmap='Greys', alpha=0.3 )
        ax3.scatter(
            subsample[subsample.z > subsample.z_delim].x,
            subsample[subsample.z > subsample.z_delim].y,
            c=subsample_embedding_color[subsample.z > subsample.z_delim],
            marker=".",
            alpha=0.8,
            s=40,
            rasterized=rasterized,
        )
        ax3.set_xlim(x - window_size, x + window_size)
        ax3.set_ylim(y - window_size, y + window_size)
        ax3.scatter(x, y, c="k", marker="+", s=100, rasterized=rasterized)
        ax3.set_aspect("equal", adjustable="box")

        ax3.set_title("ROI celltype map ,top")

        ax3 = fig.add_subplot(gs[1, 1], label="bottom_map")

        subsample = subsample[::-1]
        subsample_embedding = subsample_embedding[::-1]
        subsample_embedding_color = subsample_embedding_color[::-1]
        # plt.imshow(hist_sum.T,cmap='Greys',alpha=0.3 )
        ax3.scatter(
            subsample[subsample.z < subsample.z_delim].x,
            subsample[subsample.z < subsample.z_delim].y,
            c=subsample_embedding_color[subsample.z < subsample.z_delim],
            marker=".",
            alpha=0.8,
            s=40,
            rasterized=rasterized,
        )
        ax3.set_xlim(x - window_size, x + window_size)
        ax3.set_ylim(y - window_size, y + window_size)
        ax3.scatter(x, y, c="k", marker="+", s=100, rasterized=rasterized)
        ax3.set_aspect("equal", adjustable="box")

        ax3.set_title("ROI celltype map, bottom")

        sub_gs = gs[1, 2].subgridspec(2, 1)

        ax5 = fig.add_subplot(sub_gs[0, 0], label="x_cut")
        halving_mask = (subsample.y < (y + 4)) & (subsample.y > (y - 4))

        ax5.scatter(
            subsample.x[halving_mask],
            subsample.z[halving_mask],
            c=subsample_embedding_color[halving_mask],
            s=10,
            alpha=0.5,
            rasterized=rasterized,
        )
        ax5.set_aspect("equal", adjustable="box")
        plt.title("ROI, vertical, x-cut")

        ax4 = fig.add_subplot(sub_gs[1, 0], label="y_cut")
        halving_mask = (subsample.x < (x + 4)) & (subsample.x > (x - 4))

        ax4.scatter(
            subsample.y[halving_mask],
            subsample.z[halving_mask],
            c=subsample_embedding_color[halving_mask],
            s=10,
            alpha=0.5,
            rasterized=rasterized,
        )
        ax4.set_aspect("equal", adjustable="box")
        plt.title("ROI, vertical, y-cut")

    def plot_umap(self, display_text=True, ax=None, rasterized=False, **kwargs):
        """ """
        _plot_embeddings(
            self.embedding,
            self.colors,
            self.celltype_centers,
            self.signatures.columns,
            rasterized,
            ax,
            **kwargs,
        )

    def plot_tissue(self, rasterized=False, **kwargs):
        """ """
        ax = plt.gca()
        ax.scatter(
            self.rois_celltyping_x,
            self.rois_celltyping_y,
            c=self.colors,
            marker=".",
            alpha=1,
            rasterized=rasterized,
            **kwargs,
        )
        ax.set_aspect("equal", adjustable="box")

    def plot_fit(self, rasterized=True):
        """ """

        plt.figure(figsize=(15, 7))

        plt.subplot(121)
        self.plot_umap(rasterized=rasterized, **{"scatter_kwargs": {"s": 1}})

        plt.subplot(122)
        self.plot_tissue(rasterized=rasterized, **{"s": 1})

    def save(self, path):
        """Stores the visualizer and its attributes in an h5ad file.
        Parameters
        ----------
        path : str
            The path to the file.
        """

        import base64
        import pickle

        adata = anndata.AnnData(
            X=self.localmax_celltyping_samples.T.values,
            obs=pd.DataFrame(index=range(self.localmax_celltyping_samples.shape[1])),
            var=pd.DataFrame(index=self.localmax_celltyping_samples.index),
        )
        adata.obs["localmax_id"] = self.localmax_celltyping_samples.columns

        adata.uns["celltype_centers"] = self.celltype_centers
        adata.uns["args"] = {
            "KDE_bandwidth": self.KDE_bandwidth,
            "celltyping_min_expression": self.celltyping_min_expression,
            "celltyping_min_distance": self.celltyping_min_distance,
        }

        adata.obsm["celltype_class_assignments"] = self.celltype_class_assignments

        adata.uns["signatures"] = self.signatures
        adata.uns["pca_2d"] = {
            "components_": self.pca_2d.components_,
            "mean_": self.pca_2d.mean_,
        }

        knn_search_index_2d_pickled = base64.b64encode(
            pickle.dumps(self.embedder_2d._knn_search_index)
        ).decode("ascii")
        knn_search_index_3d_pickled = base64.b64encode(
            pickle.dumps(self.embedder_3d._knn_search_index)
        ).decode("ascii")
        # print(knn_search_index_pickled[:100])

        adata.uns["embedder_2d"] = {
            "kwargs": self.umap_kwargs,
            "_raw_data": self.embedder_2d._raw_data,
            "_small_data": self.embedder_2d._small_data,
            "_input_hash": self.embedder_2d._input_hash,
            # '_knn_search_index':self.embedder_2d._knn_search_index,
            "_knn_search_index": knn_search_index_2d_pickled,
            "_disconnection_distance": self.embedder_2d._disconnection_distance,
            "_n_neighbors": self.embedder_2d._n_neighbors,
            "embedding_": self.embedder_2d.embedding_,
            "_a": self.embedder_2d._a,
            "_b": self.embedder_2d._b,
            "_initial_alpha": self.embedder_2d._initial_alpha,
        }
        adata.uns["pca_3d"] = {
            "components_": self.pca_3d.components_,
            "mean_": self.pca_3d.mean_,
        }

        adata.uns["embedder_3d"] = {
            "kwargs": self.cumap_kwargs,
            "_raw_data": self.embedder_3d._raw_data,
            "_small_data": self.embedder_3d._small_data,
            "_input_hash": self.embedder_3d._input_hash,
            "_knn_search_index": knn_search_index_3d_pickled,  # self.embedder_3d._knn_search_index,
            "_disconnection_distance": self.embedder_3d._disconnection_distance,
            "_n_neighbors": self.embedder_3d._n_neighbors,
            "embedding_": self.embedder_3d.embedding_,
            "_a": self.embedder_3d._a,
            "_b": self.embedder_3d._b,
            "_initial_alpha": self.embedder_3d._initial_alpha,
        }

        adata.uns["colors_min_max"] = self.colors_min_max
        adata.uns["colors"] = self.colors
        adata.uns["embedding"] = self.embedding
        adata.uns["coherence_map"] = self.coherence_map
        adata.uns["signal_map"] = self.signal_map
        adata.uns["rois_celltyping_x"] = self.rois_celltyping_x
        adata.uns["rois_celltyping_y"] = self.rois_celltyping_y

        adata.write_h5ad(path)


def load_visualizer(path):
    """
    Loads a visualizer from an h5ad file.
    """

    import base64
    import pickle

    adata = anndata.read_h5ad(path)

    vis = Visualizer(**adata.uns["args"])

    vis.localmax_celltyping_samples = pd.DataFrame(
        adata.X.T, columns=adata.obs["localmax_id"], index=adata.var.index
    )
    vis.celltype_centers = adata.uns["celltype_centers"]
    vis.celltype_class_assignments = adata.obsm["celltype_class_assignments"]
    vis.signatures = adata.uns["signatures"]
    vis.genes = vis.signatures.index

    vis.pca_2d = PCA(n_components=adata.uns["pca_2d"]["components_"].shape[0])
    vis.pca_2d.components_ = adata.uns["pca_2d"]["components_"]
    vis.pca_2d.mean_ = adata.uns["pca_2d"]["mean_"]

    vis.embedder_2d = umap.UMAP(**adata.uns["embedder_2d"]["kwargs"])
    vis.embedder_2d._raw_data = adata.uns["embedder_2d"]["_raw_data"]
    vis.embedder_2d._small_data = adata.uns["embedder_2d"]["_small_data"]
    vis.embedder_2d._input_hash = adata.uns["embedder_2d"]["_input_hash"]
    vis.embedder_2d._knn_search_index = pickle.loads(
        base64.b64decode(adata.uns["embedder_2d"]["_knn_search_index"])
    )
    vis.embedder_2d._disconnection_distance = adata.uns["embedder_2d"][
        "_disconnection_distance"
    ]
    vis.embedder_2d._n_neighbors = adata.uns["embedder_2d"]["_n_neighbors"]
    vis.embedder_2d.embedding_ = adata.uns["embedder_2d"]["embedding_"]
    vis.embedder_2d._a = adata.uns["embedder_2d"]["_a"]
    vis.embedder_2d._b = adata.uns["embedder_2d"]["_b"]
    vis.embedder_2d._initial_alpha = adata.uns["embedder_2d"]["_initial_alpha"]

    vis.pca_3d = PCA(n_components=adata.uns["pca_3d"]["components_"].shape[0])
    vis.pca_3d.components_ = adata.uns["pca_3d"]["components_"]
    vis.pca_3d.mean_ = adata.uns["pca_3d"]["mean_"]

    vis.embedder_3d = umap.UMAP(**adata.uns["embedder_3d"]["kwargs"])
    vis.embedder_3d._raw_data = adata.uns["embedder_3d"]["_raw_data"]
    vis.embedder_3d._small_data = adata.uns["embedder_3d"]["_small_data"]
    vis.embedder_3d._input_hash = adata.uns["embedder_3d"]["_input_hash"]
    vis.embedder_3d._knn_search_index = pickle.loads(
        base64.b64decode(adata.uns["embedder_3d"]["_knn_search_index"])
    )
    vis.embedder_3d._disconnection_distance = adata.uns["embedder_3d"][
        "_disconnection_distance"
    ]
    vis.embedder_3d._n_neighbors = adata.uns["embedder_3d"]["_n_neighbors"]
    vis.embedder_3d.embedding_ = adata.uns["embedder_3d"]["embedding_"]
    vis.embedder_3d._a = adata.uns["embedder_3d"]["_a"]
    vis.embedder_3d._b = adata.uns["embedder_3d"]["_b"]
    vis.embedder_3d._initial_alpha = adata.uns["embedder_3d"]["_initial_alpha"]

    vis.colors_min_max = adata.uns["colors_min_max"]
    vis.colors = adata.uns["colors"]
    vis.embedding = adata.uns["embedding"]
    if "coherence_map" in adata.uns.keys():
        vis.coherence_map = adata.uns["coherence_map"]
    else:
        vis.coherence_map = None
    if "signal_map" in adata.uns.keys():
        vis.signal_map = adata.uns["signal_map"]
    else:
        vis.signal_map = None
    vis.rois_celltyping_x = adata.uns["rois_celltyping_x"]
    vis.rois_celltyping_y = adata.uns["rois_celltyping_y"]

    return vis


def compute_coherence_map(
    df,
    n_expected_celltypes=None,
    cell_diameter=10,
    # KDE_bandwidth=2,
    min_expression=0.5,
    # min_distance=8,
    # n_components_pca=0.7,
    return_visualizer=True,
    signature_matrix=None,
    metric="cosine_similarity",
    umap_kwargs={
        "n_components": 2,
        "min_dist": 0.0,
        "n_neighbors": 20,
        "random_state": 42,
        "n_jobs": 1,
    },
    cumap_kwargs={
        "n_neighbors": 10,
        "min_dist": 0,
        "metric": "euclidean",
        "random_state": 42,
        "n_jobs": 1,
    },
):
    """
    This is a wrapper function that computes the coherence map for a given spatial transcriptomics dataset.
    It includes the entire ovrlpy pipeline, returning a coherence map and a signal strength map and produces a visualizer object that can be used to plot the results.
    Parameters
    ----------
    df : pandas.DataFrame
        The spatial transcriptomics dataset.
    KDE_bandwidth : float, optional
        The bandwidth of the kernel density estimator used to construct the gene expression densities, by default 2
    min_expression : int, optional
        A threshold value to discard areas with low expression, by default 5
        Can be interpreted as: The minimum number of transcripts to appear within a radius of 'KDE_bandwidth' for a region to be considered containing a cell.
    min_distance : int, optional
        The minimum distance between two cells, by default 10

    """

    KDE_bandwidth = cell_diameter / 4
    min_distance = cell_diameter * 0.7

    if n_expected_celltypes is None:
        n_expected_celltypes = 0.8

    print("Running vertical adjustment")
    assign_xy(df)
    assign_z_mean_message_passing(df, rounds=4)

    vis = Visualizer(
        KDE_bandwidth=KDE_bandwidth,
        celltyping_min_expression=min_expression,
        celltyping_min_distance=min_distance,
        n_components_pca=n_expected_celltypes,
        umap_kwargs=umap_kwargs,
        cumap_kwargs=cumap_kwargs,
    )

    vis.fit_ssam(df, signature_matrix=signature_matrix)

    coherence_, signal_ = _compute_divergence_patched(
        df,
        vis.genes,
        vis.pca_2d.components_,
        KDE_bandwidth=KDE_bandwidth,
        min_expression=1,
    )

    vis.coherence_map = coherence_.T
    vis.signal_map = signal_.T

    return coherence_.T, signal_.T, vis
