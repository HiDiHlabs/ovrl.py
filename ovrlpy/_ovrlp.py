"""This is a package to detect overlapping cells in a 2D spatial transcriptomics sample."""

import os
import warnings
from functools import reduce
from operator import add
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from scipy.linalg import norm
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA

from ._ssam2 import _sample_expression
from ._utils import (
    SCALEBAR_PARAMS,
    UMAP_2D_PARAMS,
    UMAP_RGB_PARAMS,
    _create_histogram,
    _create_knn_graph,
    _determine_localmax_and_sample,
    _fill_color_axes,
    _get_knn_expression,
    _get_spatial_subsample_mask,
    _min_to_max,
    _plot_embeddings,
    _plot_scalebar,
    _transform_embeddings,
    compute_VSI,
)

_BIH_CMAP = LinearSegmentedColormap.from_list(
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


def _assign_xy(
    df: pd.DataFrame, xy_columns: Sequence[str] = ["x", "y"], grid_size: int = 1
):
    """
    Assigns an x,y coordinate to a DataFrame of coordinates.

    Parameters
    ----------
    df : pandas.DataFrame
        A dataframe of coordinates.
    xy_columns : list, optional
        The names of the columns containing the x,y-coordinates.
    grid_size : int, optional
        The size of the grid.

    Returns
    -------
    pandas.DataFrame
        A dataframe with an x,y coordinate assigned to each row.

    """
    x, y = xy_columns
    df[x] -= df[x].min()
    df[y] -= df[y].min()

    df["x_pixel"] = (df[x] / grid_size).astype(int)
    df["y_pixel"] = (df[y] / grid_size).astype(int)

    # assign each pixel a unique id
    df["n_pixel"] = df["x_pixel"] + df["y_pixel"] * df["x_pixel"].max()

    return df


def _assign_z_median(df: pd.DataFrame, z_column: str = "z"):
    """
    Assigns a z-coordinate to a DataFrame of coordinates.

    Parameters
    ----------
    df : pandas.DataFrame
        A dataframe of coordinates.
    z_column : str, optional
        The name of the column containing the z-coordinate.

    Returns
    -------
    pandas.DataFrame
        A dataframe with a z-coordinate assigned to each row.

    """
    if "n_pixel" not in df.columns:
        ValueError(
            "Please assign x,y coordinates to the dataframe first by running assign_xy(df)"
        )
    medians = df.groupby("n_pixel")[z_column].median()
    df["z_delim"] = medians[df.n_pixel].values

    return medians


def _assign_z_mean_message_passing(
    df: pd.DataFrame,
    z_column: str = "z",
    delim_column: str = "z_delim",
    rounds: int = 3,
):
    """
    Assigns a z-coordinate to a DataFrame of coordinates.

    Parameters
    ----------
    df : pandas.DataFrame
        A dataframe of coordinates.
    z_column : str, optional
        The name of the column containing the z-coordinate.
    rounds : int, optional
        Number of rounds for the message passing algorithm.

    Returns
    -------
    pandas.DataFrame
        A dataframe with a z-coordinate assigned to each row.

    """
    if "n_pixel" not in df.columns:
        ValueError(
            "Please assign x,y coordinates to the dataframe first by running assign_xy(df)"
        )

    means = df.groupby("n_pixel")[z_column].mean()
    df[delim_column] = means[df.n_pixel].values

    pixel_coordinate_df = (
        df[["n_pixel", "x_pixel", "y_pixel", delim_column]].groupby("n_pixel").max()
    )
    elevation_map = np.full((df.x_pixel.max() + 1, df.y_pixel.max() + 1), np.nan)

    elevation_map[pixel_coordinate_df.x_pixel, pixel_coordinate_df.y_pixel] = (
        pixel_coordinate_df[delim_column]
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for _ in range(rounds):
            elevation_map = reduce(
                add,
                (
                    np.nanmean(
                        [elevation_map, np.roll(elevation_map, shift, axis=ax)], axis=0
                    )
                    for ax in (0, 1)
                    for shift in (1, -1)
                ),
            )
            elevation_map /= 4

    df[delim_column] = elevation_map[df.x_pixel, df.y_pixel]

    return df[delim_column]


def _assign_z_mean(df: pd.DataFrame, z_column: str = "z"):
    """
    Assigns a z-coordinate to a DataFrame of coordinates.

    Parameters
    ----------
    df : pandas.DataFrame
        A dataframe of coordinates.
    z_column : str, optional
        The name of the column containing the z-coordinate.

    Returns
    -------
    pandas.DataFrame
        A dataframe with a z-coordinate assigned to each row.

    """
    if "n_pixel" not in df.columns:
        ValueError(
            "Please assign x,y coordinates to the dataframe first by running assign_xy(df)"
        )
    means = df.groupby("n_pixel")[z_column].mean()
    df["z_delim"] = means[df.n_pixel].values

    return means


def pre_process_coordinates(
    coordinate_df: pd.DataFrame,
    grid_size: int = 1,
    coordinates: tuple[str, str, str] = ("x", "y", "z"),
    use_message_passing: bool = True,
    measure: str = "mean",
    inplace: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """Runs the pre-processing routine of the coordinate dataframe.

    It assigns x,y coordinate pixels to all molecules in the data frame and determines a z-coordinate-center for each pixel.

    Parameters
    ----------
    coordinate_df : pandas.DataFrame
        A dataframe of coordinates.
    grid_size : int, optional
        The size of the pixel grid. Defaults to 1.
    coordinates : tuple[str, str, str], optional
        Name of the coordinate columns.
    use_message_passing : bool, optional
        Whether to use message passing to determine the z-dimension delimiter. Defaults to True.
    measure : str, optional
        The measure to use to determine the z-dimension delimiter. Defaults to 'mean'. Other option is 'median'.
    inplace : bool, optional
        Whether to modify the input dataframe or return a copy. Defaults to True.

    Returns
    -------
    pandas.DataFrame:
        A dataframe with added x_pixel, y_pixel and z_delim columns.
    """
    *xy, z = coordinates

    if not inplace:
        coordinate_df = coordinate_df.copy()

    _assign_xy(coordinate_df, xy_columns=xy, grid_size=grid_size)

    if use_message_passing:
        _assign_z_mean_message_passing(coordinate_df, z_column=z, **kwargs)

    else:
        if measure == "mean":
            _assign_z_mean(coordinate_df, z_column=z)
        elif measure == "median":
            _assign_z_median(coordinate_df, z_column=z)
        else:
            raise ValueError("measure must be either 'mean' or 'median'")

    return coordinate_df


def get_pseudocell_locations(
    df: pd.DataFrame,
    genes=None,
    min_distance: int = 10,
    KDE_bandwidth: float = 1,
    min_expression: float = 5,
):
    """
    Returns a list of local maxima in a kde of the data frame,
    interpreted as locations with high likelyhood to contain a cell.

    Parameters
    ----------
    df : pandas.DataFrame
        A dataframe of coordinates.
    genes : list, optional
        A list of genes to include in the expression model.
        Defaults to all genes in the dataframe.
    min_distance : int, optional
        The minimum distance between local maxima.
    KDE_bandwidth : float, optional
        Bandwidth of the Gaussian smoothing function used during the KDE.
    min_expression : float, optional
        A minimal expression threshold to report as local maximum regions of interest.

    Returns
    -------
    pseudocell_locations : list
        A list of local maxima in a KDE of the data frame.

    """

    if genes is None:
        genes = sorted(df.gene.unique())

    hist = _create_histogram(
        df, genes=genes, min_expression=min_expression, KDE_bandwidth=KDE_bandwidth
    )

    x, y, _ = _determine_localmax_and_sample(
        hist, min_distance=min_distance, min_expression=min_expression
    )

    return x, y


def sample_expression_at_xy(
    df,
    sampling_points_x,
    sampling_points_y,
    genes=None,
    KDE_bandwidth=1,
    min_expression: float = 0,
) -> pd.DataFrame:
    """
    Returns a matrix of gene expression vectors at each local maximum.

    Parameters
    ----------
    df : pandas.DataFrame
        A dataframe of coordinates that will be sampled for gene expression.
    sampling_points_x : list
        x-coordinates to sample gene expression at.
    sampling_points_y : list
        y-coordinates to sample gene expression at.
    genes : list, optional
        list of genes to sample expression for.
        Defaults to all genes in the dataframe.
    KDE_bandwidth : float, optional
        Bandwidth of the Gaussian smoothing function used during the KDE.
    min_expression : float, optional
        A minimal expression threshold to report as local maximum regions of interest.

    Returns
    -------
    pandas.DataFrame
    """

    if genes is None:
        genes = sorted(df.gene.unique())

    rois_n_pixel = sampling_points_x + sampling_points_y * df.x_pixel.max()

    expressions = pd.DataFrame(index=genes, columns=rois_n_pixel, dtype=float)
    expressions[:] = 0

    for gene in genes:
        hist = _create_histogram(
            df, genes=[gene], min_expression=min_expression, KDE_bandwidth=KDE_bandwidth
        )

        expressions.loc[gene] = hist[sampling_points_x, sampling_points_y]

    return expressions


def plot_signal_integrity(
    integrity,
    signal,
    signal_threshold: float = 2.0,
    cmap="BIH",
    figure_height: float = 10,
    plot_hist: bool = True,
    scalebar: dict | None = SCALEBAR_PARAMS,
):
    """
    Plots the determined signal integrity of the tissue sample in a signal integrity map.

    Parameters
    ----------
    integrity : numpy.ndarray
        Integrity map obtained from ovrlpy analysis
    signal : numpy.ndarray
        Signal map from ovrlpy analysis
    signal_threshold : float, optional
        Threshold below which the signal is faded out in the plot,
        to avoid displaying noisy areas with low predictive confidence.
    cmap : str | matplotlib.colors.Colormap, optional
        Colormap for display.
    plot_hist : bool, optional
        Whether to plot a histogram of integrity values alongside the map.
    scalebar : dict[str, typing.Any] | None
        If `None` no scalebar will be plotted. Otherwise a dictionary with
        additional kwargs for ``matplotlib_scalebar.scalebar.ScaleBar``.
        By default :py:attr:`ovrlpy.SCALEBAR_PARAMS`
    """

    figure_aspect_ratio = integrity.shape[0] / integrity.shape[1]

    with plt.style.context("dark_background"):
        if cmap == "BIH":
            cmap = _BIH_CMAP

        if plot_hist:
            fig, ax = plt.subplots(
                1,
                2,
                figsize=(figure_height / figure_aspect_ratio * 1.4, figure_height),
                gridspec_kw={"width_ratios": [6, 1]},
            )
        else:
            fig, ax = plt.subplots(
                1, 1, figsize=(figure_height / figure_aspect_ratio, figure_height)
            )
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

        if scalebar is not None:
            _plot_scalebar(ax[0], **SCALEBAR_PARAMS)

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
            ax[1].yaxis.set_label_position("right")

        else:
            fig.colorbar(img)

    return fig, ax


def detect_doublets(
    integrity_map,
    signal_map,
    min_distance=10,
    integrity_threshold=0.7,
    minimum_signal_strength=3,
    integrity_sigma=None,
) -> pd.DataFrame:
    """
    This function is used to find individual low peaks of signal integrity in the tissue
    map as an indicator of single occurrences overlapping cells.

    Parameters
    ----------
    integrity_map : numpy.ndarray
        Pixel map of signal integrity obtained from ovrlpy analysis
    signal_map : numpy.ndarray
        Signal strength map from ovrlpy analysis
    min_distance : int, optional
        Minimum distance between reported peaks
    integrity_threshold : float, optional
        Threhold of signal integrity value. A peak with an signal integrity > integrity_threshold is not considered.
    minimum_signal_strength : float, optional
        Minimum signal value for a peak to be considered
    integrity_sigma : float, optional
        Optional sigma value for gaussian filtering of the integrity map,
        which leads to the detection of overlap regions with larger spatial extent.
    """

    if integrity_sigma is not None:
        integrity_map = gaussian_filter(integrity_map, integrity_sigma)

    dist_x, dist_y, dist_t = _determine_localmax_and_sample(
        (1 - integrity_map) * (signal_map > minimum_signal_strength),
        min_distance=min_distance,
        min_expression=1 - integrity_threshold,
    )

    arg_idcs = np.argsort(dist_t)[::-1]
    dist_x, dist_y = dist_x[arg_idcs], dist_y[arg_idcs]

    doublet_df = pd.DataFrame(
        {
            "x": dist_y,
            "y": dist_x,
            "integrity": 1 - dist_t,
            "signal": signal_map[dist_x, dist_y],
        }
    )

    return doublet_df


def _determine_celltype_class_assignments(
    samples: pd.DataFrame, signatures: pd.DataFrame
):
    samples = samples.loc[:, signatures.columns]
    signature_mtx = signatures.to_numpy()
    # TODO: this is quite inefficient?
    # it calculates all pairwise correlation coefficients even if not needed.
    correlations = np.array(
        [
            np.corrcoef(samples.iloc[i, :], signature_mtx)[0, 1:]
            for i in range(samples.shape[0])
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
        The bandwidth of the KDE.
    celltyping_min_expression : int, optional
        Minimum expression level for cell typing.
    celltyping_min_distance : int, optional
        Minimum distance for cell typing.
    n_components_pca : float, optional
        Number of components for PCA.
    dtype
        Datatype for the KDE.
    umap_kwargs : dict, optional
        Keyword arguments for 2D UMAP embedding.
    cumap_kwargs : dict, optional
        Keyword arguments for 3D UMAP embedding.

    Attributes
    ----------
    KDE_bandwidth : float
        The bandwidth of the KDE.
    celltyping_min_expression : int
        Minimum expression level for cell typing.
    celltyping_min_distance : int
        Minimum distance for cell typing.
    pseudocell_locations_x : numpy.ndarray
        x-coordinates of cell typing regions of interest obtained through gene expression localmax sampling.
    pseudocell_locations_y : numpy.ndarray
        y-coordinates of cell typing regions of interest obtained through gene expression localmax sampling.
    pseudocell_expression_samples : pandas.DataFrame
        Gene expression matrix of the cell typing regions of interest.
    signatures : pandas.DataFrame
        A matrix of celltypes x gene signatures to use to annotate the UMAP.
    celltype_centers : numpy.ndarray
        The center of gravity of each celltype in the 2d embedding, used for UMAP annotation.
    celltype_class_assignments : numpy.ndarray
        The class assignments of the cell types.
    pca_2d : sklearn.decomposition.PCA
        The PCA object used for the 2d embedding.
    embedder_2d : umap.UMAP
        The UMAP object used for the 2d embedding.
    pca_3d : sklearn.decomposition.PCA
        The PCA object used for the 3d RGB embedding.
    embedder_3d : umap.UMAP
        The UMAP object used for the 3d RGB embedding.
    n_components_pca : float
        Number of components for PCA.
    umap_kwargs : dict
        Keyword arguments for 2D UMAP embedding object.
    cumap_kwargs : dict
        Keyword arguments for 3D UMAP RGB embedding object.
    genes : list
        A list of genes to utilize in the model.
    embedding : numpy.ndarray
        The 2d embedding of pseudocell gene expression .
    colors : numpy.ndarray
        The RGB embedding.
    colors_min_max : list
        The minimum and maximum values of the RGB embedding, necessary for normalization of the transform method.
    integrity_map : numpy.ndarray
        The integrity map of the tissue.
    signal_map : numpy.ndarray
        A pixel map of overall signal strength in the tissue, used to mask out low-signal regions that are difficult to interpret.
    """

    def __init__(
        self,
        KDE_bandwidth=1.5,
        celltyping_min_expression=10,
        celltyping_min_distance=5,
        n_components_pca=0.7,
        dtype=np.float32,
        umap_kwargs=UMAP_2D_PARAMS,
        cumap_kwargs=UMAP_RGB_PARAMS,
    ) -> None:
        self.KDE_bandwidth = KDE_bandwidth
        self.dtype = dtype

        self.celltyping_min_expression = celltyping_min_expression
        self.celltyping_min_distance = celltyping_min_distance
        self.pseudocell_locations_x, self.pseudocell_locations_y = None, None
        self.pseudocell_expression_samples = None
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

        self.integrity_map = None
        self.signal_map = None

    def fit_transcripts(
        self,
        coordinate_df: pd.DataFrame,
        genes=None,
        gene_key: str = "gene",
        signature_matrix=None,
        fit_umap: bool = True,
        patch_length: int = 500,
        n_workers: int = 8,
    ):
        """
        Fits the visualizer to a spatial transcripts dataset using the SSAM algorithm.

        Parameters
        ----------
        coordinate_df : pandas.DataFrame
            A dataframe of coordinates.
        genes : list
            A list of genes to utilize in the model. None uses all genes.
        gene_key : str
            The key in the dataframe containing the gene names.
        signature_matrix : pandas.DataFrame
            A matrix of celltypes x gene signatures to use to annotate the UMAP.
        fit_umap : bool
            Whether to fit the UMAP to the data.
        patch_length : int
            Size of the length in each dimension when calculating signal integrity in patches.
            Smaller values will use less memory, but may take longer to compute.
        n_workers : int
            The number of workers to use in the SSAM algorithm
        """

        if genes is None:
            genes = sorted(coordinate_df[gene_key].unique())
        self.genes = genes

        local_maxima, coordinates = _sample_expression(
            coordinate_df,
            gene_column=gene_key,
            minimum_expression=self.celltyping_min_expression,
            kde_bandwidth=self.KDE_bandwidth,
            n_workers=n_workers,
            min_pixel_distance=self.celltyping_min_distance,
            patch_length=patch_length,
            dtype=self.dtype,
        )

        self.pseudocell_locations_x, self.pseudocell_locations_y, _ = coordinates.T

        self.fit_pseudocells(local_maxima, fit_umap=fit_umap)

        if signature_matrix is not None:
            self.fit_signatures(signature_matrix)

    def _coordinate_center(self, assignments: np.ndarray, n: int):
        # determine the center of gravity of each assignment (celltype or gene)
        # in the embedding
        return np.array(
            [
                (
                    np.median(self.embedding[assignments == i, :], axis=0)
                    if (assignments == i).sum() > 0
                    else (np.nan, np.nan)
                )
                for i in range(n)
            ]
        )

    def fit_pseudocells(
        self,
        pseudocell_expression_samples: pd.DataFrame,
        *,
        genes: list = None,
        fit_umap: bool = True,
    ):
        """Fits the visualizer to a given pseudocell expression sample.

        Parameters
        ----------
        pseudocell_expression_samples : pandas.DataFrame
            A cell x gene matrix of gene expression
        genes : list, optional
            A list of genes to utilize in the model.
        fit_umap : bool
            Whether to fit the UMAP to the data.
        """

        self.pseudocell_expression_samples = pseudocell_expression_samples

        if genes is None:
            genes = pseudocell_expression_samples.columns
        self.genes = genes

        self.pca_2d = PCA(
            n_components=min(
                self.n_components_pca, pseudocell_expression_samples.shape[1] // 2
            )
        )
        if fit_umap:
            factors = self.pca_2d.fit_transform(pseudocell_expression_samples)

            print(f"Modeling {factors.shape[1]} pseudo-celltype clusters;")

            self.embedder_2d = umap.UMAP(**self.umap_kwargs)
            self.embedding = self.embedder_2d.fit_transform(factors)

            self.embedder_3d = umap.UMAP(**self.cumap_kwargs)
            embedding_color = self.embedder_3d.fit_transform(
                factors / norm(factors, axis=1)[..., None]
            )

            embedding_color, self.pca_3d = _fill_color_axes(embedding_color)

            self.colors = _min_to_max(embedding_color.copy())
            self.colors_min_max = [embedding_color.min(0), embedding_color.max(0)]

        else:
            self.pca_2d.fit(pseudocell_expression_samples)

    def fit_signatures(self, signature_matrix=None):
        """
        Fits the visualizer with a given signature matrix.

        Parameters
        ----------
        signature_matrix : pandas.DataFrame
            A matrix of celltypes x gene signatures to use to annotate the UMAP.
            None defaults to displaying individual genes.
        """

        if signature_matrix is None:
            signature_matrix = pd.DataFrame(
                np.eye(len(self.genes), dtype=self.dtype),
                index=self.genes,
                columns=self.genes,
            )

        self.signatures = signature_matrix

        self.celltype_class_assignments = _determine_celltype_class_assignments(
            self.pseudocell_expression_samples, signature_matrix
        )

        # determine the center of gravity of each celltype in the embedding:
        self.celltype_centers = self._coordinate_center(
            self.celltype_class_assignments, len(signature_matrix.columns)
        )

    def subsample_df(self, x, y, coordinate_df: pd.DataFrame, window_size: int = 30):
        """
        Subsamples the coordinate dataframe spatially based on given x, y coordinates and window
        size.

        Parameters
        ----------
        x : float
            x-coordinate to center the sampling window
        y : float
            y-coordinate to center the sampling window
        coordinate_df : pandas.DataFrame
            DataFrame of gene annotated molecule coordinates to create the subsample from
        window_size : int, optional
            The window size of the sampling window. Molecules within this window around (x,y)
            are sampled and returned as a new DataFrame.
        """

        subsample_mask = _get_spatial_subsample_mask(
            coordinate_df, x, y, plot_window_size=window_size
        )
        subsample = coordinate_df[subsample_mask]

        return subsample

    def transform_transcripts(self, coordinate_df: pd.DataFrame):
        """
        Transforms the coordinate dataframe to the visualizers 2d and 3d embedding space.

        Parameters
        ----------
        coordinate_df : pandas.DataFrame
            Data frame of gene-annotated molecule coordinates to transform.
        """

        distances, neighbor_indices = _create_knn_graph(
            coordinate_df[["x", "y", "z"]].values, k=90
        )
        local_expression = _get_knn_expression(
            distances,
            neighbor_indices,
            self.genes,
            coordinate_df["gene"].cat.codes.values,
            bandwidth=self.KDE_bandwidth,
        )
        local_expression /= (local_expression**2).sum(0) ** 0.5
        return self.transform_pseudocells(local_expression.T)

    def transform_pseudocells(self, pseudocell_expression_samples: pd.DataFrame):
        """Transforms a matrix of gene expression to the visualizer's 2d and 3d embedding space.

        Parameters
        ----------
        pseudocell_expression_samples : pandas.DataFrame
            A cell x gene matrix of gene expression
        """

        subsample_embedding, subsample_embedding_color = _transform_embeddings(
            pseudocell_expression_samples.to_numpy(),
            self.pca_2d,
            embedder_2d=self.embedder_2d,
            embedder_3d=self.embedder_3d,
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

    def pseudocell_df(self) -> pd.DataFrame:
        """
        Returns a pandas.DataFrame containing the gene-count matrix of the fitted
        tissue's determined pseudo-cells.

        Returns
        -------
        pandas.DataFrame
        """
        pseudocell_df = pd.DataFrame(
            {"x": self.pseudocell_locations_x, "y": self.pseudocell_locations_y}
        )

        if self.signal_map is not None:
            pseudocell_df["signal"] = self.signal_map[
                pseudocell_df.y.astype(int), pseudocell_df.x.astype(int)
            ]

        if self.integrity_map is not None:
            pseudocell_df["integrity"] = self.integrity_map[
                pseudocell_df.y.astype(int), pseudocell_df.x.astype(int)
            ]

        return pseudocell_df

    def plot_region_of_interest(
        self,
        subsample: pd.DataFrame,
        subsample_embedding_color: np.ndarray,
        x: float = None,
        y: float = None,
        window_size: int = None,
        rasterized: bool = True,
        scalebar: dict | None = SCALEBAR_PARAMS,
    ):
        """
        Plots an instance of the visualized data.

        Parameters
        ----------
        subsample : pandas.DataFrame
            A dataframe of molecule coordinates and gene assignments.
        subsample_embedding_color : pandas.DataFrame
            A list of rgb values for each molecule.
        x : float
            Center x-coordinate for the region-of-interest.
        y : float
            Center y-coordinate for the region-of-interest.
        window_size : float, optional
            Window size of the region-of-interest.
        rasterized : bool, optional
            If True all plots will be rasterized.
        scalebar : dict[str, typing.Any] | None
            If `None` no scalebar will be plotted. Otherwise a dictionary with
            additional kwargs for ``matplotlib_scalebar.scalebar.ScaleBar``.
            By default :py:attr:`ovrlpy.SCALEBAR_PARAMS`
        """
        vertical_indices = subsample.z.argsort()
        subsample = subsample.sort_values("z")
        subsample_embedding_color = subsample_embedding_color[vertical_indices]

        if x is None:
            x = (subsample.x.max() + subsample.x.min()) / 2
        if y is None:
            y = (subsample.y.max() + subsample.y.min()) / 2
        if window_size is None:
            window_size = int(1 + (subsample.x.max() - subsample.x.min()) / 2)

        roi = ((x - window_size, x + window_size), (y - window_size, y + window_size))

        fig = plt.figure(figsize=(22, 12))

        gs = fig.add_gridspec(2, 3)

        # 3D map
        ax_3d = fig.add_subplot(gs[0, 2], projection="3d", label="3d_map")
        ax_3d.scatter(
            subsample.x,
            subsample.y,
            subsample.z,
            c=subsample_embedding_color,
            marker=".",
            alpha=0.5,
            rasterized=rasterized,
        )
        ax_3d.set_zlim(
            np.median(subsample.z) - window_size, np.median(subsample.z) + window_size
        )
        ax_3d.set_title("ROI celltype map, 3D")

        # UMAP
        ax_umap = fig.add_subplot(gs[0, 0], label="umap")
        ax_umap.scatter(
            self.embedding[:, 0],
            self.embedding[:, 1],
            c=self.colors,
            alpha=0.5,
            marker=".",
            s=1,
            rasterized=rasterized,
        )

        ax_umap.set_axis_off()
        ax_umap.set_title("UMAP")

        # tissue map
        ax_tissue_whole: Axes = fig.add_subplot(gs[0, 1], label="celltype_map")
        self.plot_tissue(rasterized=rasterized, s=1)

        ax_tissue_whole.set_yticks([], [])

        artist = plt.Rectangle(
            (x - window_size, y - window_size),
            2 * window_size,
            2 * window_size,
            fill=False,
            edgecolor="k",
            linewidth=2,
        )
        ax_tissue_whole.add_artist(artist)

        ax_tissue_whole.set_title("celltype map")

        # top view of ROI
        roi_scatter_kwargs = dict(marker=".", alpha=0.8, s=40, rasterized=rasterized)

        def _plot_tissue_scatter_roi(ax: Axes, x, y, roi, *, rasterized: bool = False):
            ax.scatter(x, y, c="k", marker="+", s=100, rasterized=rasterized)
            ax.set(xlim=roi[0], ylim=roi[1])

        ax_roi_top = fig.add_subplot(gs[1, 0], label="top_map")
        top_mask = subsample.z > subsample.z_delim
        subsample_top = subsample[top_mask]
        self._plot_tissue_scatter(
            ax_roi_top,
            subsample_top["x"],
            subsample_top["y"],
            subsample_embedding_color[top_mask],
            title="ROI celltype map, top",
            **roi_scatter_kwargs,
        )
        _plot_tissue_scatter_roi(ax_roi_top, x, y, roi, rasterized=rasterized)

        ax_roi_bottom = fig.add_subplot(gs[1, 1], label="bottom_map")
        bottom_mask = subsample.z < subsample.z_delim
        subsample_bottom = subsample[bottom_mask][::-1]
        self._plot_tissue_scatter(
            ax_roi_bottom,
            subsample_bottom["x"],
            subsample_bottom["y"],
            subsample_embedding_color[bottom_mask][::-1],
            title="ROI celltype map, bottom",
            **roi_scatter_kwargs,
        )

        _plot_tissue_scatter_roi(ax_roi_bottom, x, y, roi, rasterized=rasterized)

        # side view of ROI
        roi_side_scatter_kwargs = dict(s=10, alpha=0.5, rasterized=rasterized)

        sub_gs = gs[1, 2].subgridspec(2, 1)

        ax_side_x = fig.add_subplot(sub_gs[0, 0], label="x_cut")
        halving_mask = (subsample.y < (y + 4)) & (subsample.y > (y - 4))

        self._plot_tissue_scatter(
            ax_side_x,
            subsample.x[halving_mask],
            subsample.z[halving_mask],
            subsample_embedding_color[halving_mask],
            title="ROI, vertical, x-cut",
            **roi_side_scatter_kwargs,
        )

        ax_side_y = fig.add_subplot(sub_gs[1, 0], label="y_cut")
        halving_mask = (subsample.x < (x + 4)) & (subsample.x > (x - 4))

        self._plot_tissue_scatter(
            ax_side_y,
            subsample.y[halving_mask],
            subsample.z[halving_mask],
            subsample_embedding_color[halving_mask],
            title="ROI, vertical, y-cut",
            **roi_side_scatter_kwargs,
        )

        if scalebar is not None:
            _plot_scalebar(ax_tissue_whole, **SCALEBAR_PARAMS)
            _plot_scalebar(ax_roi_top, **SCALEBAR_PARAMS)
            _plot_scalebar(ax_roi_bottom, **SCALEBAR_PARAMS)

    def plot_umap(
        self,
        ax: Optional[Axes] = None,
        rasterized: bool = False,
        **kwargs,
    ):
        """
        Plots the UMAP embedding.

        Parameters
        ----------
        ax : typing.Optional[matplotlib.axes.Axes]
            Axis object to plot on.
        rasterized : bool, optional
            If True the plot will be rasterized.
        kwargs
            Keyword arguments for :py:func:`matplotlib.pyplot.scatter`.
        """
        _plot_embeddings(
            self.embedding,
            self.colors,
            self.celltype_centers,
            None if self.signatures is None else self.signatures.index,
            rasterized,
            ax,
            **kwargs,
        )

    def plot_tissue(
        self,
        rasterized: bool = False,
        scalebar: dict | None = SCALEBAR_PARAMS,
        **kwargs,
    ):
        """
        Plots the tissue embedding.

        Parameters
        ----------
        rasterized : bool, optional
            If True the plot will be rasterized.
        scalebar : dict[str, typing.Any] | None
            If `None` no scalebar will be plotted. Otherwise a dictionary with
            additional kwargs for ``matplotlib_scalebar.scalebar.ScaleBar``.
            By default :py:attr:`ovrlpy.SCALEBAR_PARAMS`
        kwargs
            Keyword arguments for the matplotlib's scatter plot function.
        """
        ax = plt.gca()
        self._plot_tissue_scatter(
            ax,
            self.pseudocell_locations_x,
            self.pseudocell_locations_y,
            self.colors,
            marker=".",
            alpha=1,
            rasterized=rasterized,
            **kwargs,
        )

        if scalebar is not None:
            _plot_scalebar(ax, **SCALEBAR_PARAMS)

    @staticmethod
    def _plot_tissue_scatter(
        ax: Axes, xs, ys, cs, *, title: Optional[str] = None, **kwargs
    ):
        ax.scatter(xs, ys, c=cs, **kwargs)
        ax.set_aspect("equal", adjustable="box")
        if title is not None:
            ax.set_title(title)

    def plot_fit(
        self,
        rasterized: bool = True,
        umap_kwargs={"scatter_kwargs": {"s": 1}},
        tissue_kwargs={"s": 1},
    ):
        """
        Plots the fitted model.

        Parameters
        ----------
        rasterized : bool, optional
            If True all plots will be rasterized.
        """

        plt.figure(figsize=(15, 7))

        plt.subplot(121)
        self.plot_umap(rasterized=rasterized, **umap_kwargs)

        plt.subplot(122)
        self.plot_tissue(rasterized=rasterized, **tissue_kwargs)


def plot_region_of_interest(
    x: float,
    y: float,
    coordinate_df: pd.DataFrame,
    visualizer: Visualizer,
    integrity_map: np.ndarray,
    signal_map: np.ndarray,
    window_size: int = 30,
    signal_plot_threshold: float = 5.0,
    rasterized: bool = True,
    scalebar: dict | None = SCALEBAR_PARAMS,
):
    """Plot a comprehensive overview of a zoomed-in region of interest.

    Parameters
    ----------
    x : float
        x coordinate of the region of interest
    y : float
        y coordinate of the region of interest
    coordinate_df : pandas.DataFrame
        DataFrame of gene-annotated molecule coordinates
    visualizer : Visualizer
        Visualizer object containing the fitted model
    integrity_map : numpy.ndarray
        integrity map of the tissue
    signal_map : numpy.ndarray
        Signal map of the tissue
    window_size : int, optional
        Size of the window to display. Defaults to 30.
    signal_plot_threshold : float, optional
        Threshold for the signal plot. Defaults to 5.
    rasterized : bool, optional
        If True all plots will be rasterized. Defaults to True.
    scalebar : dict[str, typing.Any] | None
        If `None` no scalebar will be plotted. Otherwise a dictionary with
        additional kwargs for ``matplotlib_scalebar.scalebar.ScaleBar``.
        By default :py:attr:`ovrlpy.SCALEBAR_PARAMS`
    """

    # first, create and color-embed the subsample of the region of interest:
    subsample = visualizer.subsample_df(x, y, coordinate_df, window_size=window_size)
    _, subsample_embedding_color = visualizer.transform_transcripts(subsample)

    vertical_indices = subsample.z.argsort()
    subsample = subsample.sort_values("z")
    subsample_embedding_color = subsample_embedding_color[vertical_indices]

    if x is None:
        x = (subsample.x.max() + subsample.x.min()) / 2
    if y is None:
        y = (subsample.y.max() + subsample.y.min()) / 2
    if window_size is None:
        window_size = int(1 + (subsample.x.max() - subsample.x.min()) / 2)

    roi = ((x - window_size, x + window_size), (y - window_size, y + window_size))

    fig = plt.figure(figsize=(22, 12))

    gs = fig.add_gridspec(2, 3)

    # integrity map
    ax_integrity = fig.add_subplot(gs[0, 2], label="signal_integrity", facecolor="k")

    handler = ax_integrity.imshow(
        integrity_map,
        cmap=_BIH_CMAP,
        alpha=(signal_map / signal_plot_threshold).clip(0, 1),
        vmin=0,
        vmax=1,
    )

    ax_integrity.set_title("signal integrity")
    ax_integrity.invert_yaxis()
    plt.colorbar(handler)

    ax_integrity.set_xlim(x - window_size, x + window_size)
    ax_integrity.set_ylim(y - window_size, y + window_size)

    # UMAP
    ax_umap = fig.add_subplot(gs[0, 0], label="umap")
    ax_umap.scatter(
        visualizer.embedding[:, 0],
        visualizer.embedding[:, 1],
        c=visualizer.colors,
        alpha=0.5,
        marker=".",
        s=1,
        rasterized=rasterized,
    )

    ax_umap.set_axis_off()
    ax_umap.set_title("UMAP")

    # tissue map
    ax_tissue_whole: Axes = fig.add_subplot(gs[0, 1], label="celltype_map")
    visualizer.plot_tissue(rasterized=rasterized, s=1)

    # ax_tissue_whole.set_yticks([], [])

    artist = plt.Rectangle(
        (x - window_size, y - window_size),
        2 * window_size,
        2 * window_size,
        fill=False,
        edgecolor="k",
        linewidth=2,
    )
    ax_tissue_whole.add_artist(artist)

    ax_tissue_whole.set_title("celltype map")

    # top view of ROI
    roi_scatter_kwargs = dict(
        marker=".", alpha=0.8, s=1.5e5 / window_size**2, rasterized=rasterized
    )

    def _plot_tissue_scatter_roi(ax: Axes, x, y, roi, *, rasterized: bool = False):
        ax.scatter(x, y, c="k", marker="+", s=100, rasterized=rasterized)
        ax.set(xlim=roi[0], ylim=roi[1])

    ax_roi_top = fig.add_subplot(gs[1, 0], label="top_map")
    top_mask = subsample.z > subsample.z_delim
    subsample_top = subsample[top_mask]
    visualizer._plot_tissue_scatter(
        ax_roi_top,
        subsample_top["x"],
        subsample_top["y"],
        subsample_embedding_color[top_mask],
        title="ROI celltype map, top",
        **roi_scatter_kwargs,
    )
    _plot_tissue_scatter_roi(ax_roi_top, x, y, roi, rasterized=rasterized)

    ax_roi_bottom = fig.add_subplot(gs[1, 1], label="bottom_map")
    bottom_mask = subsample.z < subsample.z_delim
    subsample_bottom = subsample[bottom_mask][::-1]
    visualizer._plot_tissue_scatter(
        ax_roi_bottom,
        subsample_bottom["x"],
        subsample_bottom["y"],
        subsample_embedding_color[bottom_mask][::-1],
        title="ROI celltype map, bottom",
        **roi_scatter_kwargs,
    )

    _plot_tissue_scatter_roi(ax_roi_bottom, x, y, roi, rasterized=rasterized)

    if scalebar is not None:
        _plot_scalebar(ax_integrity, **SCALEBAR_PARAMS)
        _plot_scalebar(ax_tissue_whole, **SCALEBAR_PARAMS)
        _plot_scalebar(ax_roi_top, **SCALEBAR_PARAMS)
        _plot_scalebar(ax_roi_bottom, **SCALEBAR_PARAMS)

    # side view of ROI
    roi_side_scatter_kwargs = dict(s=10, alpha=0.5, rasterized=rasterized)

    sub_gs = gs[1, 2].subgridspec(2, 1)

    ax_side_x = fig.add_subplot(sub_gs[0, 0], label="x_cut")
    halving_mask = (subsample.y < (y + 4)) & (subsample.y > (y - 4))

    visualizer._plot_tissue_scatter(
        ax_side_x,
        subsample.x[halving_mask],
        subsample.z[halving_mask],
        subsample_embedding_color[halving_mask],
        title="ROI, vertical, x-cut",
        **roi_side_scatter_kwargs,
    )

    ax_side_y = fig.add_subplot(sub_gs[1, 0], label="y_cut")
    halving_mask = (subsample.x < (x + 4)) & (subsample.x > (x - 4))

    visualizer._plot_tissue_scatter(
        ax_side_y,
        subsample.y[halving_mask],
        subsample.z[halving_mask],
        subsample_embedding_color[halving_mask],
        title="ROI, vertical, y-cut",
        **roi_side_scatter_kwargs,
    )

    return fig, [
        [ax_umap, ax_tissue_whole, ax_integrity],
        [ax_roi_top, ax_roi_bottom, [ax_side_x, ax_side_y]],
    ]


def run(
    df: pd.DataFrame,
    n_expected_celltypes=None,
    cell_diameter=10,
    min_expression: float = 1.0,
    signature_matrix=None,
    fit_umap: bool = True,
    patch_length: int = 500,
    n_workers: int = min(8, len(os.sched_getaffinity(0))),
    umap_kwargs=UMAP_2D_PARAMS,
    cumap_kwargs=UMAP_RGB_PARAMS,
):
    """
    This is a wrapper function that computes the integrity map for a given spatial
    transcriptomics dataset.
    It includes the entire ovrlpy pipeline, returning a integrity map and a signal
    strength map and produces a visualizer object that can be used to plot the results.

    Parameters
    ----------
    df : pandas.DataFrame
        The spatial transcriptomics dataset.
        This dataframe should contain a *gene*, *x*, *y*, and *z* column.
    n_expected_celltypes : int, optional
        A rough estimate of expected cell type clusters.
        A higher number will lead to a more disjoint UMAP embedding and a higher
        detection of overlap events.
        Defaults to None, which will use a heuristic on the PCA explained variance to determine cluster count.
    cell_diameter : float, optional
        The expected diameter of the cell/structure of interest in the sample.
        Defaults to 10.
    min_expression : int, optional
        A threshold value to discard areas with low expression, by default 5
        Can be interpreted as: The minimum number of transcripts to appear within a
        radius of 'KDE_bandwidth' for a region to be considered containing a cell.
    signature_matrix : pandas.DataFrame, optional
        A celltypes x gene table of expression signatures to use to annotate the UMAP.
    fit_umap : bool
        Whether to fit the UMAP to the data.
    patch_length : int
        Size of the length in each dimension when calculating signal integrity in patches.
        Smaller values will use less memory, but may take longer to compute.
    n_workers : int
        The number of workers to use for processsing.
    umap_kwargs : dict, optional
        Additional keyword arguments for the 2D UMAP embedding.
    cumap_kwargs : dict, optional
        Additional keyword arguments for the 3D UMAP embedding.

    Returns
    -------
    integrity_map : numpy.ndarray
        A integrity pixel map of the spatial transcriptomics dataset.
    signal_map : numpy.ndarray
        A pixel map of the spatial transcriptomics dataset containing the detected signal.
    vis : Visualizer
        A visualizer object that can be used to plot the
        results of the integrity map
    """

    KDE_bandwidth = cell_diameter / 4
    min_distance = cell_diameter * 0.9

    if n_expected_celltypes is None:
        n_expected_celltypes = 0.8

    print("Running vertical adjustment")
    pre_process_coordinates(df, grid_size=1, rounds=20)

    if "random_state" not in umap_kwargs:
        umap_kwargs = {"n_jobs": n_workers} | umap_kwargs

    if "random_state" not in cumap_kwargs:
        cumap_kwargs = {"n_jobs": n_workers} | cumap_kwargs

    vis = Visualizer(
        KDE_bandwidth=KDE_bandwidth,
        celltyping_min_expression=min_expression,
        celltyping_min_distance=min_distance,
        n_components_pca=n_expected_celltypes,
        umap_kwargs=umap_kwargs,
        cumap_kwargs=cumap_kwargs,
    )

    print("Creating gene expression embeddings for visualization:")
    vis.fit_transcripts(
        df,
        signature_matrix=signature_matrix,
        fit_umap=fit_umap,
        patch_length=patch_length,
        n_workers=n_workers,
    )

    print("Creating signal integrity map:")
    integrity_, signal_ = compute_VSI(
        df,
        pd.DataFrame(vis.pca_2d.components_, columns=vis.genes),
        KDE_bandwidth=KDE_bandwidth,
        min_expression=None,
        patch_length=patch_length,
        n_workers=n_workers,
    )

    vis.integrity_map = integrity_.T
    vis.signal_map = signal_.T

    return integrity_.T, signal_.T, vis
