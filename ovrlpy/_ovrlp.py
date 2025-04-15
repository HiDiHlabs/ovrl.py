import os
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import reduce
from math import ceil
from queue import SimpleQueue
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import polars as pl
import tqdm
from anndata import AnnData
from scipy.linalg import norm
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA
from umap import UMAP

from ._kde import _TRUNCATE, _sample_expression, kde_2d
from ._patching import _patches, n_patches
from ._subslicing import pre_process_coordinates
from ._utils import (
    UMAP_2D_PARAMS,
    UMAP_RGB_PARAMS,
    _calculate_embedding,
    _cosine_similarity,
    _create_knn_graph,
    _determine_localmax_and_sample,
    _fill_color_axes,
    _knn_expression,
    _minmax_scaling,
    _transform_embeddings,
)


class Ovrlp:
    """
    Main analysis class for spatial overlap analysis.

    Parameters
    ----------
    transcripts : polars.DataFrame | pandas.DataFrame
        Transcript information containing coordinates and gene name/id.
    KDE_bandwidth : float, optional
        The bandwidth of the KDE.
    min_distance : float, optional
        Minimum distance for cell typing.
    n_components : int, optional
        Number of components for PCA.
    gene_key : str
        Name of the gene column
    coordinate_keys : collections.abc.Sequence[str]
        Names of the coordinate columns.
    n_workers : int
        Number of threads sed in parallel processing.
    dtype : numpy.typing.DTypeLike
        Datatype used for KDE calcculations.
    patch_length : int
        Upper bound for size of each patch. (Only relevant for processing)
    umap_kwargs : dict, optional
        Keyword arguments for 2D UMAP embedding.
    cumap_kwargs : dict, optional
        Keyword arguments for 3D UMAP embedding.

    Attributes
    ----------
    transcripts : polars.DataFrame
        Transcript information containing coordinates and gene name/id.
    KDE_bandwidth : float
        The bandwidth of the KDE.
    min_distance : int
        Minimum distance between pseudocells (local maxima).
    pseudocells : anndata.AnnData
        Gene expression matrix of the pseudcells.
    signatures : pandas.DataFrame
        A matrix of celltypes x gene signatures to use to annotate the UMAP.
    celltype_centers : numpy.ndarray
        The center of gravity of each celltype in the 2D embedding, used for UMAP annotation.
    celltype_assignments : numpy.ndarray
        The assignments of the cell types.
    pca_2d : sklearn.decomposition.PCA
        The PCA object used for the 2D embedding.
    embedder_2d : umap.UMAP
        The UMAP object used for the 2D embedding.
    pca_3d : sklearn.decomposition.PCA
        The PCA object used for the 3D RGB embedding.
    embedder_3d : umap.UMAP
        The UMAP object used for the 3D RGB embedding.
    genes : list
        A list of genes to utilize in the model.
    integrity_map : numpy.ndarray
        The integrity map of the tissue.
    signal_map : numpy.ndarray
        A pixel map of overall signal strength in the tissue, used to mask out low-signal regions.
    dtype : numpy.typing.DTypeLike
        Datatype used for KDE calcculations.
    patch_length : int
        Upper bound for size of each patch. (Only relevant for processing)
    n_workers : int
        Number of threads used in parallel processing.
    """

    def __init__(
        self,
        transcripts: pl.DataFrame | pd.DataFrame,
        /,
        KDE_bandwidth: float = 2.5,
        min_distance: float = 8,
        n_components: int = 30,
        *,
        gene_key: str = "gene",
        coordinate_keys: tuple[str, str, str] = ("x", "y", "z"),
        n_workers: int = min(8, len(os.sched_getaffinity(0))),
        dtype: npt.DTypeLike = np.float32,
        patch_length: int = 500,
        umap_kwargs: dict[str, Any] = UMAP_2D_PARAMS,
        cumap_kwargs: dict[str, Any] = UMAP_RGB_PARAMS,
    ) -> None:
        columns = {gene_key: "gene"} | dict(zip(coordinate_keys, ["x", "y", "z"]))
        if isinstance(transcripts, pd.DataFrame):
            transcripts = pl.from_pandas(transcripts)
        self.transcripts = transcripts.rename(columns)

        self.KDE_bandwidth = KDE_bandwidth
        self.dtype = dtype

        self.n_workers = n_workers
        self.patch_length = patch_length

        self.min_distance = min_distance

        if "random_state" not in umap_kwargs and "n_jobs" not in umap_kwargs:
            umap_kwargs = {"n_jobs": n_workers} | umap_kwargs

        if "random_state" not in cumap_kwargs and "n_jobs" not in cumap_kwargs:
            cumap_kwargs = {"n_jobs": n_workers} | cumap_kwargs

        self.pca_2d = PCA(n_components=n_components)
        self.embedder_2d = UMAP(**(umap_kwargs | {"n_components": 2}))
        self.pca_3d = PCA(n_components=3)
        self.embedder_3d = UMAP(**(cumap_kwargs | {"n_components": 3}))

    def process_coordinates(self, gridsize: float = 1, **kwargs):
        """
        Process the coordinates of the transcripts dataframe.

        Parameters
        ----------
        gridsize : float, optional
            The size of the pixel grid.
        kwargs
            Other keyword arguments are passed to :py:func:`ovrlpy.pre_process_coordinates`
        """
        self.transcripts = pre_process_coordinates(
            self.transcripts, gridsize=gridsize, **kwargs
        )

    def _expression_threshold(self, n: float = 10, scale: float = 1.1) -> float:
        return n * scale / (2 * np.pi * self.KDE_bandwidth**2)

    def fit_transcripts(
        self,
        /,
        min_transcript: float = 10,
        *,
        genes: list | None = None,
        fit_umap: bool = True,
    ):
        """
        Fits a spatial transcripts dataset using the SSAM algorithm.

        Parameters
        ----------
        min_transcript : float
            Minimum expression for a local maximum to be considered. Expressed in terms
            of transcripts.
        genes : list
            A list of genes to utilize in the model. `None` uses all genes.
        fit_umap : bool
            Whether to fit the UMAP to the data.
        """

        if genes is None:
            genes = sorted(self.transcripts["gene"].unique())
        self.genes = genes

        local_maxima = _sample_expression(
            self.transcripts,
            min_expression=self._expression_threshold(min_transcript),
            kde_bandwidth=self.KDE_bandwidth,
            n_workers=self.n_workers,
            min_pixel_distance=self.min_distance,
            patch_length=self.patch_length,
            dtype=self.dtype,
        )

        self.fit_pseudocells(local_maxima, fit_umap=fit_umap)

    def fit_pseudocells(self, pseudocells: AnnData, *, fit_umap: bool = True):
        """
        Fits the expression of pseudocells.

        Parameters
        ----------
        pseudocells : anndata.AnnData
            Gene expression to use for fitting.
        fit_umap : bool
            Whether to fit the UMAP to the data.
        """

        self.pseudocells = pseudocells
        X = pseudocells[:, self.genes].X
        self.pca_2d.fit(X)

        if fit_umap:
            factors = self.pca_2d.transform(X)

            print(f"Modeling {factors.shape[1]} pseudo-celltype clusters;")

            self.pseudocells.obsm["2D_UMAP"] = self.embedder_2d.fit_transform(factors)

            embedding_color = self.embedder_3d.fit_transform(
                factors / norm(factors, axis=1)[..., None]
            )
            embedding_color = _fill_color_axes(embedding_color, self.pca_3d, fit=True)

            self._colors_min_max = (
                embedding_color.min(axis=0),
                embedding_color.max(axis=0),
            )

            self.pseudocells.obsm["RGB"] = _minmax_scaling(embedding_color)

    @staticmethod
    def _determine_celltype(samples: AnnData, signatures: pd.DataFrame) -> np.ndarray:
        X = samples[:, signatures.columns].X
        signature_mtx = signatures.to_numpy()
        # TODO: this is quite inefficient?
        # it calculates all pairwise correlation coefficients even if not needed.
        correlations = np.array(
            [np.corrcoef(X[i, :], signature_mtx)[0, 1:] for i in range(X.shape[0])]
        )
        return np.argmax(correlations, axis=-1)

    @staticmethod
    def _coordinate_center(
        embedding: np.ndarray, assignments: np.ndarray, n: int
    ) -> np.ndarray:
        # determine the center of gravity of each assignment (celltype or gene)
        # in the embedding
        return np.array(
            [
                (
                    np.median(embedding[assignments == i, :], axis=0)
                    if (assignments == i).sum() > 0
                    else (np.nan, np.nan)
                )
                for i in range(n)
            ]
        )

    def fit_signatures(self, signatures: pd.DataFrame):
        """
        Fits a signature matrix.

        Parameters
        ----------
        signatures : pandas.DataFrame
            A matrix of celltypes x gene signatures to use to annotate the UMAP.
        """

        if self.pseudocells is None:
            raise Exception("`fit_pseudocells` must be run before fitting signatures")
        if "2D_UMAP" not in self.pseudocells.obsm:
            raise Exception("Signature can only be fitted if a UMAP has been fit")

        self.signatures = signatures

        self.celltype_assignments = self._determine_celltype(
            self.pseudocells, signatures
        )

        # determine the center of gravity of each celltype in the embedding:
        self.celltype_centers = self._coordinate_center(
            self.pseudocells.obsm["2D_UMAP"],
            self.celltype_assignments,
            len(signatures.columns),
        )

    def compute_VSI(self, *, min_transcript: float = 2):
        """
        Calculate the vertical signal integrity (VSI).

        Parameters
        ----------
        min_transcript : float | None, optional
            Minimum expression value to consider in calculation.
            Defaults to the 110% of the maximum expression profile of two molecules in the KDE.
        """
        assert self.genes is not None

        min_expression = self._expression_threshold(min_transcript)

        padding = int(ceil(_TRUNCATE * self.KDE_bandwidth))

        gene2idx = {gene: i for i, gene in enumerate(self.genes)}

        signal = kde_2d(
            self.transcripts["x"].to_numpy(),
            self.transcripts["y"].to_numpy(),
            bandwidth=self.KDE_bandwidth,
            dtype=self.dtype,
        )

        cosine_similarity = np.zeros_like(signal)

        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            for patch_df, offset, size in tqdm.tqdm(
                _patches(
                    self.transcripts.select(["gene", "x", "y", "z", "z_delim"]),
                    self.patch_length,
                    padding,
                    size=signal.shape,
                ),
                total=n_patches(self.patch_length, signal.shape),
            ):
                assert isinstance(patch_df, pl.DataFrame)

                if len(patch_df) == 0:
                    continue
                patch_signal = kde_2d(
                    patch_df["x"].to_numpy(),
                    patch_df["y"].to_numpy(),
                    bandwidth=self.KDE_bandwidth,
                    dtype=self.dtype,
                )
                patch_mask = patch_signal > min_expression
                n_pixels = patch_mask.sum()

                if n_pixels == 0:
                    continue

                patch_df = patch_df.filter(pl.col("gene").is_in(gene2idx))
                gene_queue: SimpleQueue[tuple[int, pl.DataFrame]] = SimpleQueue()
                for (gene, *_), df in patch_df.group_by("gene"):
                    if gene in gene2idx:
                        gene_queue.put((gene2idx[gene], df.drop("gene")))

                embedding_top, embedding_bottom = reduce(
                    lambda x, y: (x[0] + y[0], x[1] + y[1]),  # type: ignore
                    map(
                        lambda x: x.result(),
                        as_completed(
                            executor.submit(
                                _calculate_embedding,
                                gene_queue,
                                patch_mask,
                                self.pca_2d.components_,
                                bandwidth=self.KDE_bandwidth,
                                dtype=self.dtype,
                            )
                            for _ in range(self.n_workers)
                        ),
                    ),
                )

                if (isinstance(embedding_top, int) and embedding_top == 0) or (
                    isinstance(embedding_bottom, int) and embedding_bottom == 0
                ):
                    continue
                assert isinstance(embedding_top, np.ndarray)
                assert isinstance(embedding_bottom, np.ndarray)
                patch_cosine_similarity = np.zeros_like(patch_signal)
                patch_cosine_similarity[patch_mask] = _cosine_similarity(
                    embedding_top, embedding_bottom
                )
                # remove padding
                patch_cosine_similarity = patch_cosine_similarity[
                    padding : padding + size[0], padding : padding + size[1]
                ]

                x_pad = offset[0] + padding
                y_pad = offset[1] + padding

                cosine_similarity[
                    x_pad : x_pad + patch_cosine_similarity.shape[0],
                    y_pad : y_pad + patch_cosine_similarity.shape[1],
                ] = patch_cosine_similarity

        self.signal_map = signal.T
        self.integrity_map = cosine_similarity.T

    def analyse(
        self, gridsize: float = 1, genes: list | None = None, fit_umap: bool = True
    ):
        """
        Run main ovrlpy analysis.

        Parameters
        ----------
        gridsize : float, optional
            The size of the pixel grid.
        genes : list
            A list of genes to utilize in the model. `None` uses all genes.
        fit_umap : bool
            Whether to fit the UMAP to the data.
        """

        print("Running vertical adjustment")
        self.process_coordinates(gridsize=gridsize, n_iter=20)

        print("Creating gene expression embeddings for visualization")
        self.fit_transcripts(genes=genes, fit_umap=fit_umap)

        print("Creating signal integrity map")
        self.compute_VSI()

    def detect_doublets(
        self,
        min_distance: int = 10,
        min_integrity: float = 0.7,
        min_signal: float = 3,
        integrity_sigma: float | None = None,
    ) -> pd.DataFrame:
        """
        This function is used to find individual low peaks of signal integrity in the tissue
        map as an indicator of single occurrences overlapping cells.

        Parameters
        ----------
        min_distance : int, optional
            Minimum distance between reported peaks
        min_integrity : float, optional
            Threshold of signal integrity value. A peak with an
            `signal_integrity < min_integrity` is not considered.
        min_signal : float, optional
            Minimum signal value for a peak to be considered
        integrity_sigma : float, optional
            Optional sigma value for gaussian filtering of the integrity map,
            which leads to the detection of overlap regions with larger spatial extent.
        """

        if not hasattr(self, "signal_map") or not hasattr(self, "integrity_map"):
            raise Exception("Run `compute_VSI` before detecting doublets")

        if integrity_sigma is not None:
            integrity_map = gaussian_filter(self.integrity_map, integrity_sigma)
        else:
            integrity_map = self.integrity_map

        dist_x, dist_y, dist_t = _determine_localmax_and_sample(
            (1 - integrity_map) * (self.signal_map > min_signal),
            min_distance=min_distance,
            min_value=1 - min_integrity,
        )

        doublets = pd.DataFrame(
            {
                "x": dist_y,
                "y": dist_x,
                "integrity": 1 - dist_t,
                "signal": self.signal_map[dist_x, dist_y],
            }
        ).sort_values("integrity")

        return doublets

    def subset_transcripts(
        self, x: float, y: float, *, window_size: int = 30
    ) -> pl.DataFrame:
        """
        Subset the transcript dataframe spatially based on given x, y coordinates and window
        size.

        Parameters
        ----------
        x : float
            x-coordinate to center the region
        y : float
            y-coordinate to center the region
        window_size : int, optional
            The window size of the region. Molecules within this window around (x, y)
            are returned as a new DataFrame.
        """
        return self.transcripts.filter(
            pl.col("x").is_between(x - window_size, x + window_size)
            & pl.col("y").is_between(y - window_size, y + window_size)
        ).clone()

    def transform_transcripts(
        self,
        transcripts: pl.DataFrame | pd.DataFrame,
        *,
        gene_key: str = "gene",
        coordinate_keys: Sequence[str] = ["x", "y", "z"],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Transforms the coordinate dataframe to the 2D and 3D embedding space.

        Parameters
        ----------
        transcripts : polars.DataFrame
            Data frame of transcript coordinates to transform.
        gene_key : str
            Name of the gene column.
        coordinate_keys : collections.abc.Sequence[str]
            Names of the coordinate columns.
        """
        assert self.genes is not None

        distances, neighbor_indices = _create_knn_graph(
            transcripts[list(coordinate_keys)].to_numpy(), k=90
        )

        # ensure the gene categories have the same order as the previously fit transcripts
        genes = pd.Categorical(transcripts[gene_key], categories=self.genes).codes

        expression = _knn_expression(
            genes, distances, neighbor_indices, self.genes, bandwidth=self.KDE_bandwidth
        )
        expression /= np.linalg.norm(expression, axis=1)[:, None]
        return self.transform_pseudocells(expression)

    def transform_pseudocells(
        self, pseudocells: pl.DataFrame | pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Transforms a matrix of gene expression to the 2D and 3D embedding space.

        Parameters
        ----------
        pseudocells : polars.DataFrame | pandas.DataFrame
            A cell x gene matrix of gene expression
        """

        embedding, embedding_color = _transform_embeddings(
            pseudocells.to_numpy(),
            self.pca_2d,
            embedder_2d=self.embedder_2d,
            embedder_3d=self.embedder_3d,
        )
        embedding_color = _fill_color_axes(embedding_color, self.pca_3d)
        color_min, color_max = self._colors_min_max
        embedding_color = (embedding_color - color_min) / (color_max - color_min)
        embedding_color = np.clip(embedding_color, 0, 1)

        return embedding, embedding_color

    def pseudocell_integrity(self) -> pd.DataFrame:
        """
        Returns a DataFrame containing the gene-count matrix of the fitted
        tissue's determined pseudo-cells.

        Returns
        -------
        pandas.DataFrame
        """
        assert self.pseudocells is not None
        pseudocells = pd.DataFrame(
            self.pseudocells.obsm["spatial"][:, :2], columns=["x", "y"]
        )

        if self.signal_map is not None:
            pseudocells["signal"] = self.signal_map[pseudocells["y"], pseudocells["x"]]

        if self.integrity_map is not None:
            pseudocells["integrity"] = self.integrity_map[
                pseudocells["y"], pseudocells["x"]
            ]

        return pseudocells
