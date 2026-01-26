from __future__ import annotations

from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import reduce
from math import ceil
from queue import SimpleQueue
from typing import TYPE_CHECKING, Any, overload

import numpy as np
import numpy.typing as npt
import polars as pl
import tqdm
from scipy.linalg import norm
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA
from umap import UMAP

from ._kde import _TRUNCATE, KDE_2D, _sample_expression, kde_2d_discrete
from ._patching import _patches, n_patches
from ._subslicing import process_coordinates
from ._utils import (
    UMAP_2D_PARAMS,
    UMAP_RGB_PARAMS,
    _calculate_embedding,
    _cosine_similarity,
    _determine_localmax_and_sample,
    _fill_color_axes,
    _gaussian_weighted_neighbors,
    _minmax_scaling,
    _transform_embeddings,
    _weighted_average_expression,
)

if TYPE_CHECKING:
    import pandas as pd
    from anndata import AnnData


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
    coordinate_keys : collections.abc.Iterable[str]
        Names of the coordinate columns.
    n_workers : int
        Number of threads used in parallel processing.
    dtype : numpy.typing.DTypeLike
        Datatype used for KDE calculations.
    patch_length : int
        Upper bound for size of each patch. (Only relevant for processing)
    umap_kwargs : dict, optional
        Keyword arguments for 2D UMAP embedding.
    cumap_kwargs : dict, optional
        Keyword arguments for 3D UMAP embedding.
    random_state : int | numpy.random.RandomState | None
        Random state used to seed UMAP and PCA.

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
    signatures : polars.DataFrame
        A dataframe of celltypes x gene signatures used to annotate the UMAP.
    celltype_centers : numpy.ndarray
        The center of gravity of each celltype in the 2D embedding, used for UMAP annotation.
    celltype_assignments : numpy.ndarray
        The assignments of the cell types.
    pca : sklearn.decomposition.PCA
        The PCA object used for the 2D embedding and calculating the VSI score.
    umap_2d : umap.UMAP
        The UMAP object used for the 2D embedding.
    pca_rgb : sklearn.decomposition.PCA
        The PCA object used for the 3D RGB embedding.
    umap_rgb : umap.UMAP
        The UMAP object used for the 3D RGB embedding.
    genes : list
        A list of genes to utilize in the model.
    gridsize : float
        The size of a pixel.
    origin : tuple[float, float]
        The origin of the grid (corresponds to the amount the coordinates
        have been shifted to minimize the analysis area).
    integrity_map : numpy.ndarray
        The integrity map of the tissue.
    signal_map : numpy.ndarray
        A pixel map of overall signal strength in the tissue, used to mask out low-signal regions.
    dtype : numpy.typing.DTypeLike
        Datatype used for KDE calculations.
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
        coordinate_keys: Iterable[str] = ("x", "y", "z"),
        n_workers: int = 1,
        dtype: npt.DTypeLike = np.float32,
        patch_length: int = 500,
        umap_kwargs: dict[str, Any] = UMAP_2D_PARAMS,
        cumap_kwargs: dict[str, Any] = UMAP_RGB_PARAMS,
        random_state: int | np.random.RandomState | None = None,
    ) -> None:
        columns = {gene_key: "gene"} | dict(zip(coordinate_keys, ["x", "y", "z"]))
        if not isinstance(transcripts, pl.DataFrame):
            transcripts = pl.from_pandas(transcripts)
        self.transcripts = transcripts.rename(columns)

        self.KDE_bandwidth = KDE_bandwidth
        self.dtype = dtype

        self.n_workers = n_workers
        self.patch_length = patch_length

        self.min_distance = min_distance

        umap_kwargs = {"random_state": random_state} | umap_kwargs
        cumap_kwargs = {"random_state": random_state} | cumap_kwargs

        if "n_jobs" not in umap_kwargs:
            # umap_kwargs["random_state"] is not necessarily == random_state
            n_jobs = n_workers if umap_kwargs.get("random_state") is None else 1
            umap_kwargs["n_jobs"] = n_jobs
        if "n_jobs" not in cumap_kwargs:
            n_jobs = n_workers if cumap_kwargs.get("random_state") is None else 1
            cumap_kwargs["n_jobs"] = n_jobs

        self.pca = PCA(n_components=n_components, random_state=random_state)
        self.umap_2d = UMAP(**(umap_kwargs | {"n_components": 2}))
        self.pca_rgb = PCA(n_components=3, random_state=random_state)
        self.umap_rgb = UMAP(**(cumap_kwargs | {"n_components": 3}))

    def process_coordinates(self, gridsize: float = 1, **kwargs):
        """
        Process the coordinates of the transcripts dataframe.

        Parameters
        ----------
        gridsize : float, optional
            The size of each pixel in the grid.
            Measured in units of the transcript locations (usually µm).
        kwargs
            Other keyword arguments are passed to :py:func:`ovrlpy.process_coordinates`
        """
        self.gridsize = gridsize
        transcripts, origin = process_coordinates(
            self.transcripts, gridsize=gridsize, return_shift=True, **kwargs
        )
        assert isinstance(transcripts, pl.DataFrame)
        assert isinstance(origin, tuple)
        self.transcripts = transcripts
        self.origin = origin

    def _expression_threshold(self, n: float = 10, scale: float = 1.1) -> float:
        return n * scale / (2 * np.pi * self.KDE_bandwidth**2)

    def fit_transcripts(
        self,
        /,
        min_transcripts: float = 10,
        *,
        genes: Iterable[str] | None = None,
        fit_umap: bool = True,
    ):
        """
        Fits a spatial transcripts dataset using the SSAM algorithm.

        Parameters
        ----------
        min_transcripts : float
            Minimum expression for a local maximum to be considered. Expressed in terms
            of transcripts.
        genes : collections.abc.Iterable[str] | None
            A list of genes to utilize in the model. `None` uses all genes.
            Local maxima are always detected based on all genes.
        fit_umap : bool
            Whether to fit the UMAP to the data.
        """

        if genes is None:
            self.genes = sorted(self.transcripts["gene"].unique())
        else:
            self.genes = list(genes)

        local_maxima = _sample_expression(
            self.transcripts,
            min_expression=self._expression_threshold(min_transcripts),
            kde_bandwidth=self.KDE_bandwidth,
            genes=self.genes,
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
        self.pca.fit(X)

        if fit_umap:
            factors = self.pca.transform(X)

            print(f"Modeling {factors.shape[1]} pseudo-celltype clusters;")

            self.pseudocells.obsm["2D_UMAP"] = self.umap_2d.fit_transform(factors)

            embedding_color = self.umap_rgb.fit_transform(
                factors / norm(factors, axis=1, keepdims=True)
            )
            embedding_color = _fill_color_axes(embedding_color, self.pca_rgb, fit=True)

            self._colors_min_max = (
                embedding_color.min(axis=0),
                embedding_color.max(axis=0),
            )

            self.pseudocells.obsm["RGB"] = _minmax_scaling(embedding_color)

    @staticmethod
    def _determine_celltype(samples: AnnData, signatures: pl.DataFrame) -> np.ndarray:
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

    @overload
    def fit_signatures(self, signatures: pl.DataFrame, key: str): ...
    @overload
    def fit_signatures(self, signatures: pd.DataFrame, key: None = None): ...

    def fit_signatures(
        self, signatures: pl.DataFrame | pd.DataFrame, key: None | str = None
    ):
        """
        Fits a signature matrix.

        Parameters
        ----------
        signatures : polars.DataFrame | pandas.DataFrame
            A matrix of celltypes x gene signatures to use to annotate the UMAP.
        key : str | None
            Name of the column with name of the signature.
            Only used if `signatures` is a :py:class:`polars.DataFrame`,
            for :py:class:`pandas.DataFrame` the names are expected as index.
        """

        if self.pseudocells is None:
            raise Exception("`fit_pseudocells` must be run before fitting signatures")
        if "2D_UMAP" not in self.pseudocells.obsm:
            raise Exception("Signature can only be fitted if a UMAP has been fit")

        if isinstance(signatures, pl.DataFrame):
            assert key is not None
            # ensure annotation as first column
            signatures = signatures.select(key, pl.exclude(key))
        else:
            if signatures.index.name is None:
                key = "index"
            else:
                key = str(signatures.index.name)
            signatures = pl.from_pandas(signatures.reset_index())

        self.signatures = signatures

        self.celltype_assignments = self._determine_celltype(
            self.pseudocells, signatures.drop(key)
        )

        # determine the center of gravity of each celltype in the embedding:
        self.celltype_centers = self._coordinate_center(
            self.pseudocells.obsm["2D_UMAP"],
            self.celltype_assignments,
            signatures.shape[0],
        )

    def compute_VSI(self, *, min_transcripts: float = 2):
        """
        Calculate the vertical signal integrity (VSI).

        Parameters
        ----------
        min_transcripts : float | None, optional
            Minimum expression value to consider in calculation.
            Defaults to the 110% of the maximum expression profile of two molecules in the KDE.
        """

        min_expression = self._expression_threshold(min_transcripts)

        padding = int(ceil(_TRUNCATE * self.KDE_bandwidth))

        gene2idx = {gene: i for i, gene in enumerate(self.genes)}

        signal = kde_2d_discrete(
            self.transcripts["x_pixel"].to_numpy(),
            self.transcripts["y_pixel"].to_numpy(),
            bandwidth=self.KDE_bandwidth,
            dtype=self.dtype,
        )
        shape = signal.shape

        cosine_similarity = np.zeros_like(signal)

        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            for patch_df, padded, unpadded in tqdm.tqdm(
                _patches(
                    self.transcripts[["gene", "x_pixel", "y_pixel", "z", "z_center"]],
                    self.patch_length,
                    padding,
                    size=shape,
                    coordinates=("x_pixel", "y_pixel"),
                ),
                total=n_patches(self.patch_length, shape),
            ):
                assert isinstance(patch_df, pl.DataFrame)

                if len(patch_df) == 0:
                    continue

                # remove padding
                left_pad = unpadded[0].start - padded[0].start
                bottom_pad = unpadded[1].start - padded[1].start
                x_size = unpadded[0].stop - unpadded[0].start
                y_size = unpadded[1].stop - unpadded[1].start
                remove_pad = (
                    slice(left_pad, left_pad + x_size),
                    slice(bottom_pad, bottom_pad + y_size),
                )

                patch_signal: KDE_2D = signal[padded]

                not_padding = np.zeros(patch_signal.shape, dtype=bool)
                not_padding[remove_pad] = True

                patch_mask = (patch_signal > min_expression) & not_padding
                n_pixels = patch_mask.sum()

                if n_pixels == 0:
                    continue

                patch_df = patch_df.filter(
                    pl.col("gene").cast(pl.String).is_in(gene2idx)
                )
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
                                self.pca.components_,
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
                cosine_similarity[unpadded] = patch_cosine_similarity[remove_pad]

        self.signal_map = signal.T
        self.integrity_map = cosine_similarity.T

    def analyse(
        self,
        gridsize: float = 1,
        min_transcripts: float = 10,
        genes: Iterable[str] | None = None,
        fit_umap: bool = True,
    ):
        """
        Run main ovrlpy analysis.

        Parameters
        ----------
        gridsize : float, optional
            The size of the pixel grid.
        min_transcripts : float
            Minimum expression for a local maximum to be considered. Expressed in terms
            of transcripts.
        genes : collections.abc.Iterable[str] | None
            A list of genes to utilize in the model. `None` uses all genes.
            Local maxima are always detected based on all genes.
        fit_umap : bool
            Whether to fit the UMAP to the data.
        """

        print("Running vertical adjustment")
        self.process_coordinates(gridsize=gridsize, n_iter=20)

        print("Creating gene expression embeddings for visualization")
        self.fit_transcripts(min_transcripts, genes=genes, fit_umap=fit_umap)

        print("Creating signal integrity map")
        self.compute_VSI()

    def detect_doublets(
        self,
        min_distance: int = 10,
        min_integrity: float = 0.7,
        min_signal: float = 3,
        integrity_sigma: float | None = None,
    ) -> pl.DataFrame:
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

        Returns
        -------
        polars.DataFrame
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

        doublets = pl.DataFrame(
            {
                "x": dist_y,
                "y": dist_x,
                "integrity": 1 - dist_t,
                "signal": self.signal_map[dist_x, dist_y],
            }
        ).sort("integrity")

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

        Returns
        -------
        polars.DataFrame
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
        transcripts : polars.DataFrame | pandas.DataFrame
            Dataframe of transcript coordinates to transform.
        gene_key : str
            Name of the gene column.
        coordinate_keys : collections.abc.Sequence[str]
            Names of the coordinate columns.

        Returns
        -------
        embedding : numpy.ndarray
            2D UMAP embedding
        rgb : numpy.ndarray
            3D RGB UMAP embedding
        """

        neighbors = _gaussian_weighted_neighbors(
            transcripts[list(coordinate_keys)],
            self.KDE_bandwidth * _TRUNCATE,
            self.KDE_bandwidth,
            n_workers=self.n_workers,
        )

        expression = _weighted_average_expression(
            pl.Series(transcripts[gene_key]), neighbors, self.genes
        )
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

        Returns
        -------
        embedding : numpy.ndarray
            2D UMAP embedding
        rgb : numpy.ndarray
            3D RGB UMAP embedding
        """

        embedding, embedding_color = _transform_embeddings(
            pseudocells.to_numpy(),
            self.pca,
            umap_2d=self.umap_2d,
            umap_rgb=self.umap_rgb,
        )
        embedding_color = _fill_color_axes(embedding_color, self.pca_rgb)
        color_min, color_max = self._colors_min_max
        embedding_color = (embedding_color - color_min) / (color_max - color_min)
        embedding_color = np.clip(embedding_color, 0, 1)

        return embedding, embedding_color

    def pseudocell_integrity(self) -> pl.DataFrame:
        """
        Returns a DataFrame containing the gene-count matrix of the fitted
        tissue's determined pseudo-cells.

        Returns
        -------
        polars.DataFrame
        """
        assert self.pseudocells is not None
        pseudocells = pl.DataFrame(
            self.pseudocells.obsm["spatial"][:, :2], schema=["x", "y"]
        )

        if self.signal_map is not None:
            pseudocells = pseudocells.with_columns(
                signal=self.signal_map[pseudocells["y"], pseudocells["x"]]
            )

        if self.integrity_map is not None:
            pseudocells = pseudocells.with_columns(
                integrity=self.integrity_map[pseudocells["y"], pseudocells["x"]]
            )

        return pseudocells
