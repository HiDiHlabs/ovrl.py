import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from math import floor
from typing import Iterable

import numpy as np
import pandas as pd
import polars as pl
import tqdm
from anndata import AnnData, ImplicitModificationWarning
from numpy.typing import DTypeLike
from polars.datatypes import Int32
from scipy.ndimage import gaussian_filter
from scipy.sparse import coo_array
from skimage.feature import peak_local_max

from ._patching import _patches, n_patches

_TRUNCATE = 4

Shape1D = tuple[int]
Shape2D = tuple[int, int]

Array1D = np.ndarray[Shape1D, np.dtype]
Array1D_Int = np.ndarray[Shape1D, np.dtype[np.integer]]


def kde_2d_discrete(
    x: Array1D_Int,
    y: Array1D_Int,
    bandwidth: float,
    size: tuple[int, int] | None = None,
    dtype: DTypeLike = np.float32,
    **kwargs,
) -> np.ndarray[Shape2D, np.dtype]:
    """
    Calculate the 2D KDE using discrete (i.e. integer) coordinates.
    """
    n = x.shape[0]
    if x.shape[0] != y.shape[0]:
        raise ValueError("All coordinates must have the same number of rows")

    if n == 0:
        if size is None:
            raise ValueError("If no coordinates are provided, size must be provided.")
        else:
            return np.zeros(size, dtype=dtype)

    if size is None:
        size = (x.max() + 1, y.max() + 1)

    min_x = x.min()
    min_y = y.min()
    if min_x != 0:
        x = x - min_x
    if min_y != 0:
        y = y - min_y

    counts = coo_array((np.ones(n, dtype=np.uint32), (x, y))).toarray()
    kde = _kde(counts, bandwidth, dtype=dtype, **kwargs)

    if kde.shape != size:
        output = np.zeros(size, dtype=dtype)
        output[min_x : kde.shape[0] + min_x, min_y : kde.shape[1] + min_y] = kde
        return output
    else:
        return kde


def kde_nd(
    *coordinates: Array1D,
    bandwidth: float,
    size: tuple[int, ...] | None = None,
    dtype: DTypeLike = np.float32,
    **kwargs,
) -> np.ndarray:
    """
    Calculate the KDE using the (continuous) coordinates.
    """
    assert len(coordinates) >= 1

    n = coordinates[0].shape[0]
    if not all(x.shape[0] == n for x in coordinates[1:]):
        raise ValueError("All coordinates must have the same number of rows")

    if n == 0:
        if size is None:
            raise ValueError("If no coordinates are provided, size must be provided.")
        else:
            return np.zeros(size, dtype=dtype)

    if size is None:
        size = tuple(int(floor(c.max() + 1)) for c in coordinates)

    dim_bins = [
        np.arange(int(c.min()), int(floor(c.max() + 1)) + 1) for c in coordinates
    ]
    counts, bins = np.histogramdd(coordinates, bins=dim_bins)
    kde = _kde(counts, bandwidth, dtype=dtype, **kwargs)

    if kde.shape != size:
        output = np.zeros(size, dtype=dtype)
        output[tuple(slice(b[0], b[-1]) for b in bins)] = kde
        return output
    else:
        return kde


def _kde(
    x: np.ndarray,
    bandwidth: float,
    truncate: float = _TRUNCATE,
    dtype: DTypeLike = np.float32,
) -> np.ndarray:
    kde = gaussian_filter(
        x, sigma=bandwidth, truncate=truncate, mode="constant", output=dtype
    )
    return kde


def find_local_maxima(
    x: np.ndarray, min_pixel_distance: int = 5, min_expression: float = 2
):
    local_maxima = peak_local_max(
        x,
        min_distance=min_pixel_distance,
        threshold_abs=min_expression,
        exclude_border=False,
    )

    return local_maxima


def kde_and_sample(
    *coordinates: Array1D, sampling_coordinates: np.ndarray, gene: object, **kwargs
) -> tuple[object, np.ndarray]:
    """
    Create a kde of the data and sample at 'sampling_coordinates'.
    """

    sampling_coordinates = np.rint(sampling_coordinates).astype(int)
    n_dims = sampling_coordinates.shape[1]

    kde = kde_nd(*coordinates, **kwargs)

    return gene, kde[tuple(sampling_coordinates[:, i] for i in range(n_dims))]


def _sample_expression(
    transcripts: pl.DataFrame,
    kde_bandwidth: float = 2.5,
    min_expression: float = 2,
    min_pixel_distance: float = 5,
    coord_columns: Iterable[str] = ["x", "y", "z"],
    gene_column: str = "gene",
    n_workers: int = 8,
    patch_length: int = 500,
    dtype: DTypeLike = np.float32,
) -> AnnData:
    """
    Sample expression from a transcripts dataframe.

    Parameters
    ----------
        transcripts : pandas.DataFrame
            The input transcripts dataframe.
        kde_bandwidth : float
            Bandwidth for kernel density estimation.
        minimum_expression : int
            Minimum expression value for local maxima determination.
        min_pixel_distance : int
            Minimum pixel distance for local maxima determination.
        coord_columns : Iterable[str], optional
            Name of the coordinate columns in the coordinate dataframe.
        gene_column : str, optional
            Name of the gene column in the coordinate dataframe.
        n_workers : int, optional
            Number of parallel workers for sampling.
        patch_length : int
            Size of the length in each dimension when calculating signal integrity in patches.
            Smaller values will use less memory, but may take longer to compute.
        dtype : numpy.typing.DTypeLike
            Datatype for the KDE.

    Returns
    -------
        anndata.AnnData
    """

    coord_columns = list(coord_columns)
    assert len(coord_columns) == 3 or len(coord_columns) == 2

    # lower resolution instead of increasing bandwidth!
    transcripts = transcripts.select(coord_columns + [gene_column]).with_columns(
        (pl.col(c) / kde_bandwidth).cast(Int32) for c in coord_columns
    )

    print("determining pseudocells")

    # perform a global KDE to determine local maxima:
    kde = kde_nd(
        *(transcripts[c].to_numpy() for c in coord_columns), bandwidth=1, dtype=dtype
    )

    min_dist = 1 + int(min_pixel_distance / kde_bandwidth)
    local_maximum_coordinates = find_local_maxima(
        kde, min_pixel_distance=min_dist, min_expression=min_expression
    )

    print("found", len(local_maximum_coordinates), "pseudocells")

    size = kde.shape
    del kde

    # truncate * bandwidth -> _TRUNCATE * 1
    padding = _TRUNCATE

    print("sampling expression:")
    patches = []
    coords = []
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        for patch_df, padded, unpadded in tqdm.tqdm(
            _patches(transcripts, patch_length, padding, size=size),
            total=n_patches(patch_length, size),
        ):
            assert isinstance(patch_df, pl.DataFrame)
            patch_maxima = local_maximum_coordinates[
                (local_maximum_coordinates[:, 0] >= unpadded[0].start)
                & (local_maximum_coordinates[:, 0] < unpadded[0].stop)
                & (local_maximum_coordinates[:, 1] >= unpadded[1].start)
                & (local_maximum_coordinates[:, 1] < unpadded[1].stop),
                :,
            ]
            coords.append(patch_maxima)

            # we need to shift the maximum coordinates so they are in the correct
            # relative position of the patch
            maxima = patch_maxima.copy()
            maxima[:, 0] -= padded[0].start
            maxima[:, 1] -= padded[1].start

            # patch_size is 2D, make 3D if KDE is calculated as 3D
            patch_size = (
                padded[0].stop - padded[0].start,
                padded[1].stop - padded[1].start,
                *size[2:],
            )

            futures = set(
                executor.submit(
                    kde_and_sample,
                    *(df[c].to_numpy() for c in coord_columns),
                    sampling_coordinates=maxima,
                    gene=gene[0],
                    size=patch_size,
                    bandwidth=1,
                    dtype=dtype,
                )
                for gene, df in patch_df.group_by(gene_column)
            )

            # TODO: improve
            patches.append(
                pd.DataFrame(dict(f.result() for f in as_completed(futures)))
            )
            del futures

    gene_list = sorted(transcripts[gene_column].unique())
    # TODO: sparse?
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ImplicitModificationWarning)
        adata = AnnData(pd.concat(patches).reset_index(drop=True)[gene_list].fillna(0))
    adata.obsm["spatial"] = np.rint(np.vstack(coords) * kde_bandwidth).astype(np.int32)
    return adata
