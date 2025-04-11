from concurrent.futures import ThreadPoolExecutor, as_completed
from math import floor
from typing import Iterable, TypeVar

import numpy as np
import pandas as pd
import tqdm
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max

from ._patching import _patches, n_patches

_TRUNCATE = 4

N = TypeVar("N", bound=int)
Shape1D = tuple[N]

Shape2DAny = tuple[int, int]
Shape3DAny = tuple[int, int, int]

T = TypeVar("T", bound=np.dtype)

Array1D_T = np.ndarray[Shape1D, T]


def kde_2d(x: Array1D_T, y: Array1D_T, **kwargs) -> np.ndarray[Shape2DAny, np.dtype]:
    """
    Calculate the 2D KDE using the first 2 columns of coordinates.
    """
    return _kde_nd(x, y, **kwargs)


def kde_3d(
    x: Array1D_T, y: Array1D_T, z: Array1D_T, **kwargs
) -> np.ndarray[Shape3DAny, np.dtype]:
    """
    Calculate the 3D KDE using the first 3 columns of coordinates.
    """
    return _kde_nd(x, y, z, **kwargs)


def _kde_nd(
    *coordinates: Array1D_T,
    size: tuple[int, ...] | None = None,
    bandwidth: float = 1.5,
    truncate: float = _TRUNCATE,
    dtype=np.float32,
) -> np.ndarray:
    """
    Calculate the KDE using the coordinates.
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

    output = np.zeros(size, dtype=dtype)

    dim_bins = list()
    for c in coordinates:
        c_min = int(c.min())
        # the last interval of np.histogram is closed (while the rest is half-open)
        # therefore we add an additional bin if the max is an int
        c_max = int(floor(c.max() + 1))
        dim_bins.append(np.linspace(c_min, c_max, c_max - c_min + 1))

    histogram, bins = np.histogramdd(coordinates, bins=dim_bins)

    kde = gaussian_filter(
        histogram, sigma=bandwidth, truncate=truncate, mode="constant", output=dtype
    )

    output[
        tuple(slice(int(bins[i].min()), int(bins[i].max())) for i in range(len(bins)))
    ] = kde

    return output


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
    *coordinates: Array1D_T, sampling_coordinates: np.ndarray, gene: str, **kwargs
) -> tuple[str, np.ndarray]:
    """
    Create a kde of the data and sample at 'sampling_coordinates'.
    """

    sampling_coordinates = np.rint(sampling_coordinates, dtype=int)
    n_dims = sampling_coordinates.shape[1]

    kde = _kde_nd(*coordinates, **kwargs)

    return gene, kde[tuple(sampling_coordinates[:, i] for i in range(n_dims))]


def _sample_expression(
    transcripts: pd.DataFrame,
    kde_bandwidth: float = 2.5,
    min_expression: float = 2,
    min_pixel_distance: float = 5,
    coord_columns: Iterable[str] = ["x", "y", "z"],
    gene_column: str = "gene",
    n_workers: int = 8,
    patch_length: int = 500,
    dtype=np.float32,
) -> tuple[pd.DataFrame, np.ndarray]:
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
        dtype
            Datatype for the KDE.

    Returns
    -------
        pandas.DataFrame: Gene expression KDE of local maxima.
        numpy.ndarray: Coordinates of local maxima
    """

    coord_columns = list(coord_columns)
    assert len(coord_columns) == 3 or len(coord_columns) == 2
    transcripts = transcripts[coord_columns + [gene_column]].copy()

    # lower resolution instead of increasing bandwidth!
    transcripts[coord_columns] /= kde_bandwidth

    print("determining pseudocells")

    # perform a global KDE to determine local maxima:
    kde = _kde_nd(
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
        for patch_df, offset, patch_size in tqdm.tqdm(
            _patches(transcripts, patch_length, padding, size=size),
            total=n_patches(patch_length, size),
        ):
            assert isinstance(patch_df, pd.DataFrame)
            patch_maxima = local_maximum_coordinates[
                (local_maximum_coordinates[:, 0] >= offset[0])
                & (local_maximum_coordinates[:, 0] < offset[0] + patch_size[0])
                & (local_maximum_coordinates[:, 1] >= offset[1])
                & (local_maximum_coordinates[:, 1] < offset[1] + patch_size[1]),
                :,
            ]
            coords.append(patch_maxima)

            # we need to shift the maximum coordinates so they are in the correct
            # relative position of the patch
            maxima = patch_maxima.copy()
            maxima[:, 0] -= offset[0]
            maxima[:, 1] -= offset[1]

            # patch_size is 2D, make 3D if KDE is calculated as 3D
            patch_size = (
                patch_size[0] + 2 * padding,
                patch_size[1] + 2 * padding,
                *size[2:],
            )

            futures = set(
                executor.submit(
                    kde_and_sample,
                    *(df[c].to_numpy() for c in coord_columns),
                    sampling_coordinates=maxima,
                    gene=gene,
                    size=patch_size,
                    bandwidth=1,
                    dtype=dtype,
                )
                for gene, df in patch_df.groupby(gene_column, observed=True)
            )

            patches.append(
                pd.DataFrame(dict(f.result() for f in as_completed(futures)))
            )
            del futures

    gene_list = sorted(transcripts[gene_column].unique())
    locations = np.vstack(coords) * kde_bandwidth
    expression = pd.concat(patches).reset_index(drop=True)[gene_list].fillna(0)

    return expression, locations
