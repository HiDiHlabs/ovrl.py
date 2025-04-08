from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable

import numpy as np
import pandas as pd
import tqdm

from .._patching import _patches, n_patches
from . import _utils
from ._utils import _TRUNCATE


def _sample_expression(
    coordinates: pd.DataFrame,
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
    Sample expression from a coordinate dataframe.

    Parameters
    ----------
        coordinates : Optional[pandas.DataFrame]
            The input coordinate dataframe.
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
    coordinates = coordinates[coord_columns + [gene_column]].copy()

    # lower resolution instead of increasing bandwidth!
    coordinates[coord_columns] /= kde_bandwidth

    print("determining pseudocells")

    # perform a global KDE to determine local maxima:
    kde = _utils._kde_nd(
        coordinates[coord_columns].to_numpy(), bandwidth=1, dtype=dtype
    )

    min_dist = 1 + int(min_pixel_distance / kde_bandwidth)
    local_maximum_coordinates = _utils.find_local_maxima(
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
            _patches(coordinates, patch_length, padding, size=size),
            total=n_patches(patch_length, size),
        ):
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

            futures = {
                executor.submit(
                    _utils.kde_and_sample,
                    df[coord_columns].to_numpy(),
                    maxima,
                    size=patch_size,
                    bandwidth=1,
                    dtype=dtype,
                ): gene
                for gene, df in patch_df.groupby(gene_column, observed=True)
            }

            patches.append(
                pd.DataFrame({futures[f]: f.result() for f in as_completed(futures)})
            )
            del futures

    gene_list = sorted(coordinates[gene_column].unique())
    locations = np.vstack(coords) * kde_bandwidth
    expression = pd.concat(patches).reset_index(drop=True)[gene_list].fillna(0)

    return expression, locations
