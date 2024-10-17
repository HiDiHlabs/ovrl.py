from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Collection, Iterable, Optional

import anndata
import numpy as np
import pandas as pd
import tqdm
from scipy.signal import fftconvolve

from . import utils


def sample_expression(
    coordinate_dataframe: Optional[pd.DataFrame] = None,
    kde_bandwidth: float = 2.5,
    minimum_expression: int = 2,
    min_pixel_distance: int = 5,
    coords: Optional[Collection[np.ndarray]] = None,
    genes: Optional[np.ndarray] = None,
    coord_columns: Iterable[str] = ["x", "y", "z"],
    gene_column: str = "gene",
    n_workers: int = 8,
    mode: Optional[str] = None,
) -> anndata.AnnData:
    """
    Sample expression from a coordinate dataframe.

    Parameters
    ----------
        coordinate_dataframe : Optional[pandas.DataFrame]
            The input coordinate dataframe.
        kde_bandwidth : float
            Bandwidth for kernel density estimation.
        minimum_expression : int
            Minimum expression value for local maxima determination.
        min_pixel_distance : int
            Minimum pixel distance for local maxima determination.
        coords : Collection[numpy.ndarray], optional
            A list of arrays holding the coordinates.
        genes : numpy.ndarray, optional
            Array of gene values.
        coord_columns : Iterable[str], optional
            Name of the coordinate columns in the coordinate dataframe.
        gene_column : str, optional
            Name of the gene column in the coordinate dataframe.
        n_workers : int, optional
            Number of parallel workers for sampling.
        mode : str, optional
            Sampling mode, either '2d' or '3d'.

    Returns
    -------
        anndata.AnnData: An Anndata object containing sampled expression values.
    """

    coord_columns = list(coord_columns)

    if mode is None:
        if coordinate_dataframe is not None:
            mode = "3d" if coord_columns[-1] in coordinate_dataframe.columns else "2d"
        elif coords is not None:
            mode = "3d" if len(coords) == 3 else "2d"
        else:
            raise ValueError(
                "Either `coordinate_dataframe` or `coords` and `genes` must be provided."
            )

    if mode == "2d":
        print("Analyzing in 2d mode:")
        coord_columns = coord_columns[:2]

    elif mode == "3d":
        print("Analyzing in 3d mode:")

    else:
        raise ValueError(
            "Could not determine whether to use '2d' or '3d' analysis mode. Please specify mode='2d' or mode='3d'."
        )

    return _sample_expression_nd(
        coordinate_dataframe=coordinate_dataframe,
        kde_bandwidth=kde_bandwidth,
        minimum_expression=minimum_expression,
        min_pixel_distance=min_pixel_distance,
        coords=coords,
        genes=genes,
        coord_columns=coord_columns,
        gene_column=gene_column,
        n_workers=n_workers,
    )


def _sample_expression_nd(
    coordinate_dataframe: Optional[pd.DataFrame] = None,
    kde_bandwidth: float = 2.5,
    minimum_expression: float = 2,
    min_pixel_distance: int = 5,
    coords: Optional[Iterable[np.ndarray]] = None,
    genes: Optional[np.ndarray] = None,
    coord_columns: Iterable[str] = ["x", "y", "z"],
    gene_column: str = "gene",
    n_workers: int = 8,
) -> anndata.AnnData:
    coord_columns = list(coord_columns)

    if coordinate_dataframe is None:
        if coords is None or genes is None:
            raise ValueError(
                "Either `coordinate_dataframe` or `coords` and `genes` must be provided."
            )
        else:
            coordinate_dataframe_ = pd.DataFrame(
                dict(zip(coord_columns, coords)) | {gene_column: genes}
            )
    else:
        coordinate_dataframe_ = coordinate_dataframe.copy()

    gene_list = sorted(coordinate_dataframe_[gene_column].unique())

    # lower resolution instead of increasing bandwidth!
    coordinate_dataframe_[coord_columns] /= kde_bandwidth

    print("determining local maxima:")
    bounds = [
        (
            int(np.min(coordinate_dataframe_[c])),
            int(np.ceil(np.max(coordinate_dataframe_[c]))),
        )
        for c in coord_columns
    ]

    print(bounds)
    # perform a global KDE to determine local maxima:
    vector_field_norm = utils._kde_nd(
        coordinate_dataframe_[coord_columns].values, bandwidth=1.1
    )
    local_maximum_coordinates = utils.find_local_maxima(
        vector_field_norm,
        min_pixel_distance=min_pixel_distance,
        min_expression=minimum_expression,
    )

    print("found", len(local_maximum_coordinates), "local maxima")

    size = vector_field_norm.shape

    del vector_field_norm

    # store in anndata object:
    adata_ssam = anndata.AnnData(
        var=pd.DataFrame(index=gene_list),
        obs=pd.DataFrame(index=range(len(local_maximum_coordinates))),
    )

    adata_ssam.obsm["spatial"] = local_maximum_coordinates * kde_bandwidth

    ssam_params = {
        "kde_bandwidth": kde_bandwidth,
        "size": size,
        "coordinates": coordinate_dataframe,
        "gene_column": gene_column,
    }
    ssam_params |= {
        k: v for k, v in zip(["x_column", "y_column", "z_column"], coord_columns)
    }
    for (min_i, max_i), name in zip(bounds, ["x", "y", "z"]):
        ssam_params |= {f"{name}_": min_i, f"_{name}": max_i}

    adata_ssam.uns["ssam"] = ssam_params

    adata_ssam.X = np.zeros((len(local_maximum_coordinates), len(gene_list)))

    df_gene_grouped = coordinate_dataframe_.groupby(gene_column, observed=False).apply(
        lambda x: x[coord_columns].values
    )

    df_gene_grouped = df_gene_grouped[df_gene_grouped.apply(lambda x: x.shape[0] > 0)]

    print("sampling expression:")
    with tqdm.tqdm(total=len(gene_list)) as pbar:
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            fs = {
                executor.submit(
                    utils.kde_and_sample,
                    df_gene_grouped.loc[gene],
                    local_maximum_coordinates,
                    size=size,
                ): gene
                for gene in gene_list
            }
            for f in as_completed(fs):
                gene = fs[f]
                pbar.set_description(gene)

                try:
                    output = f.result()
                    adata_ssam.X[:, adata_ssam.var_names == gene] = output[:, None]

                except Exception as exc:
                    print("%r generated an exception: %s" % (gene, exc))

                pbar.update(1)

    return adata_ssam


def count_cells_at_localmax(adata, step=1, radius=50):
    """
    Sweep a sphere window along a lattice on the image, and count the number of cell types in each window.

    :param step: The lattice spacing.
    :type step: int
    :param radius: The radius of the sphere window.
    :type radius: int
    """

    def make_sphere_mask(radius):
        dia = radius * 2 + 1
        X, Y, Z = np.ogrid[:dia, :dia, :dia]
        dist_from_center = np.sqrt(
            (X - radius) ** 2 + (Y - radius) ** 2 + (Z - radius) ** 2
        )
        mask = dist_from_center <= radius
        return mask

    mask = make_sphere_mask(radius)

    ct_map = adata.uns["spatial"]["ct_map_filtered"]

    celltype_count_at_localmax = pd.DataFrame(
        index=adata.obs_names, columns=np.arange(ct_map.max())
    )
    celltype_count_at_localmax.loc[:] = 0
    celltype_count_at_localmax = celltype_count_at_localmax.astype(int)

    for i, celltype in enumerate(np.arange(ct_map.max())):
        count_aggregated_map = fftconvolve(
            (ct_map == celltype).astype(int), mask, mode="same"
        )
        celltype_count_at_localmax.iloc[:, i] = (
            count_aggregated_map[
                (adata.obsm["spatial"][:, 0] / 2.5).astype(int),
                (adata.obsm["spatial"][:, 1] / 2.5).astype(int),
            ]
            .round()
            .astype(int)
        )

    return
