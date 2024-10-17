from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

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
    x: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    z: Optional[np.ndarray] = None,
    genes: Optional[np.ndarray] = None,
    x_column: str = "x",
    y_column: str = "y",
    z_column: str = "z",
    gene_column: str = "gene",
    n_workers: int = 8,
    mode: str = None,
) -> anndata.AnnData:
    """
    Sample expression from a coordinate dataframe.

    Parameters:
    - coordinate_dataframe (Optional[pd.DataFrame]): The input coordinate dataframe.
    - kde_bandwidth (float): Bandwidth for kernel density estimation.
    - minimum_expression (int): Minimum expression value for local maxima determination.
    - min_pixel_distance (int): Minimum pixel distance for local maxima determination.
    - x (Optional[np.ndarray]): Array of x-coordinates.
    - y (Optional[np.ndarray]): Array of y-coordinates.
    - z (Optional[np.ndarray]): Array of z-coordinates.
    - genes (Optional[np.ndarray]): Array of gene values.
    - x_column (str): Name of the x-coordinate column in the coordinate dataframe.
    - y_column (str): Name of the y-coordinate column in the coordinate dataframe.
    - z_column (str): Name of the z-coordinate column in the coordinate dataframe.
    - gene_column (str): Name of the gene column in the coordinate dataframe.
    - n_workers (int): Number of parallel workers for sampling.
    - mode (str): Sampling mode, either '2d' or '3d'.

    Returns:
    - anndata.AnnData: An Anndata object containing sampled expression values.
    """

    if mode is None:
        if coordinate_dataframe is not None:
            if z_column in coordinate_dataframe.columns:
                mode = "3d"
            else:
                mode = "2d"
        else:
            if z is not None:
                mode = "3d"
            else:
                mode = "2d"

    if mode == "2d":
        print("Analyzing in 2d mode:")
        return _sample_expression_2d(
            coordinate_dataframe=coordinate_dataframe,
            kde_bandwidth=kde_bandwidth,
            minimum_expression=minimum_expression,
            x=x,
            y=y,
            genes=genes,
            x_column=x_column,
            y_column=y_column,
            gene_column=gene_column,
            n_workers=n_workers,
        )
    elif mode == "3d":
        print("Analyzing in 3d mode:")
        return _sample_expression_3d(
            coordinate_dataframe=coordinate_dataframe,
            kde_bandwidth=kde_bandwidth,
            minimum_expression=minimum_expression,
            min_pixel_distance=min_pixel_distance,
            x=x,
            y=y,
            z=z,
            genes=genes,
            x_column=x_column,
            y_column=y_column,
            z_column=z_column,
            gene_column=gene_column,
            n_workers=n_workers,
        )
    else:
        raise ValueError(
            "Could not determine whether to use '2d' or '3d' analysis mode. Please specify mode='2d' or mode='3d'."
        )


def _sample_expression_2d(
    coordinate_dataframe=None,
    kde_bandwidth=2.5,
    minimum_expression=2,
    min_pixel_distance=5,
    x=None,
    y=None,
    genes=None,
    x_column="x",
    y_column="y",
    gene_column="gene",
    n_workers=8,
):
    """
    Sample expression from a coordinate dataframe.
    """

    gene_list = sorted(coordinate_dataframe[gene_column].unique())

    if coordinate_dataframe is None:
        if x is None or y is None or genes is None:
            raise ValueError(
                "Either coordinate_dataframe_ or x,y,genes must be provided."
            )
        else:
            coordinate_dataframe_ = pd.DataFrame(
                {
                    x_column: x,
                    y_column: y,
                    # z_column:z,
                    gene_column: genes,
                }
            )

    else:
        coordinate_dataframe_ = coordinate_dataframe.copy()

    # lower resolution instead of increasing bandwidth!
    coordinate_dataframe_[[x_column, y_column]] /= kde_bandwidth / 1.5

    print("determining local maxima:")
    x_, _x = (
        int(np.min(coordinate_dataframe_[x_column])),
        int(np.ceil(np.max(coordinate_dataframe_[x_column]))),
    )
    y_, _y = (
        int(np.min(coordinate_dataframe_[y_column])),
        int(np.ceil(np.max(coordinate_dataframe_[y_column]))),
    )

    # perform a global KDE to determine local maxima:
    vector_field_norm = utils.kde_2d(
        coordinate_dataframe_[[x_column, y_column]].values, bandwidth=1.5
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

    adata_ssam.uns["ssam"] = {
        "kde_bandwidth": kde_bandwidth,
        "x_": x_,
        "_x": _x,
        "y_": y_,
        "_y": _y,
        "size": size,
        "coordinates": coordinate_dataframe,
        "x_column": x_column,
        "y_column": y_column,
        "gene_column": gene_column,
    }
    adata_ssam.X = np.zeros((len(local_maximum_coordinates), len(gene_list)))

    df_gene_grouped = coordinate_dataframe_.groupby(gene_column, observed=False).apply(
        lambda x: x[[x_column, y_column]].values
    )

    df_gene_grouped = df_gene_grouped[df_gene_grouped.apply(lambda x: x.shape[0] > 0)]

    print("sampling expression:")

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

            try:
                output = f.result()
                adata_ssam.X[:, adata_ssam.var.index == gene] = output[:, None]

            except Exception as exc:
                print("%r generated an exception: %s" % (gene, exc))

    return adata_ssam


def _sample_expression_3d(
    coordinate_dataframe: Optional[pd.DataFrame] = None,
    kde_bandwidth: float = 2.5,
    minimum_expression: float = 2,
    min_pixel_distance: int = 5,
    x: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    z: Optional[np.ndarray] = None,
    genes: Optional[np.ndarray] = None,
    x_column: str = "x",
    y_column: str = "y",
    z_column: str = "z",
    gene_column: str = "gene",
    n_workers: int = 8,
) -> anndata.AnnData:
    """
    Sample expression from a coordinate dataframe.

    Parameters:
    - coordinate_dataframe (Optional[pd.DataFrame]): The input coordinate dataframe.
    - kde_bandwidth (float): Bandwidth for kernel density estimation.
    - minimum_expression (float): Minimum expression value for local maxima determination.
    - min_pixel_distance (int): Minimum pixel distance for local maxima determination.
    - x (Optional[np.ndarray]): Array of x-coordinates.
    - y (Optional[np.ndarray]): Array of y-coordinates.
    - z (Optional[np.ndarray]): Array of z-coordinates.
    - genes (Optional[np.ndarray]): Array of gene values.
    - x_column (str): Name of the x-coordinate column in the coordinate dataframe.
    - y_column (str): Name of the y-coordinate column in the coordinate dataframe.
    - z_column (str): Name of the z-coordinate column in the coordinate dataframe.
    - gene_column (str): Name of the gene column in the coordinate dataframe.

    Returns:
    - anndata.AnnData: An Anndata object containing sampled expression values.
    """

    gene_list = sorted(coordinate_dataframe[gene_column].unique())

    if coordinate_dataframe is None:
        if x is None or y is None or z is None or genes is None:
            raise ValueError(
                "Either coordinate_dataframe_ or x,y,z,genes must be provided."
            )
        else:
            coordinate_dataframe_ = pd.DataFrame(
                {x_column: x, y_column: y, z_column: z, gene_column: genes}
            )

    else:
        coordinate_dataframe_ = coordinate_dataframe.copy()

    # lower resolution instead of increasing bandwidth!
    coordinate_dataframe_[[x_column, y_column, z_column]] /= kde_bandwidth

    print("determining local maxima:")
    x_, _x = (
        int(np.min(coordinate_dataframe_[x_column])),
        int(np.ceil(np.max(coordinate_dataframe_[x_column]))),
    )
    y_, _y = (
        int(np.min(coordinate_dataframe_[y_column])),
        int(np.ceil(np.max(coordinate_dataframe_[y_column]))),
    )
    z_, _z = (
        int(np.min(coordinate_dataframe_[z_column])),
        int(np.ceil(np.max(coordinate_dataframe_[z_column]))),
    )

    print(x_, _x, y_, _y, z_, _z)
    # perform a global KDE to determine local maxima:
    vector_field_norm = utils.kde_3d(
        coordinate_dataframe_[[x_column, y_column, z_column]].values, bandwidth=1.1
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

    adata_ssam.uns["ssam"] = {
        "kde_bandwidth": kde_bandwidth,
        "x_": x_,
        "_x": _x,
        "y_": y_,
        "_y": _y,
        "z_": z_,
        "_z": _z,
        "size": size,
        "coordinates": coordinate_dataframe,
        "x_column": x_column,
        "y_column": y_column,
        "z_column": z_column,
        "gene_column": gene_column,
    }
    adata_ssam.X = np.zeros((len(local_maximum_coordinates), len(gene_list)))

    df_gene_grouped = coordinate_dataframe_.groupby(gene_column, observed=False).apply(
        lambda x: x[[x_column, y_column, z_column]].values
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
                    adata_ssam.X[:, adata_ssam.var.index == gene] = output[:, None]

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
