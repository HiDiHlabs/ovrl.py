from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Collection, Iterable, Optional

import anndata
import numpy as np
import pandas as pd
import tqdm

from . import _utils


def _sample_expression(
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

    print("determining pseudocells:")
    bounds = [
        (
            int(np.min(coordinate_dataframe_[c])),
            int(np.ceil(np.max(coordinate_dataframe_[c]))),
        )
        for c in coord_columns
    ]

    # perform a global KDE to determine local maxima:
    vector_field_norm = _utils._kde_nd(
        coordinate_dataframe_[coord_columns].values, bandwidth=1
    )
    local_maximum_coordinates = _utils.find_local_maxima(
        vector_field_norm,
        min_pixel_distance=1 + int(min_pixel_distance / kde_bandwidth),
        min_expression=minimum_expression,
    )

    print("found", len(local_maximum_coordinates), "pseudocells")

    size = vector_field_norm.shape

    del vector_field_norm

    # store in anndata object:
    adata_ssam = anndata.AnnData(
        X=np.zeros((len(local_maximum_coordinates), len(gene_list))),
        var=pd.DataFrame(index=gene_list),
        obsm={"spatial": local_maximum_coordinates * kde_bandwidth},
    )

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

    gene_coord_dict = {
        gene: df[coord_columns].to_numpy()
        for gene, df in coordinate_dataframe_.groupby(gene_column, observed=True)
    }

    print("sampling expression:")
    with tqdm.tqdm(total=len(gene_coord_dict)) as pbar:
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            fs = {
                executor.submit(
                    _utils.kde_and_sample, coords, local_maximum_coordinates, size=size
                ): gene
                for gene, coords in gene_coord_dict.items()
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
