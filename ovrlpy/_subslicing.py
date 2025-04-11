import warnings
from functools import reduce
from operator import add
from typing import Literal, Sequence

import numpy as np
import polars as pl
from polars.datatypes import DataType, Float32, Int32


def _assign_xy(
    df: pl.DataFrame, xy_columns: Sequence[str] = ("x", "y"), gridsize: float = 1
) -> pl.DataFrame:
    """
    Assigns an x,y pixel coordinate.

    Parameters
    ----------
    df : polars.DataFrame
        A dataframe of coordinates.
    xy_columns : list, optional
        The names of the columns containing the x,y-coordinates.
    gridsize : float, optional
        The size of the grid.
    """
    df = df.with_columns(
        (pl.col(xy) - pl.col(xy).min() for xy in xy_columns)
    ).with_columns(
        (pl.col(xy_columns[0]) / gridsize).cast(Int32).alias("x_pixel"),
        (pl.col(xy_columns[1]) / gridsize).cast(Int32).alias("y_pixel"),
    )
    return df


def _message_passing(x: np.ndarray, /, n_iter: int) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for _ in range(n_iter):
            x = reduce(
                add,
                (
                    np.nanmean([x, np.roll(x, shift, axis=ax)], axis=0)
                    for ax in (0, 1)
                    for shift in (1, -1)
                ),
            )
            x /= 4
    return x


def _mean_elevation(
    df: pl.DataFrame, z_column: str, dtype: DataType
) -> np.ndarray[tuple[int, int], np.dtype]:
    pixels = df.group_by(["x_pixel", "y_pixel"]).agg(
        pl.col(z_column).mean().cast(dtype)
    )
    z = pixels.drop_in_place(z_column).to_numpy()

    elevation_map = np.full(
        (pixels["x_pixel"].max() + 1, pixels["y_pixel"].max() + 1),
        np.nan,
        dtype=z.dtype,
    )
    elevation_map[pixels["x_pixel"], pixels["y_pixel"]] = z
    return elevation_map


def _assign_z_mean_message_passing(
    df: pl.DataFrame,
    z_column: str = "z",
    n_iter: int = 3,
    *,
    dtype: DataType = Float32,
) -> pl.Series:
    """
    Calculates a z-split coordinate.

    Parameters
    ----------
    df : polars.DataFrame
        A dataframe of coordinates.
    z_column : str, optional
        The name of the column containing the z-coordinate.
    n_iter : int, optional
        Number of iterations for the message passing algorithm.

    Returns
    -------
    numpy.ndarray
    """
    elevation_map = _mean_elevation(df, z_column, dtype)
    elevation_map = _message_passing(elevation_map, n_iter)
    return pl.Series("z", elevation_map[df["x_pixel"], df["y_pixel"]])


def _assign_z_mean(
    df: pl.DataFrame, z_column: str = "z", dtype: DataType = Float32
) -> pl.Series:
    """
    Calculates a z-split coordinate.

    Parameters
    ----------
    df : polars.DataFrame
        A dataframe of coordinates.
    z_column : str, optional
        The name of the column containing the z-coordinate.

    Returns
    -------
    polars.Series
    """

    return df.select(
        pl.col(z_column).mean().over(["x_pixel", "y_pixel"]).cast(dtype)
    ).to_series()


def _assign_z_median(
    df: pl.DataFrame, z_column: str = "z", dtype: DataType = Float32
) -> pl.Series:
    """
    Calculates a z-split coordinate.

    Parameters
    ----------
    df : polars.DataFrame
        A dataframe of coordinates.
    z_column : str, optional
        The name of the column containing the z-coordinate.

    Returns
    -------
    polars.Series
    """

    return df.select(
        pl.col(z_column).median().over(["x_pixel", "y_pixel"]).cast(dtype)
    ).to_series()


def pre_process_coordinates(
    coordinates: pl.DataFrame,
    /,
    gridsize: float = 1,
    *,
    coordinate_keys: tuple[str, str, str] = ("x", "y", "z"),
    method: Literal["mean", "median", "message_passing"] = "message_passing",
    dtype: DataType = Float32,
    **kwargs,
) -> pl.DataFrame:
    """
    Runs the pre-processing routine of the coordinate dataframe.

    It assigns x,y coordinate pixels to all molecules in the data frame and
    determines a z-coordinate-center for each pixel.

    Parameters
    ----------
    coordinates : polars.DataFrame
        A dataframe of coordinates.
    gridsize : float, optional
        The size of the pixel grid.
    coordinate_keys : tuple[str, str, str], optional
        Name of the coordinate columns.
    method : bool, optional
        The measure to use to determine the z-dimension threshold for subslicing.
        One of, mean, median, and message_passing.
    dtype : numpy.typing.DTypeLike, optional
        Datatype of the z-coordinate center.
    kwargs
        Other keywords arguments are passed to the message_passing function.

    Returns
    -------
    polars.DataFrame:
        A dataframe with added z_delim column.
    """
    *xy, z = coordinate_keys

    coordinates = _assign_xy(coordinates, xy_columns=xy, gridsize=gridsize)

    match method:
        case "message_passing":
            z_delim = _assign_z_mean_message_passing(
                coordinates, z_column=z, dtype=dtype, **kwargs
            )
        case "mean":
            z_delim = _assign_z_mean(coordinates, z_column=z, dtype=dtype)
        case "median":
            z_delim = _assign_z_median(coordinates, z_column=z, dtype=dtype)
        case _:
            raise ValueError(
                "`method` must be one of 'mean', 'median', or 'message_passing'"
            )
    coordinates = coordinates.drop(["x_pixel", "y_pixel"]).with_columns(
        pl.Series("z_delim", z_delim)
    )
    return coordinates
