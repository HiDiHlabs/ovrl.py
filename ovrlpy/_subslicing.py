import warnings
from collections.abc import Callable, Sequence
from functools import reduce
from operator import add
from typing import Literal, TypeVar, get_args, overload

import numpy as np
import polars as pl
from polars.datatypes import DataType, Float32, Int32
from scipy.sparse import coo_array


def _assign_xy(
    df: pl.DataFrame,
    xy_columns: Sequence[str] = ("x", "y"),
    gridsize: float = 1,
    shift: bool = True,
) -> tuple[pl.DataFrame, tuple[float, float]]:
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
    shift : bool
        Whether to shift the x,y coordinates to 0 to minimize the analysis area.
        If False, the sample will keep its original shape.
    """
    lf = df.lazy()
    min_coord = (0, 0)
    if shift:
        min_coord = df[xy_columns].min().row(0)
        lf = lf.with_columns(
            pl.col(xy_columns[0]) - min_coord[0],
            pl.col(xy_columns[1]) - min_coord[1],
        )
    elif df.select(pl.col(xy_columns).min() < 0).to_numpy().any():
        raise ValueError("shift=False is only possible if coordinates are > 0")

    df = (
        lf.with_columns(pl.col(xy_columns) / gridsize)
        .with_columns(
            pl.col(xy_columns[0]).cast(Int32).alias("x_pixel"),
            pl.col(xy_columns[1]).cast(Int32).alias("y_pixel"),
        )
        .collect(engine="streaming")  # reduce memory usage by using streaming
    )
    df = df.with_columns(df[xy].shrink_dtype() for xy in xy_columns)

    return df, min_coord


Shape = TypeVar("Shape", bound=tuple[int, ...])


def _message_passing(
    x: np.ndarray[Shape, np.dtype], /, n_iter: int
) -> np.ndarray[Shape, np.dtype]:
    with warnings.catch_warnings():
        # ignore 'mean of empty slice' warning if all values are nan
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for _ in range(n_iter):
            x = reduce(
                add,
                (
                    np.nanmean([x, np.roll(x, shift, axis=axis)], axis=0)  # type: ignore
                    for axis in (0, 1)
                    for shift in (1, -1)
                ),
            )
            x /= 4
    return x


def _mean_elevation(
    df: pl.DataFrame, z_key: str, *, dtype: type[DataType]
) -> np.ndarray[tuple[int, int], np.dtype]:
    x = df["x_pixel"].to_numpy()
    y = df["y_pixel"].to_numpy()

    z = coo_array((df[z_key].cast(dtype).to_numpy(), (x, y))).toarray()
    n = coo_array((np.ones(len(df), dtype=np.uint16), (x, y))).toarray()

    with np.errstate(divide="ignore", invalid="ignore"):
        # ignore zero division warnings as we want to transform 0 to nan
        z /= n

    return z


def _assign_z_mean_message_passing(
    df: pl.DataFrame, n_iter: int, z_key: str = "z", *, dtype: type[DataType] = Float32
) -> pl.Series:
    elevation_map = _mean_elevation(df, z_key, dtype=dtype)
    elevation_map = _message_passing(elevation_map, n_iter)
    return pl.Series("z", elevation_map[df["x_pixel"], df["y_pixel"]])


def _transform(
    df: pl.DataFrame,
    f: Callable[[str], pl.Expr],
    z_key: str = "z",
    dtype: type[DataType] = Float32,
) -> pl.Series:
    # transforming over the on-the-fly created pixel-id is more efficient than
    # using the two pixel columns
    _pixel_id = pl.col("x_pixel") + pl.col("y_pixel") * (pl.col("x_pixel").max() + 1)
    z = df.lazy().select(f(z_key).over(_pixel_id).cast(dtype)).collect().to_series()
    return z


_METHODS = Literal["mean", "median", "message_passing"]


@overload
def process_coordinates(
    coordinates: pl.DataFrame,
    /,
    gridsize: float = 1,
    *,
    coordinate_keys: tuple[str, str, str] = ("x", "y", "z"),
    method: _METHODS = "message_passing",
    shift: bool = True,
    n_iter: int = 20,
    dtype: type[DataType] = Float32,
    return_shift: Literal[False],
) -> pl.DataFrame: ...
@overload
def process_coordinates(
    coordinates: pl.DataFrame,
    /,
    gridsize: float = 1,
    *,
    coordinate_keys: tuple[str, str, str] = ("x", "y", "z"),
    method: _METHODS = "message_passing",
    shift: bool = True,
    n_iter: int = 20,
    dtype: type[DataType] = Float32,
    return_shift: Literal[True],
) -> tuple[pl.DataFrame, tuple[float, float]]: ...
@overload
def process_coordinates(
    coordinates: pl.DataFrame,
    /,
    gridsize: float = 1,
    *,
    coordinate_keys: tuple[str, str, str] = ("x", "y", "z"),
    method: _METHODS = "message_passing",
    shift: bool = True,
    n_iter: int = 20,
    dtype: type[DataType] = Float32,
    return_shift: bool = False,
) -> pl.DataFrame | tuple[pl.DataFrame, tuple[float, float]]: ...


def process_coordinates(
    coordinates: pl.DataFrame,
    /,
    gridsize: float = 1,
    *,
    coordinate_keys: tuple[str, str, str] = ("x", "y", "z"),
    method: _METHODS = "message_passing",
    shift: bool = True,
    n_iter: int = 20,
    dtype: type[DataType] = Float32,
    return_shift: bool = False,  # for backwards compatibility
) -> pl.DataFrame | tuple[pl.DataFrame, tuple[float, float]]:
    """
    Runs the pre-processing routine of the coordinate dataframe.

    x,y coordinates are rescaled by the gridsize and x,y pixels assigned to all
    molecules. A z-coordinate-center for each pixel is calculated.

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
    shift : bool
        Whether to shift the x,y coordinates to 0 to minimize the analysis area.
        If False, the sample will keep its original shape.
    n_iter : int, optional
        Number of iterations. Only used if the method is message_passing.
    dtype : type[polars.DataType], optional
        Datatype of the z-coordinate center.
    return_shift : bool
        Return the amount the coordinates have been shifted.

    Returns
    -------
    polars.DataFrame:
        A dataframe with added x_pixel, y_pixel, and z_center column.
    """
    if method not in get_args(_METHODS):
        raise ValueError(f"`method` must be one of {get_args(_METHODS)} not '{method}'")

    *xy, z = coordinate_keys

    coordinates, shifted = _assign_xy(
        coordinates, xy_columns=xy, gridsize=gridsize, shift=shift
    )

    if method == "message_passing":
        z_center = _assign_z_mean_message_passing(
            coordinates, n_iter=n_iter, z_key=z, dtype=dtype
        )
    else:
        if method == "mean":
            f = pl.mean
        elif method == "median":
            f = pl.median

        z_center = _transform(coordinates, f, z, dtype=dtype)

    coordinates = coordinates.with_columns(pl.Series("z_center", z_center))

    if return_shift:
        return coordinates, shifted
    else:
        return coordinates
