import warnings
from functools import reduce
from operator import add
from typing import Sequence

import numpy as np
import pandas as pd


def _assign_xy(
    df: pd.DataFrame, xy_columns: Sequence[str] = ("x", "y"), grid_size: float = 1
):
    """
    Assigns an x,y pixel coordinate.

    Parameters
    ----------
    df : pandas.DataFrame
        A dataframe of coordinates.
    xy_columns : list, optional
        The names of the columns containing the x,y-coordinates.
    grid_size : float, optional
        The size of the grid.
    """
    x, y = xy_columns
    df[x] -= df[x].min()
    df[y] -= df[y].min()

    df["x_pixel"] = (df[x] / grid_size).astype(int)
    df["y_pixel"] = (df[y] / grid_size).astype(int)


def _assign_z_mean_message_passing(
    df: pd.DataFrame, z_column: str = "z", rounds: int = 3
) -> np.ndarray:
    """
    Calculates a z-split coordinate.

    Parameters
    ----------
    df : pandas.DataFrame
        A dataframe of coordinates.
    z_column : str, optional
        The name of the column containing the z-coordinate.
    rounds : int, optional
        Number of rounds for the message passing algorithm.

    Returns
    -------
    numpy.ndarray
    """
    if "pixel_id" not in df.columns:
        ValueError(
            "Please assign x,y coordinates to the dataframe first by running assign_xy(df)"
        )

    pixels = df.groupby(["x_pixel", "y_pixel"]).agg({z_column: "mean"}).reset_index()

    elevation_map = np.full(
        (pixels["x_pixel"].max() + 1, pixels["y_pixel"].max() + 1), np.nan
    )
    elevation_map[pixels["x_pixel"], pixels["y_pixel"]] = pixels[z_column]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for _ in range(rounds):
            elevation_map = reduce(
                add,
                (
                    np.nanmean(
                        [elevation_map, np.roll(elevation_map, shift, axis=ax)], axis=0
                    )
                    for ax in (0, 1)
                    for shift in (1, -1)
                ),
            )
            elevation_map /= 4

    return elevation_map[df["x_pixel"], df["y_pixel"]]


def _assign_z_mean(df: pd.DataFrame, z_column: str = "z") -> np.ndarray:
    """
    Calculates a z-split coordinate.

    Parameters
    ----------
    df : pandas.DataFrame
        A dataframe of coordinates.
    z_column : str, optional
        The name of the column containing the z-coordinate.

    Returns
    -------
    numpy.ndarray
    """
    if "pixel_id" not in df.columns:
        ValueError(
            "Please assign x,y coordinates to the dataframe first by running assign_xy(df)"
        )
    return df.groupby(["x_pixel", "y_pixel"])[z_column].transform("mean").to_numpy()


def _assign_z_median(df: pd.DataFrame, z_column: str = "z") -> np.ndarray:
    """
    Calculates a z-split coordinate.

    Parameters
    ----------
    df : pandas.DataFrame
        A dataframe of coordinates.
    z_column : str, optional
        The name of the column containing the z-coordinate.

    Returns
    -------
    numpy.ndarray
    """
    if "pixel_id" not in df.columns:
        ValueError(
            "Please assign x,y coordinates to the dataframe first by running assign_xy(df)"
        )
    return df.groupby(["x_pixel", "y_pixel"])[z_column].transform("median").to_numpy()


def pre_process_coordinates(
    coordinates: pd.DataFrame,
    gridsize: float = 1,
    coordinate_keys: tuple[str, str, str] = ("x", "y", "z"),
    method: str = "message_passing",
    inplace: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """
    Runs the pre-processing routine of the coordinate dataframe.

    It assigns x,y coordinate pixels to all molecules in the data frame and
    determines a z-coordinate-center for each pixel.

    Parameters
    ----------
    coordinates : pandas.DataFrame
        A dataframe of coordinates.
    gridsize : float, optional
        The size of the pixel grid.
    coordinate_keys : tuple[str, str, str], optional
        Name of the coordinate columns.
    method : bool, optional
        The measure to use to determine the z-dimension delimiter.
    inplace : bool, optional
        Whether to modify the input dataframe or return a copy.

    Returns
    -------
    pandas.DataFrame:
        A dataframe with added x_pixel, y_pixel and z_delim columns.
    """
    *xy, z = coordinate_keys

    if not inplace:
        coordinates = coordinates.copy()

    _assign_xy(coordinates, xy_columns=xy, grid_size=gridsize)

    match method:
        case "message_passing":
            z_delim = _assign_z_mean_message_passing(coordinates, z_column=z, **kwargs)
        case "mean":
            z_delim = _assign_z_mean(coordinates, z_column=z)
        case "median":
            z_delim = _assign_z_median(coordinates, z_column=z)
        case _:
            raise ValueError(
                "`method` must be one of 'mean', 'median', or 'message_passing'"
            )
    coordinates["z_delim"] = z_delim
    return coordinates.drop(columns=["x_pixel", "y_pixel"])
