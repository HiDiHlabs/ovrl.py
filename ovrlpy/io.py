import os
from collections.abc import Collection, Mapping
from pathlib import Path

import pandas as pd


def _filter_genes(df: pd.DataFrame, remove_features: Collection[str]) -> pd.DataFrame:
    if len(remove_features) > 0:
        df = df.loc[~df["gene"].str.contains(f"{'|'.join(remove_features)}")]
        df = df.assign(
            gene=lambda df: df["gene"].astype("category").cat.remove_unused_categories()
        )
    return df


# 10x Xenium
_XENIUM_COLUMNS = {
    "feature_name": "gene",
    "x_location": "x",
    "y_location": "y",
    "z_location": "z",
}

XENIUM_CTRLS = [
    "^BLANK",
    "^DeprecatedCodeword",
    "^Intergenic",
    "^NegControl",
    "^UnassignedCodeword",
]
"""Patterns for Xenium controls"""


def read_Xenium(
    filepath: str | os.PathLike,
    *,
    min_qv: float | None = None,
    remove_features: Collection[str] = XENIUM_CTRLS,
) -> pd.DataFrame:
    """
    Read a Xenium transcripts file.

    Parameters
    ----------
    filepath : os.PathLike or str
        Path to the Xenium transcripts file. Both, .csv.gz and .parquet files, are supported.
    min_qv : float | None, optional
        Minimum Phred-scaled quality value (Q-Score) of a transcript to be included.
        If `None` no filtering is performed.
    remove_features : collections.abc.Collection[str], optional
        List of regex patterns to filter the 'feature_name' column,
        :py:attr:`ovrlpy.io.XENIUM_CTRLS` by default.

    Returns
    -------
    pandas.DataFrame
    """
    filepath = Path(filepath)
    columns = list(_XENIUM_COLUMNS.keys())

    if min_qv is not None:
        columns.append("qv")

    if filepath.suffix == ".parquet":
        transcripts = pd.read_parquet(filepath, columns=columns)
        transcripts["feature_name"] = transcripts["feature_name"].astype("category")

        # v2/v3 versions of the XOA files encode the feature_name column as string
        # while in v1 it is only designated as binary so we need to cast
        # that's why we check whether the data in the column is bytes (and not string)
        if isinstance(transcripts["feature_name"].cat.categories[0], bytes):
            decoded_cat = transcripts["feature_name"].cat.categories.str.decode("utf-8")
            transcripts["feature_name"] = transcripts[
                "feature_name"
            ].cat.rename_categories(decoded_cat)

        # TODO: 'is_gene' column exists for Xenium v3 which only has .parquet
        # can be used to filter

    else:
        transcripts = pd.read_csv(filepath, usecols=columns, dtype={"gene": "category"})

    if min_qv is not None:
        transcripts = transcripts.loc[transcripts["qv"] >= min_qv].drop(columns="qv")

    transcripts = transcripts.rename(columns=_XENIUM_COLUMNS)
    transcripts = _filter_genes(transcripts, remove_features)

    return transcripts


# Vizgen MERFISH
_MERFISH_COLUMNS = {"gene": "gene", "global_x": "x", "global_y": "y", "global_z": "z"}

MERFISH_CTRLS = ["^Blank"]
"""Patterns for Vizgen controls"""


def read_MERFISH(
    filepath: str | os.PathLike,
    z_scale: float = 1.5,
    *,
    remove_genes: Collection[str] = MERFISH_CTRLS,
) -> pd.DataFrame:
    """
    Read a Vizgen transcripts file.

    Parameters
    ----------
    filepath : os.PathLike or str
        Path to the Vizgen transcripts file.
    z_scale : float
        Factor to scale z-plane index to um, i.e. distance between z-planes.
    remove_genes : collections.abc.Collection[str], optional
        List of regex patterns to filter the 'gene' column,
        :py:attr:`ovrlpy.io.MERFISH_CTRLS` by default.

    Returns
    -------
    pandas.DataFrame
    """

    transcripts = pd.read_csv(
        Path(filepath), usecols=_MERFISH_COLUMNS.keys(), dtype={"gene": "category"}
    ).rename(columns=_MERFISH_COLUMNS)

    transcripts = _filter_genes(transcripts, remove_genes)

    # convert plane to um
    transcripts["z"] *= z_scale

    return transcripts


# Nanostring CosMx
_COSMX_COLUMNS = {"target": "gene", "x_global_px": "x", "y_global_px": "y", "z": "z"}

COSMX_CTRLS = ["^NegPrb"]
"""Patterns for CosMx controls"""


def read_CosMx(
    filepath: str | os.PathLike,
    scale: Mapping[str, float] = {"xy": 0.12028, "z": 0.8},
    *,
    remove_targets: Collection[str] = COSMX_CTRLS,
) -> pd.DataFrame:
    """
    Read a Nanostring CosMx transcripts file.

    Parameters
    ----------
    filepath : os.PathLike or str
        Path to the CosMx transcripts file.
    scale : collections.abc.Mapping[str, float]
        The factors for scaling the coordinates from pixel space to um.
    remove_targets : collections.abc.Collection[str], optional
        List of regex patterns to filter the 'target' column,
        :py:attr:`ovrlpy.io.COSMX_CTRLS` by default.

    Returns
    -------
    pandas.DataFrame
    """

    transcripts = pd.read_csv(
        Path(filepath), usecols=_COSMX_COLUMNS.keys(), dtype={"target": "category"}
    ).rename(columns=_COSMX_COLUMNS)

    transcripts = _filter_genes(transcripts, remove_targets)

    # convert pixel to um
    transcripts[["x", "y"]] *= scale["xy"]
    transcripts["z"] *= scale["z"]

    return transcripts
