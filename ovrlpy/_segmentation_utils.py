from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from ._ovrlp import Ovrlp

if TYPE_CHECKING:
    from shapely.geometry.base import BaseGeometry


def _values_from_pixels(ovrlp: Ovrlp, cell_pixels: pl.DataFrame) -> pl.DataFrame:
    coord = (cell_pixels["y_pixel"], cell_pixels["x_pixel"])

    return cell_pixels.with_columns(
        signal=ovrlp.signal_map[coord],
        vsi=ovrlp.integrity_map[coord],
    )


def cell_integrity_from_transcripts(
    ovrlp: Ovrlp, *, cell_id: str = "cell_id", unassigned=-1
) -> pl.DataFrame:
    """
    Collect VSI per cell.

    Get the signal strength and VSI values for each cell using the assignment of transcripts to cells.
    This requires a column in the :py:attr:`ovrlpy.Ovrlp.transcripts` DataFrame that maps
    transcripts to cell identifiers.

    Parameters
    ----------
    ovrlp : ovrlpy.Ovrlp
    cell_id : str, optional
        Name of the column in the `ovrlp.transcripts` DataFame that stores the cell identifier.
    unassigned
        Value in the cell identifier column that indicates that a transcript is not
        assigned to any cell.

    Returns
    -------
    polars.DataFrame
        DataFrame containing all pixels and their corresponding signal strength and VSI
        values per cell.
    """
    cell_pixels = (
        ovrlp.transcripts.lazy()
        .filter(pl.col(cell_id) != unassigned)
        .select([pl.col(cell_id), "x_pixel", "y_pixel"])
        .unique()
        .collect()
    )

    return _values_from_pixels(ovrlp, cell_pixels)


def _close_pairs(
    geometries: Sequence[BaseGeometry], px_size: float
) -> np.ndarray[tuple[int, int], np.dtype[np.int64]]:
    from shapely.strtree import STRtree

    dist = math.sqrt(2 * px_size**2)  # length of the diagonal of a pixel

    tree = STRtree(geometries)
    close_pairs = tree.query(geometries, predicate="dwithin", distance=dist)
    # we only need one of the pairs i,j and j,i
    return close_pairs[:, close_pairs[0] < close_pairs[1]].T


def _disjoint_groups(
    geometries: Sequence[BaseGeometry], *, px_size: float
) -> list[list[int]]:
    import networkx as nx

    g = nx.Graph()
    g.add_nodes_from(range(len(geometries)))
    g.add_edges_from(_close_pairs(geometries, px_size=px_size))

    coloring = nx.coloring.greedy_color(g)
    n_colors = max(coloring.values()) + 1

    groups: dict[int, list[int]] = {c: [] for c in range(n_colors)}
    for i, c in coloring.items():
        groups[c].append(i)

    return list(groups.values())


def _segmentation_to_pixel(
    geometries: Sequence[tuple[BaseGeometry, int]], *, transform, shape: tuple[int, int]
) -> pl.DataFrame:
    from rasterio.features import rasterize

    assert all(i >= 0 for _, i in geometries)

    segmentation_map = rasterize(
        geometries,
        out_shape=shape,
        fill=-1,
        transform=transform,
        all_touched=True,
        dtype=int,
    )

    y, x = np.where(segmentation_map >= 0)

    return pl.DataFrame({"cell_id": segmentation_map[y, x], "x_pixel": x, "y_pixel": y})


def cell_integrity_from_masks(
    ovrlp: Ovrlp, segmentation_masks: Sequence[BaseGeometry], cell_ids: Sequence
) -> pl.DataFrame:
    """
    Collect VSI per cell using segmentation masks.

    Get the signal strength and VSI values for each cell using their segmentation masks.
    The segmentation masks will be rasterized and all pixels corresponding to a cell mapped to
    extract VSI and signal strength values.
    This requires the :py:attr:`ovrlpy.Ovrlp.origin` and :py:attr:`ovrlpy.Ovrlp.gridsize`
    to be set (which is done automatically if processing the transcripts with
    :py:func:`ovrlpy.Ovrlp.process_coordinates`).

    Parameters
    ----------
    ovrlp : ovrlpy.Ovrlp
    segmentation_masks : collections.abc.Sequence[shapely.geometry.base.BaseGeometry]
        List of segmentation masks as `shapely` Geometries (e.g., :py:class:`shapely.Polygon`).
    cell_ids : collections.abc.Sequence
        List of the cell IDs or labels. Must be in the same order as the segmentation masks.

    Returns
    -------
    polars.DataFrame
        DataFrame containing all pixels and their corresponding signal strength and VSI
        values per cell.
    """

    from rasterio.transform import from_origin
    from shapely import prepare

    for mask in segmentation_masks:
        prepare(mask)

    px_size = ovrlp.gridsize

    # get groups where within each group no pair of cells is too close to each other
    groups = _disjoint_groups(segmentation_masks, px_size=px_size)

    shape = ovrlp.integrity_map.shape
    transform = from_origin(ovrlp.origin[0], ovrlp.origin[1], px_size, -px_size)

    # get pixels that belong to each cell
    cell_pixels = pl.concat(
        [
            _segmentation_to_pixel(
                [(segmentation_masks[i], i) for i in g],
                transform=transform,
                shape=shape,
            )
            for g in groups
        ],
        how="vertical",
    )

    # add cell-label
    index_to_label = dict(enumerate(cell_ids))
    dtype = pl.Series(cell_ids).dtype

    cell_pixels = cell_pixels.with_columns(
        pl.col("cell_id").replace_strict(index_to_label, return_dtype=dtype)
    )

    return _values_from_pixels(ovrlp, cell_pixels)
