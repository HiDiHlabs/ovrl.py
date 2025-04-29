from collections.abc import Iterable, Mapping
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from matplotlib_scalebar.scalebar import ScaleBar

from ._ovrlp import Ovrlp

SCALEBAR_PARAMS: dict[str, Any] = {"dx": 1, "units": "um"}
"""Default scalebar parameters"""

BIH_CMAP = LinearSegmentedColormap.from_list(
    "BIH",
    [
        "#430541",
        "mediumvioletred",
        "violet",
        "powderblue",
        "powderblue",
        "white",
        "white",
    ][::-1],
)


def _plot_scalebar(ax: Axes, dx: float = 1, units="um", **kwargs):
    ax.add_artist(ScaleBar(dx, units=units, **kwargs))


def _untangle_text(
    text_artists: list[Text], ax: Axes | None = None, max_iterations: int = 10_000
):
    if ax is None:
        ax = plt.gca()
    inv = ax.transData.inverted()

    artist_coords = np.array(
        [text_artist.get_position() for text_artist in text_artists]
    )
    artist_coords = artist_coords + np.random.normal(0, 0.001, artist_coords.shape)
    artist_bbox = [text_artist.get_window_extent() for text_artist in text_artists]
    artist_extents = np.array(
        [inv.transform(bbox.get_points()) for bbox in artist_bbox]
    )
    artist_extents = artist_extents[:, 1] - artist_extents[:, 0]

    for i in range(max_iterations):
        relative_positions_x = (
            artist_coords[:, 0][:, None] - artist_coords[:, 0][None, :]
        )
        relative_positions_y = (
            artist_coords[:, 1][:, None] - artist_coords[:, 1][None, :]
        )

        relative_positions_x /= (
            0.1 + (artist_extents[:, 0][:, None] + artist_extents[:, 0][None, :]) / 2
        )
        relative_positions_y /= (
            0.1 + (artist_extents[:, 1][:, None] + artist_extents[:, 1][None, :]) / 2
        )

        # distances = np.sqrt(relative_positions_x**2+relative_positions_y**2)
        distances = np.abs(relative_positions_x) + np.abs(relative_positions_y)

        gaussian_repulsion = 1 * np.exp(-distances / 0.5)

        velocities_x = np.zeros_like(relative_positions_x)
        velocities_y = np.zeros_like(relative_positions_y)

        velocities_x[distances > 0] = (
            gaussian_repulsion[distances > 0]
            * relative_positions_x[distances > 0]
            / distances[distances > 0]
        )
        velocities_y[distances > 0] = (
            gaussian_repulsion[distances > 0]
            * relative_positions_y[distances > 0]
            / distances[distances > 0]
        )

        velocities_x[np.eye(velocities_x.shape[0], dtype=bool)] = 0
        velocities_y[np.eye(velocities_y.shape[0], dtype=bool)] = 0

        delta = np.stack([velocities_x, velocities_y], axis=1).mean(-1)
        # # delta = delta.clip(-0.1,0.1)
        artist_coords = artist_coords + delta * 0.1
        # artist_coords  = artist_coords*0.9 + initial_artist_coords*0.1

    for i, text_artist in enumerate(text_artists):
        text_artist.set_position(artist_coords[i, :])


def plot_umap(ovrlp: Ovrlp, *, annotate: bool = True, ax: Axes | None = None, **kwargs):
    """
    Plots the UMAP embedding.

    Parameters
    ----------
    annotate : bool
        Whether to add cell-type annotation if it was calculated using `Ovrlp.fit_signatures`.
    ax : matplotlib.axes.Axes | None
        Axis object to plot on.
    kwargs
        Keyword arguments for :py:func:`matplotlib.pyplot.scatter`.
    """
    if ax is None:
        _, ax = plt.subplots()
    annotation = (
        ovrlp.signatures[:, 0] if annotate and hasattr(ovrlp, "signatures") else None
    )
    assert ("2D_UMAP" in ovrlp.pseudocells.obsm) and ("RGB" in ovrlp.pseudocells.obsm)
    ct_center = ovrlp.celltype_centers if hasattr(ovrlp, "celltype_centers") else None
    _plot_embeddings(
        ax,
        ovrlp.pseudocells.obsm["2D_UMAP"],
        ovrlp.pseudocells.obsm["RGB"],
        ct_center,
        annotation,
        **kwargs,
    )


# define a function that plots the embeddings, with celltype centers rendered as plt.texts on top:
def _plot_embeddings(
    ax: Axes,
    embedding: np.ndarray,
    color: np.ndarray,
    celltype_centers: np.ndarray | None = None,
    celltypes: Iterable[str] | None = None,
    scatter_kwargs: dict[str, Any] = {"alpha": 0.1, "marker": "."},
):
    ax.axis("off")

    alpha = scatter_kwargs.pop("alpha", 0.1)
    marker = scatter_kwargs.pop("marker", ".")

    ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=color,
        alpha=alpha,
        marker=marker,
        **scatter_kwargs,
    )
    ax.set(aspect="equal")

    if celltypes is not None and celltype_centers is not None:
        text_artists: list[Text] = []
        for i, celltype in enumerate(celltypes):
            if not np.isnan(celltype_centers[i, 0]):
                t = ax.text(
                    np.nan_to_num((celltype_centers[i, 0])),
                    np.nan_to_num(celltype_centers[i, 1]),
                    celltype,
                    color="k",
                    fontsize=12,
                )
                text_artists.append(t)

        _untangle_text(text_artists, ax)


def plot_tissue(
    ovrlp: Ovrlp,
    *,
    scalebar: dict | None = SCALEBAR_PARAMS,
    ax: Axes | None = None,
    **kwargs,
):
    """
    Plots the tissue colored by the UMAP embedding.

    Parameters
    ----------
    scalebar : dict[str, typing.Any] | None
        If `None` no scalebar will be plotted. Otherwise a dictionary with
        additional kwargs for ``matplotlib_scalebar.scalebar.ScaleBar``.
        By default :py:attr:`ovrlpy.SCALEBAR_PARAMS`
    ax : matplotlib.axes.Axes | None
        Axis object to plot on.
    kwargs
        Keyword arguments for the matplotlib's scatter plot function.
    """
    if ax is None:
        _, ax = plt.subplots()

    _plot_tissue_scatter(
        ax,
        ovrlp.pseudocells.obsm["spatial"][:, 0],
        ovrlp.pseudocells.obsm["spatial"][:, 1],
        ovrlp.pseudocells.obsm["RGB"],
        marker=".",
        **kwargs,
    )

    if scalebar is not None:
        _plot_scalebar(ax, **scalebar)


def _plot_tissue_scatter(ax: Axes, xs, ys, cs, *, title: str | None = None, **kwargs):
    ax.scatter(xs, ys, c=cs, **kwargs)
    ax.set_aspect("equal", adjustable="box")
    if title is not None:
        ax.set_title(title)


def _mark_roi_center(ax: Axes, x, y, roi):
    ax.scatter(x, y, c="k", marker="+", s=100)
    ax.set(xlim=roi[0], ylim=roi[1])


def plot_pseudocells(
    ovrlp: Ovrlp,
    *,
    umap_kwargs: Mapping[str, Any] = {"scatter_kwargs": {"s": 1}},
    tissue_kwargs: Mapping[str, Any] = {"s": 1},
    **kwargs,
) -> Figure:
    """
    Plots the UMAP and tissue side-to-side.

    Parameters
    ----------
    ovrlp : ovrlpy.Ovrlp
    umap_kwargs
        Keyword arguments for :py:func:`ovrlpy.plot_umap`.
    tissue_kwargs
        Keyword arguments for :py:func:`ovrlpy.plot_tissue`.
    kwargs
        Other keyword arguments will be passed to :py:func:`matplotlib.pyplot.subplots`

    Returns
    -------
    matplotlib.figure.Figure
    """

    fig, axs = plt.subplots(ncols=2, **kwargs)

    plot_umap(ovrlp, ax=axs[0], **umap_kwargs)
    plot_tissue(ovrlp, ax=axs[1], **tissue_kwargs)

    return fig


def plot_signal_integrity(
    ovrlp: Ovrlp,
    *,
    signal_threshold: float = 2.0,
    cmap: str | Colormap = BIH_CMAP,
    histogram: bool = True,
    scalebar: dict | None = SCALEBAR_PARAMS,
    **kwargs,
) -> Figure:
    """
    Plots the determined signal integrity of the tissue sample in a signal integrity map.

    Parameters
    ----------
    ovrlp : Ovrlp
    signal_threshold : float, optional
        Threshold below which the signal is faded out in the plot,
        to avoid displaying noisy areas with low predictive confidence.
    cmap : str | matplotlib.colors.Colormap, optional
        Colormap for display.
    histogram : bool, optional
        Whether to plot a histogram of integrity values alongside the map.
    scalebar : dict[str, typing.Any] | None
        If `None` no scalebar will be plotted. Otherwise a dictionary with
        additional kwargs for ``matplotlib_scalebar.scalebar.ScaleBar``.
        By default :py:attr:`ovrlpy.SCALEBAR_PARAMS`
    kwargs
        Other keyword arguments are passed to :py:func:`matplotlib.pyplot.subplots`

    Returns
    -------
    matplotlib.figure.Figure
    """
    integrity = ovrlp.integrity_map
    signal = ovrlp.signal_map

    cmap = Colormap(cmap) if isinstance(cmap, str) else cmap

    with plt.style.context("dark_background"):
        if histogram:
            kwargs = {"width_ratios": [6, 1]} | kwargs
            fig, axs = plt.subplots(ncols=2, **kwargs)
            ax_im, ax_hist = axs
            assert isinstance(ax_hist, Axes)
        else:
            fig, ax_im = plt.subplots(**kwargs)

        assert isinstance(ax_im, Axes)

        img = ax_im.imshow(
            integrity,
            cmap=cmap,
            alpha=((signal / signal_threshold).clip(0, 1) ** 2),
            vmin=0,
            vmax=1,
        )
        ax_im.invert_yaxis()
        ax_im.spines[["top", "right"]].set_visible(False)

        if scalebar is not None:
            _plot_scalebar(ax_im, **scalebar)

        if histogram:
            vals, bins = np.histogram(
                integrity[signal > signal_threshold],
                bins=50,
                range=(0, 1),
                density=True,
            )
            colors = cmap(bins[1:-1])
            bars = ax_hist.barh(bins[1:-1], vals[1:], height=0.01)
            for i, bar in enumerate(bars):
                bar.set_color(colors[i])
            ax_hist.set(ylim=(0, 1), ylabel="signal integrity")
            ax_hist.yaxis.tick_right()
            ax_hist.yaxis.set_label_position("right")
            ax_hist.set_xticks([], [])
            ax_hist.invert_xaxis()
            ax_hist.spines[["top", "bottom", "left"]].set_visible(False)

        else:
            fig.colorbar(img)

    return fig


def plot_region_of_interest(
    ovrlp: Ovrlp,
    x: float,
    y: float,
    *,
    window_size: int = 30,
    signal_threshold: float = 5.0,
    scalebar: dict | None = SCALEBAR_PARAMS,
    **kwargs,
) -> Figure:
    """
    Plot an overview of a zoomed-in region of interest.

    Parameters
    ----------
    ovrlp : Ovrlp
        Ovrlp object containing the fitted model
    x : float
        x coordinate of the region of interest
    y : float
        y coordinate of the region of interest
    window_size : int, optional
        Size of the window to display.
    signal_threshold : float, optional
        Threshold for the signal plot.
    scalebar : dict[str, typing.Any] | None
        If `None` no scalebar will be plotted. Otherwise a dictionary with
        additional kwargs for ``matplotlib_scalebar.scalebar.ScaleBar``.
        By default :py:attr:`ovrlpy.SCALEBAR_PARAMS`
    kwargs
        Other keyword arguments are passed to :py:func:`matplotlib.pyplot.figure`

    Returns
    -------
    matplotlib.figure.Figure
    """
    integrity = ovrlp.integrity_map
    signal = ovrlp.signal_map
    embedding = ovrlp.pseudocells.obsm["2D_UMAP"]

    # first, create and color-embed the subsample of the region of interest
    roi_transcripts = ovrlp.subset_transcripts(x, y, window_size=window_size).sort("z")
    _, embedding_color = ovrlp.transform_transcripts(roi_transcripts)
    roi_transcripts = roi_transcripts.with_columns(RGB=embedding_color)

    if x is None:
        x = (roi_transcripts["x"].max() + roi_transcripts["x"].min()) / 2
    if y is None:
        y = (roi_transcripts["y"].max() + roi_transcripts["y"].min()) / 2
    if window_size is None:
        window_size = int(
            1 + (roi_transcripts["x"].max() - roi_transcripts["x"].min()) / 2
        )

    roi = ((x - window_size, x + window_size), (y - window_size, y + window_size))

    fig = plt.figure(**kwargs)
    gs = fig.add_gridspec(2, 3)

    # integrity map
    ax_integrity = fig.add_subplot(gs[0, 2], label="signal_integrity", facecolor="k")

    img = ax_integrity.imshow(
        integrity,
        cmap=BIH_CMAP,
        alpha=(signal / signal_threshold).clip(0, 1),
        vmin=0,
        vmax=1,
    )

    ax_integrity.set_title("signal integrity")
    ax_integrity.invert_yaxis()
    fig.colorbar(img)

    ax_integrity.set_xlim(x - window_size, x + window_size)
    ax_integrity.set_ylim(y - window_size, y + window_size)

    # UMAP
    ax_umap = fig.add_subplot(gs[0, 0], label="umap")
    _plot_embeddings(ax_umap, embedding, ovrlp.pseudocells.obsm["RGB"])
    ax_umap.set_title("UMAP")

    # tissue map
    ax_tissue_whole: Axes = fig.add_subplot(gs[0, 1], label="celltype_map")
    plot_tissue(ovrlp, ax=ax_tissue_whole, s=1)

    roi_box = Rectangle(
        (x - window_size, y - window_size),
        2 * window_size,
        2 * window_size,
        fill=False,
        edgecolor="k",
        linewidth=2,
    )
    ax_tissue_whole.add_artist(roi_box)
    ax_tissue_whole.set_title("celltype map")

    # top view of ROI
    roi_scatter_kwargs = dict(marker=".", alpha=0.8, s=1.5e5 / window_size**2)

    ax_roi_top = fig.add_subplot(gs[1, 0], label="top_map")
    roi_top = roi_transcripts.filter(pl.col("z") > pl.col("z_center"))
    _plot_tissue_scatter(
        ax_roi_top,
        roi_top["x"],
        roi_top["y"],
        roi_top["RGB"].to_numpy(),
        title="ROI celltype map, top",
        **roi_scatter_kwargs,
    )
    _mark_roi_center(ax_roi_top, x, y, roi)

    # bottom view of ROI
    ax_roi_bottom = fig.add_subplot(gs[1, 1], label="bottom_map")
    roi_bottom = roi_transcripts.filter(pl.col("z") < pl.col("z_center"))[::-1]
    _plot_tissue_scatter(
        ax_roi_bottom,
        roi_bottom["x"],
        roi_bottom["y"],
        roi_bottom["RGB"].to_numpy(),
        title="ROI celltype map, bottom",
        **roi_scatter_kwargs,
    )
    _mark_roi_center(ax_roi_bottom, x, y, roi)

    if scalebar is not None:
        _plot_scalebar(ax_integrity, **scalebar)
        _plot_scalebar(ax_tissue_whole, **scalebar)
        _plot_scalebar(ax_roi_top, **scalebar)
        _plot_scalebar(ax_roi_bottom, **scalebar)

    # side view of ROI
    roi_side_scatter_kwargs = dict(s=10, alpha=0.5)

    sub_gs = gs[1, 2].subgridspec(2, 1)

    ax_side_x = fig.add_subplot(sub_gs[0, 0], label="x_cut")
    roi_side_x = roi_transcripts.filter(pl.col("y") < (y + 4), pl.col("y") > (y - 4))

    _plot_tissue_scatter(
        ax_side_x,
        roi_side_x["x"],
        roi_side_x["z"],
        roi_side_x["RGB"].to_numpy(),
        title="ROI, vertical, x-cut",
        **roi_side_scatter_kwargs,
    )

    ax_side_y = fig.add_subplot(sub_gs[1, 0], label="y_cut")
    roi_side_y = roi_transcripts.filter(pl.col("x") < (x + 4), pl.col("x") > (x - 4))

    _plot_tissue_scatter(
        ax_side_y,
        roi_side_y["y"],
        roi_side_y["z"],
        roi_side_y["RGB"].to_numpy(),
        title="ROI, vertical, y-cut",
        **roi_side_scatter_kwargs,
    )

    fig.tight_layout()

    return fig
