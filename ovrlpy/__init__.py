from importlib.metadata import PackageNotFoundError, version

from . import io
from ._ovrlp import (
    Visualizer,
    compute_VSI,
    detect_doublets,
    get_pseudocell_locations,
    plot_region_of_interest,
    plot_signal_integrity,
    pre_process_coordinates,
    run,
    sample_expression_at_xy,
)
from ._utils import SCALEBAR_PARAMS, UMAP_2D_PARAMS, UMAP_RGB_PARAMS

try:
    __version__ = version("ovrlpy")
except PackageNotFoundError:
    __version__ = "unknown version"

del PackageNotFoundError, version


__all__ = [
    "io",
    "compute_VSI",
    "detect_doublets",
    "sample_expression_at_xy",
    "get_pseudocell_locations",
    "plot_region_of_interest",
    "plot_signal_integrity",
    "pre_process_coordinates",
    "Visualizer",
    "run",
    "SCALEBAR_PARAMS",
    "UMAP_2D_PARAMS",
    "UMAP_RGB_PARAMS",
]
