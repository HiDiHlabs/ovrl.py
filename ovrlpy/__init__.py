from importlib.metadata import PackageNotFoundError, version

from ._ovrlp import (
    Visualizer,
    detect_doublets,
    get_pseudocell_locations,
    load_visualizer,
    plot_region_of_interest,
    plot_signal_integrity,
    pre_process_coordinates,
    run,
    sample_expression_at_xy,
)

try:
    __version__ = version("ovrlpy")
except PackageNotFoundError:
    __version__ = "unknown version"

del PackageNotFoundError, version


__all__ = [
    "detect_doublets",
    "sample_expression_at_xy",
    "get_pseudocell_locations",
    "load_visualizer",
    "plot_region_of_interest",
    "plot_signal_integrity",
    "pre_process_coordinates",
    "Visualizer",
    "run",
]
