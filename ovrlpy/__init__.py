from importlib.metadata import PackageNotFoundError, version

from ._ovrlp import (
    Visualizer,
    compute_coherence_map,
    detect_doublets,
    get_expression_vectors_at_rois,
    get_rois,
    load_visualizer,
    plot_instance,
    plot_signal_integrity,
    pre_process_coordinates,
)

try:
    __version__ = version("ovrlpy")
except PackageNotFoundError:
    __version__ = "unknown version"

del PackageNotFoundError, version


__all__ = [
    "compute_coherence_map",
    "detect_doublets",
    "get_expression_vectors_at_rois",
    "get_rois",
    "load_visualizer",
    "plot_instance",
    "plot_signal_integrity",
    "pre_process_coordinates",
    "Visualizer",
]
