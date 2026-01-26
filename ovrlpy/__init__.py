from importlib.metadata import PackageNotFoundError, version

from . import io
from ._ovrlp import Ovrlp
from ._plotting import (
    SCALEBAR_PARAMS,
    plot_pseudocells,
    plot_region_of_interest,
    plot_signal_integrity,
    plot_tissue,
    plot_umap,
)
from ._segmentation_utils import (
    cell_integrity_from_masks,
    cell_integrity_from_transcripts,
)
from ._subslicing import process_coordinates
from ._utils import UMAP_2D_PARAMS, UMAP_RGB_PARAMS

try:
    __version__ = version("ovrlpy")
except PackageNotFoundError:
    __version__ = "unknown version"

del PackageNotFoundError, version


__all__ = [
    "io",
    "Ovrlp",
    "plot_pseudocells",
    "plot_region_of_interest",
    "plot_signal_integrity",
    "plot_tissue",
    "plot_umap",
    "process_coordinates",
    "cell_integrity_from_masks",
    "cell_integrity_from_transcripts",
    "SCALEBAR_PARAMS",
    "UMAP_2D_PARAMS",
    "UMAP_RGB_PARAMS",
]
