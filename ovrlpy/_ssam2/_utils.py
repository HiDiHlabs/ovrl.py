### utils file with basic functions needed for the SSAM algorithm.


import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max

_TRUNCATE = 4


def kde_2d(coordinates: np.ndarray, **kwargs):
    """
    Create a histogram of the data.
    """
    return _kde_nd(coordinates[:, :2], **kwargs)


def kde_3d(coordinates: np.ndarray, **kwargs):
    """
    Create a histogram of the data.
    """
    return _kde_nd(coordinates[:, :3], **kwargs)


def _kde_nd(
    coordinates: np.ndarray,
    size=None,
    bandwidth: float = 1.5,
    truncate: float = _TRUNCATE,
    dtype=np.float32,
):
    """
    Create a histogram of the data.
    """

    if coordinates.shape[0] == 0:
        if size is None:
            raise ValueError("If no coordinates are provided, size must be provided.")
        else:
            return np.zeros(size, dtype=dtype)

    if size is None:
        size = np.ceil(np.max(coordinates, axis=0)).astype(int)

    output = np.zeros(size, dtype=dtype)

    dim_bins = list()
    for i in range(coordinates.shape[1]):
        c_min = int(np.min(coordinates[:, i]))
        c_max = int(np.ceil(np.max(coordinates[:, i])))
        dim_bins.append(np.linspace(c_min, c_max, c_max - c_min + 1))

    histogram, bins = np.histogramdd(
        [coordinates[:, i] for i in range(coordinates.shape[1])], bins=dim_bins
    )

    kde = gaussian_filter(
        histogram, sigma=bandwidth, truncate=truncate, mode="constant", output=dtype
    )

    output[
        tuple(slice(int(bins[i].min()), int(bins[i].max())) for i in range(len(bins)))
    ] = kde

    return output


def find_local_maxima(
    vf: np.ndarray, min_pixel_distance: int = 5, min_expression: float = 2
):
    """
    Find local maxima in a vector field.
    """
    local_maxima = peak_local_max(
        vf,
        min_distance=min_pixel_distance,
        threshold_abs=min_expression,
        exclude_border=False,
    )

    return local_maxima


def kde_and_sample(coordinates: np.ndarray, sampling_coordinates: np.ndarray, **kwargs):
    """
    Create a kde of the data and sample at 'sampling_coordinates'.
    """

    sampling_coordinates = np.round(sampling_coordinates).astype(int)
    n_dims = sampling_coordinates.shape[1]

    kde = _kde_nd(coordinates, **kwargs)

    return kde[tuple(sampling_coordinates[:, i] for i in range(n_dims))]
