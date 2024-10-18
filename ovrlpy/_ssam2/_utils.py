### utils file with basic functions needed for the SSAM algorithm.

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage import measure
from skimage.feature import peak_local_max


def kde_2d(coordinates, name=None, size=None, bandwidth=1.5):
    """
    Create a histogram of the data.
    """

    return _kde_nd(coordinates[:, :2], size=size, bandwidth=bandwidth)


def kde_3d(coordinates, size=None, bandwidth=1.5):
    """
    Create a histogram of the data.
    """

    return _kde_nd(coordinates[:, :3], size=size, bandwidth=bandwidth)


def _kde_nd(coordinates, size=None, bandwidth=1.5):
    """
    Create a histogram of the data.
    """

    if coordinates.shape[0] == 0:
        if size is None:
            raise ValueError("If no coordinates are provided, size must be provided.")
        else:
            return np.zeros(size)

    if size is None:
        size = np.ceil(np.max(coordinates, axis=0)).astype(int)

    output = np.zeros(size)

    dim_bins = list()
    for i in range(coordinates.shape[1]):
        c_min = int(np.min(coordinates[:, i]))
        c_max = int(np.ceil(np.max(coordinates[:, i])))
        dim_bins.append(np.linspace(c_min, c_max, c_max - c_min + 1))

    histogram, bins = np.histogramdd(
        [coordinates[:, i] for i in range(coordinates.shape[1])], bins=dim_bins
    )

    kde = gaussian_filter(histogram, sigma=bandwidth)

    output[
        tuple(slice(int(bins[i].min()), int(bins[i].max())) for i in range(len(bins)))
    ] = kde

    return output


def find_local_maxima(vf, min_pixel_distance: int = 5, min_expression: float = 2):
    """
    Find local maxima in a vector field.
    """
    local_maxima = peak_local_max(
        vf,
        min_distance=min_pixel_distance,
        threshold_abs=min_expression,
        exclude_border=True,
    )

    return local_maxima


def kde_and_sample(coordinates, sampling_coordinates, size=None, bandwidth=1.5):
    """
    Create a kde of the data and sample at 'sampling_coordinates'.
    """

    sampling_coordinates = np.round(sampling_coordinates).astype(int)

    kde = _kde_nd(coordinates, size=size, bandwidth=bandwidth)
    n_dims = sampling_coordinates.shape[1]
    output = kde[tuple(sampling_coordinates[:, i] for i in range(n_dims))]

    return output


def crosscorr(x, y):
    """
    Calculate the cross-correlation between two matrices.
    """
    x -= np.array(x.mean(1))[:, None]
    y -= np.array(y.mean(1))[:, None]
    c = (np.dot(x, y.T) / x.shape[1]).squeeze()

    return np.nan_to_num(
        np.nan_to_num(c / np.array(x.std(1))[:, None]) / np.array(y.std(1))[None, :]
    )


def fill_celltypemaps(
    ct_map, fill_blobs=True, min_blob_area=0, filter_params={}, output_mask=None
):
    """
    Post-filter cell type maps created by `map_celltypes`.

    :param min_r: minimum threshold of the correlation.
    :type min_r: float
    :param min_norm: minimum threshold of the vector norm.
        If a string is given instead, then the threshold is automatically determined using
        sklearn's `threshold filter functions <https://scikit-image.org/docs/dev/api/skimage.filters.html>`_ (The functions start with `threshold_`).
    :type min_norm: str or float
    :param fill_blobs: If true, then the algorithm automatically fill holes in each blob.
    :type fill_blobs: bool
    :param min_blob_area: The blobs with its area less than this value will be removed.
    :type min_blob_area: int
    :param filter_params: Filter parameters used for the sklearn's threshold filter functions.
        Not used when `min_norm` is float.
    :type filter_params: dict
    :param output_mask: If given, the cell type maps will be filtered using the output mask.
    :type output_mask: np.ndarray(bool)
    """

    filtered_ctmaps = np.zeros_like(ct_map) - 1

    for cidx in np.unique(ct_map):
        mask = ct_map == cidx
        if min_blob_area > 0 or fill_blobs:
            blob_labels = measure.label(mask, background=0)
            for bp in measure.regionprops(blob_labels):
                if min_blob_area > 0 and bp.filled_area < min_blob_area:
                    for c in bp.coords:
                        mask[c[0], c[1]] = 0

                    continue
                if fill_blobs and bp.area != bp.filled_area:
                    minx, miny, maxx, maxy = bp.bbox
                    mask[minx:maxx, miny:maxy] |= bp.filled_image

        filtered_ctmaps[
            np.logical_and(mask == 1, np.logical_or(ct_map == -1, ct_map == cidx))
        ] = cidx

    return filtered_ctmaps
