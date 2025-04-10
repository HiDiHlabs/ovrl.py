from math import floor

import numpy as np
import pandas as pd


def ceildiv(a: int, b: int):
    return -(a // -b)


def n_patches(length: int, size: tuple[int, int]):
    return ceildiv(size[0], length) * ceildiv(size[1], length)


def _patches(
    df: pd.DataFrame,
    length: int,
    padding: int,
    *,
    size: None | tuple[int, int] = None,
    coordinates: tuple[str, str] = ("x", "y"),
):
    x, y = coordinates

    if size is None:
        size = (int(floor(df[x].max() + 1)), int(floor(df[y].max() + 1)))

    # ensure that patch_length is an upper-bound for the actual size
    patch_count_x = ceildiv(size[0], length) + 1
    patch_count_y = ceildiv(size[1], length) + 1

    x_patches = np.linspace(0, size[0], patch_count_x, dtype=int)
    y_patches = np.linspace(0, size[1], patch_count_y, dtype=int)

    for i in range(len(x_patches) - 1):
        for j in range(len(y_patches) - 1):
            x_ = x_patches[i] - padding
            y_ = y_patches[j] - padding
            _x = x_patches[i + 1] + padding
            _y = y_patches[j + 1] + padding

            size_x = _x - x_ - padding * 2
            size_y = _y - y_ - padding * 2

            patch = df[
                (df[x] >= x_) & (df[x] < _x) & (df[y] >= y_) & (df[y] < _y)
            ].copy()
            patch[x] -= x_
            patch[y] -= y_
            yield patch, (x_, y_), (size_x, size_y)
