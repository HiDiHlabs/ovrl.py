from math import floor

import numpy as np
import polars as pl


def ceildiv(a: int, b: int) -> int:
    return -(a // -b)


def n_patches(length: int, size: tuple[int, int]) -> int:
    return ceildiv(size[0], length) * ceildiv(size[1], length)


def _patches(
    df: pl.DataFrame,
    length: int,
    padding: int,
    *,
    size: None | tuple[int, int] = None,
    coordinates: tuple[str, str] = ("x", "y"),
):
    x_key, y_key = coordinates

    if size is None:
        size = (int(floor(df[x_key].max() + 1)), int(floor(df[y_key].max() + 1)))

    # ensure that patch_length is an upper-bound for the actual size
    patch_count_x = ceildiv(size[0], length)
    patch_count_y = ceildiv(size[1], length)

    x_patches = np.linspace(0, size[0], patch_count_x + 1, dtype=int)
    y_patches = np.linspace(0, size[1], patch_count_y + 1, dtype=int)

    for i in range(len(x_patches) - 1):
        for j in range(len(y_patches) - 1):
            # coordinates with padding
            x_ = max(0, x_patches[i] - padding)
            y_ = max(0, y_patches[j] - padding)
            _x = min(size[0], x_patches[i + 1] + padding)
            _y = min(size[1], y_patches[j + 1] + padding)

            padded_range = (slice(x_, _x), slice(y_, _y))

            patch = df.filter(
                pl.col(x_key).is_between(x_, _x, closed="left")
                & pl.col(y_key).is_between(y_, _y, closed="left")
            ).with_columns(pl.col(x_key) - x_, pl.col(y_key) - y_)

            unpadded_range = (
                slice(x_patches[i], x_patches[i + 1]),
                slice(y_patches[j], y_patches[j + 1]),
            )

            yield patch, padded_range, unpadded_range
