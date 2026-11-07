"""Median baselines and the classical two-stage adaptive median filter.

The adaptive algorithm is an implementation of a known method, not a new algorithm.
All neighborhoods are read from the original noisy input (no in-place feedback).
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def windows(image, size):
    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError('Expected a 2D uint8 grayscale image')
    if size < 3 or size % 2 != 1:
        raise ValueError('Window size must be an odd integer >= 3')
    # symmetric padding repeats the boundary pixel, like scipy mode=reflect.
    return sliding_window_view(np.pad(image, size // 2, mode='symmetric'), (size, size))


def fixed_median(image, size=3):
    return np.median(windows(image, size), axis=(-2, -1)).astype(np.uint8)


def switching_median(image):
    """Ablation: keep interior intensities, replace only 0/255 with 3x3 median."""
    med = fixed_median(image, 3)
    return np.where((image == 0) | (image == 255), med, image).astype(np.uint8)


def adaptive_median(image, max_size=7):
    """Classical stage A/B decision; fall back to median at maximum window."""
    if max_size < 3 or max_size % 2 != 1:
        raise ValueError('max_size must be an odd integer >= 3')
    result = image.copy()
    unresolved = np.ones(image.shape, dtype=bool)
    for size in range(3, max_size + 1, 2):
        w = windows(image, size)
        lo, hi = w.min(axis=(-2, -1)), w.max(axis=(-2, -1))
        med = np.median(w, axis=(-2, -1)).astype(np.uint8)
        # Comparisons avoid unsigned subtraction overflow.
        ready = unresolved & (med > lo) & (med < hi)
        keep = (image > lo) & (image < hi)
        result[ready] = np.where(keep, image, med)[ready]
        unresolved[ready] = False
        if not unresolved.any():
            break
        if size == max_size:
            result[unresolved] = med[unresolved]
    return result
