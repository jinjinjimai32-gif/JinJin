"""Preserved starting version: a single 3x3 median on a grayscale image."""
import argparse
import numpy as np
from PIL import Image
from numpy.lib.stride_tricks import sliding_window_view

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('output')
    args = parser.parse_args()
    image = np.asarray(Image.open(args.input).convert('L'))
    padded = np.pad(image, 1, mode='symmetric')
    windows = sliding_window_view(padded, (3, 3))
    output = np.median(windows, axis=(-2, -1)).astype(np.uint8)
    Image.fromarray(output).save(args.output)
