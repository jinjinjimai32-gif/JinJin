# Preserved baseline: fixed 3x3 median, symmetric boundary padding.
import argparse
import numpy as np
from PIL import Image
from numpy.lib.stride_tricks import sliding_window_view

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('input')
    p.add_argument('output')
    a = p.parse_args()
    image = np.asarray(Image.open(a.input).convert('L'))
    padded = np.pad(image, 1, mode='symmetric')
    windows = sliding_window_view(padded, (3, 3))
    output = np.median(windows, axis=(-2, -1)).astype(np.uint8)
    Image.fromarray(output).save(a.output)
