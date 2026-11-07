"""PSNR and 7x7 uniform-window SSIM with sample covariance, valid interior."""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def psnr(reference, image):
    error = np.mean((reference.astype(np.float64) - image.astype(np.float64)) ** 2)
    return float('inf') if error == 0 else float(10 * np.log10(255.0 ** 2 / error))


def ssim(reference, image):
    x, y = reference.astype(np.float64), image.astype(np.float64)
    def mean7(a):
        return sliding_window_view(a, (7, 7)).mean(axis=(-2, -1))
    ux, uy = mean7(x), mean7(y)
    vx = (mean7(x*x) - ux*ux) * 49/48
    vy = (mean7(y*y) - uy*uy) * 49/48
    cov = (mean7(x*y) - ux*uy) * 49/48
    c1, c2 = (0.01*255)**2, (0.03*255)**2
    return float(np.mean(((2*ux*uy+c1)*(2*cov+c2)) / ((ux*ux+uy*uy+c1)*(vx+vy+c2))))
