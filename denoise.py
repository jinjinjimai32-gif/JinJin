"""Process a user-supplied image using the selected median filter."""
import argparse
import numpy as np
from PIL import Image
from src.filters import fixed_median, adaptive_median, switching_median

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('input')
    p.add_argument('output')
    p.add_argument('--method', choices=['median3','median5','switching3','adaptive7'], default='adaptive7')
    a = p.parse_args()
    methods = {'median3': lambda x: fixed_median(x,3), 'median5': lambda x: fixed_median(x,5),
               'switching3': switching_median, 'adaptive7': adaptive_median}
    image = np.asarray(Image.open(a.input).convert('L'))
    Image.fromarray(methods[a.method](image)).save(a.output)
