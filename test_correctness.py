"""Small independent checks for numerical and boundary correctness."""
import unittest
import numpy as np
from src.filters import adaptive_median, fixed_median
from src.metrics import psnr, ssim


def slow_reference(a, maximum):
    out=a.copy()
    for y in range(a.shape[0]):
        for x in range(a.shape[1]):
            for size in range(3,maximum+1,2):
                p=size//2
                padded=np.pad(a,p,mode='symmetric')
                values=sorted(padded[y:y+size,x:x+size].ravel().tolist())
                lo,mid,hi=values[0],values[len(values)//2],values[-1]
                if lo < mid < hi:
                    out[y,x]=int(a[y,x]) if lo < int(a[y,x]) < hi else mid
                    break
                if size==maximum: out[y,x]=mid
    return out


class Correctness(unittest.TestCase):
    def test_adaptive_matches_independent_pixel_reference(self):
        rng=np.random.default_rng(11)
        for a in [rng.integers(0,256,(9,11),dtype=np.uint8),
                  rng.choice([0,30,90,255],(9,11)).astype(np.uint8),
                  np.zeros((8,8),dtype=np.uint8),np.full((8,8),255,dtype=np.uint8)]:
            for size in [3,5,7]:
                np.testing.assert_array_equal(adaptive_median(a,size),slow_reference(a,size))

    def test_constant_and_impulse(self):
        a=np.full((9,9),100,dtype=np.uint8);a[4,4]=255
        self.assertEqual(int(fixed_median(a)[4,4]),100)
        np.testing.assert_array_equal(adaptive_median(a),np.full_like(a,100))

    def test_metrics(self):
        x=np.arange(100,dtype=np.uint8).reshape(10,10)
        self.assertAlmostEqual(ssim(x,x),1)
        self.assertTrue(np.isinf(psnr(x,x)))
        self.assertAlmostEqual(psnr(np.zeros_like(x),np.full_like(x,10)),28.1308036087)
        self.assertLess(ssim(x,np.flipud(x)),1)

    def test_invalid_window(self):
        with self.assertRaises(ValueError): adaptive_median(np.zeros((8,8),dtype=np.uint8),4)

if __name__=='__main__': unittest.main()
