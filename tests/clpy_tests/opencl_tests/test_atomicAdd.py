import unittest

import numpy as np

import clpy


class TestAtomicAdd(unittest.TestCase):
    """test atomicAdd function"""

    def test_float32(self):
        size = 128
        dtype = np.float32
        npX = np.arange(size, dtype=dtype)
        npY = np.flip(np.arange(size, dtype=dtype) * 2, -1)

        x = clpy.array(npX, dtype=dtype)
        y = clpy.array(npY, dtype=dtype)

        kernel = clpy.core.ElementwiseKernel(
            'raw T x, T y',
            'T z',
            '''
            atomicAdd(&x[i], y);
            z = x[i];
            ''',
            'test_atomicAdd'
        )
        z = kernel(x, y)

        actual = z.get()
        expected = npX + npY

        self.assertTrue(np.allclose(actual, expected))

    def test_float32_conflict(self):
        size = 128
        dtype = np.float32
        npX = np.arange(size, dtype=dtype)
        npZ = np.arange(size, dtype=dtype)

        x = clpy.array(npX, dtype=dtype)
        z = clpy.array(npZ, dtype=dtype)

        kernel = clpy.core.ElementwiseKernel(
            'T x',
            'raw T z',
            '''
            atomicAdd(&z[0], x);
            ''',
            'test_atomicAdd'
        )
        kernel(x, z)

        actual = z.get()
        npZ[0] += np.sum(npX)
        expected = npZ

        self.assertTrue(np.allclose(actual, expected))


if __name__ == "__main__":
    unittest.main()
