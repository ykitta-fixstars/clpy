import unittest

import numpy as np

import clpy
import clpy.backend.memory

# TODO(LWisteria): Merge to core_tests


class TestNdarray(unittest.TestCase):
    """test class of ndarray"""

    def test_create(self):
        clpy.ndarray([1, 2])
        # Always OK if no exception when ndarray.__init__
        self.assertTrue(True)

    def test_set(self):
        src = np.array([0, 1, 2, 3], dtype="float64")
        dst = clpy.ndarray(src.shape)
        dst.set(src)
        self.assertTrue(True)  # Always OK if no exception when ndarray.set

    def test_single_getset(self):
        expected = np.array([0, 1, 2, 3], dtype="float64")

        ar = clpy.ndarray(expected.shape)
        ar.set(expected)

        actual = ar.get()

        self.assertTrue((expected == actual).all())

    def test_multiple_getset(self):
        expected0 = np.array([0, 1, 2, 3], dtype="float64")
        ar0 = clpy.ndarray(expected0.shape)
        ar0.set(expected0)

        expected1 = np.array([4, 5, 6, 7], dtype="float64")
        ar1 = clpy.ndarray(expected1.shape)
        ar1.set(expected1)

        actual0 = ar0.get()
        actual1 = ar1.get()
        self.assertTrue((expected0 == actual0).all())
        self.assertTrue((expected1 == actual1).all())

    def test_array(self):
        ar = clpy.core.array([
            [1, 2, 3],
            [4, 5, 6]], dtype='float32')

        actual = ar.get()
        expected = np.array([
            [1, 2, 3],
            [4, 5, 6]], dtype='float32')

        self.assertTrue((expected == actual).all())

    def test_data(self):
        ar = clpy.ndarray([1, 2])
        self.assertIsInstance(ar.data.buf, clpy.backend.memory.Buf)

    def test_dot(self):
        an_array = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        a = clpy.core.array(an_array, dtype='float32')
        b = clpy.core.array(an_array, dtype='float32')

        expected = np.array(an_array).dot(np.array(an_array))
        actual = a.dot(b).get()

        self.assertTrue((expected == actual).all())

    def test_reshape(self):
        an_array = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        a = clpy.core.array(an_array, dtype='float32')

        expected = np.array(an_array, dtype='float32').reshape((2, 6))
        actual = a.reshape((2, 6)).get()

        self.assertTrue(expected.shape == actual.shape)
        self.assertTrue((expected == actual).all())

    def test_ravel(self):
        an_array = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        a = clpy.core.array(an_array, dtype='float32')

        expected = np.array(an_array, dtype='float32').ravel()
        actual = a.ravel().get()

        self.assertTrue((expected == actual).all())

    def test_reduced_view(self):
        # more sophisticated test may be needed
        an_array = [[[1], [2], [3]], [[4], [5], [6]], [[7], [8], [9]]]
        a = clpy.core.array(an_array, dtype='float32')

        expected = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype='float32')
        actual = a.reduced_view().get()

        self.assertTrue(expected.shape == actual.shape)
        self.assertTrue((expected == actual).all())

    def test_fill(self):
        expected = np.ndarray((3, 3), dtype='float32')
        expected.fill(42.0)

        a = clpy.ndarray((3, 3), dtype='float32')
        a.fill(42.0)
        actual = a.get()

        self.assertTrue((expected == actual).all())

    def test_astype(self):
        an_array = [[1.3, 2.3, 3.3]]
        a = clpy.array(an_array, dtype='float32')

        expected = np.array(an_array, dtype='float32').astype('int32')
        actual = a.astype('int32').get()

        self.assertTrue(expected.dtype == actual.dtype)
        self.assertTrue((expected == actual).all())

    def test_transpose(self):
        x_np = np.array([[5, 7, 9], [6, 8, 10]], dtype='int8')
        x = clpy.array(x_np)

        expected = x_np.transpose()
        y = x.transpose()
        actual = y.get()

        self.assertTrue(np.all(expected == actual))

    def test_transpose_float(self):
        x_np = np.array([[1, 3], [2, 4]], dtype='float32')
        x = clpy.array(x_np)

        expected = x_np.transpose()
        y = x.transpose()
        actual = y.get()

        self.assertTrue(np.all(expected == actual))

    def test_max(self):
        x_np = np.array([[1, 3, 2, 4]], dtype='float32')
        x = clpy.array(x_np)

        expected = x_np.max()
        y = x.max()
        actual = y.get()

        self.assertTrue(np.all(expected == actual))

    def test_argmax(self):
        x_np = np.array([[1, 3, 2, 4]], dtype='float32')
        x = clpy.array(x_np)

        expected = x_np.argmax()
        y = x.argmax()
        actual = y.get()

        self.assertTrue(np.all(expected == actual))

    def test_min(self):
        x_np = np.array([[1, 3, 2, 4]], dtype='float32')
        x = clpy.array(x_np)

        expected = x_np.min()
        y = x.min()
        actual = y.get()

        self.assertTrue(np.all(expected == actual))

    def test_argmin(self):
        x_np = np.array([[4, 3, 1, 2]], dtype='float32')
        x = clpy.array(x_np)

        expected = x_np.argmin()
        y = x.argmin()
        actual = y.get()

        self.assertTrue(np.all(expected == actual))
        
    def test_sum(self):
        x_np = np.array([[1, 3, 2, 4]], dtype='float32')
        x = clpy.array(x_np)

        expected = x_np.sum()
        y = x.sum()
        actual = y.get()

        self.assertTrue(np.all(expected == actual))

    def test_ellipsis(self):
        x_np = np.array([1, 3, 2, 4], dtype='float32')
        x = clpy.array(x_np)

        x_np[...] = np.asarray(0)
        x[...] = clpy.asarray(0)

        expected = x_np
        actual = x.get()
        self.assertTrue(np.all(expected == actual))


if __name__ == "__main__":
    unittest.main()
