import unittest

import clpy
import numpy


class TestConcatenate(unittest.TestCase):
    """test clpy.manipulate.join.concatenate method"""

    def get_numpy_clpy_concatenated_result(self, dtype, shapes, axis):
        length = []
        numpy_ar = []
        clpy_ar = []
        num_array = len(shapes)

        for i in range(num_array):
            length.append(numpy.prod(shapes[i]))
            numpy_ar.append(numpy.arange(
                length[i], dtype=dtype).reshape(shapes[i]))
            clpy_ar.append(clpy.array(numpy_ar[i]))

        clpy_result = clpy.concatenate((clpy_ar), axis).get()
        numpy_result = numpy.concatenate((numpy_ar), axis)

        return (numpy_result, clpy_result)

    def test_concatenate_2d_2array_axis0(self):
        dtype = "int64"
        axis = 0
        shapes = [(2, 2), (3, 2)]

        numpy_result, clpy_result = self.get_numpy_clpy_concatenated_result(
            dtype, shapes, axis)

        self.assertTrue(numpy.array_equal(clpy_result, numpy_result))

    def test_concatenate_2d_2array_axis1(self):
        dtype = "int64"
        axis = 1
        shapes = [(2, 2), (2, 3)]

        numpy_result, clpy_result = self.get_numpy_clpy_concatenated_result(
            dtype, shapes, axis)

        self.assertTrue(numpy.array_equal(clpy_result, numpy_result))

    def test_concatenate_3d_3array_axis0(self):
        dtype = "int64"
        axis = 0
        shapes = [(2, 2, 2), (3, 2, 2), (4, 2, 2)]

        numpy_result, clpy_result = self.get_numpy_clpy_concatenated_result(
            dtype, shapes, axis)

        self.assertTrue(numpy.array_equal(clpy_result, numpy_result))

    def test_concatenate_3d_3array_axis1(self):
        dtype = "int64"
        axis = 1
        shapes = [(2, 2, 2), (2, 3, 2), (2, 4, 2)]

        numpy_result, clpy_result = self.get_numpy_clpy_concatenated_result(
            dtype, shapes, axis)

        self.assertTrue(numpy.array_equal(clpy_result, numpy_result))

    def test_concatenate_3d_3array_axis2(self):
        dtype = "int64"
        axis = 2
        shapes = [(2, 2, 2), (2, 2, 3), (2, 2, 4)]

        numpy_result, clpy_result = self.get_numpy_clpy_concatenated_result(
            dtype, shapes, axis)

        self.assertTrue(numpy.array_equal(clpy_result, numpy_result))

    def test_concatenate_3d_4array_axis0(self):
        dtype = "int64"
        axis = 0
        shapes = [(2, 2, 2), (3, 2, 2), (4, 2, 2), (5, 2, 2)]

        numpy_result, clpy_result = self.get_numpy_clpy_concatenated_result(
            dtype, shapes, axis)

        self.assertTrue(numpy.array_equal(clpy_result, numpy_result))

    def test_concatenate_3d_4array_axis1(self):
        dtype = "int64"
        axis = 1
        shapes = [(2, 2, 2), (2, 3, 2), (2, 4, 2), (2, 5, 2)]

        numpy_result, clpy_result = self.get_numpy_clpy_concatenated_result(
            dtype, shapes, axis)

        self.assertTrue(numpy.array_equal(clpy_result, numpy_result))

    def test_concatenate_3d_4array_axis2(self):
        dtype = "int64"
        axis = 2
        shapes = [(2, 2, 2), (2, 2, 3), (2, 2, 4), (2, 2, 5)]

        numpy_result, clpy_result = self.get_numpy_clpy_concatenated_result(
            dtype, shapes, axis)

        self.assertTrue(numpy.array_equal(clpy_result, numpy_result))


if __name__ == '__main__':
    unittest.main()
