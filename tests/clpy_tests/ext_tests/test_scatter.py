import unittest

import numpy

import clpy
from clpy import testing


@testing.gpu
class TestScatter(unittest.TestCase):

    def test_scatter_add(self):
        a = clpy.zeros((3,), dtype=numpy.float32)
        i = clpy.array([1, 1], numpy.int32)
        v = clpy.array([2., 1.], dtype=numpy.float32)
        clpy.scatter_add(a, i, v)
        testing.assert_array_equal(a, clpy.array([0, 3, 0]))
