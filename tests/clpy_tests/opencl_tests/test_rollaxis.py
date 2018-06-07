# -*- coding: utf-8 -*-

import unittest

import clpy
import numpy


class TestRollaxis(unittest.TestCase):
    """test class of rollaxis"""

    def test_import(self):
        self.assertTrue(True)  # Always OK if no exeption from import

    def test_2_3_matrix(self):
        npA = numpy.array([[1, 2, 3], [4, 5, 6]])
        expectedA = numpy.rollaxis(npA, 1, 0)
        clpA = clpy.array([[1, 2, 3], [4, 5, 6]])
        actualA = clpy.rollaxis(clpA, 1, 0)
        self.assertTrue(numpy.allclose(expectedA, actualA.get()))

    def test_2_3_4_matrix(self):
        npA = numpy.array([[[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]], [
                          [4, 5, 6, 7], [5, 6, 7, 8], [6, 7, 8, 9]]])
        expectedA = numpy.rollaxis(npA, 1, 0)
        clpA = clpy.array([[[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]], [
                          [4, 5, 6, 7], [5, 6, 7, 8], [6, 7, 8, 9]]])
        actualA = clpy.rollaxis(clpA, 1, 0)
        self.assertTrue(numpy.allclose(expectedA, actualA.get()))


if __name__ == "__main__":
    unittest.main()
