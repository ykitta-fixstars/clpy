import unittest

import numpy

import clpy
from clpy import testing


@testing.gpu
class TestFormatting(unittest.TestCase):

    _multiprocess_can_split_ = True

    def test_array_repr(self):
        a = testing.shaped_arange((2, 3, 4), clpy)
        b = testing.shaped_arange((2, 3, 4), numpy)
        self.assertEqual(clpy.array_repr(a), numpy.array_repr(b))

    def test_array_str(self):
        a = testing.shaped_arange((2, 3, 4), clpy)
        b = testing.shaped_arange((2, 3, 4), numpy)
        self.assertEqual(clpy.array_str(a), numpy.array_str(b))
