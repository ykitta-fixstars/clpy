# coding: utf-8

import unittest

import clpy
from clpy import backend
from clpy import testing


@testing.gpu
class TestScan(unittest.TestCase):

    @testing.for_all_dtypes()
    def test_scan(self, dtype):
        element_num = 10000

        if dtype in {clpy.int8, clpy.uint8, clpy.float16}:
            element_num = 100

        a = clpy.ones((element_num,), dtype=dtype)
        prefix_sum = clpy.core.core.scan(a)
        expect = clpy.arange(start=1, stop=element_num + 1).astype(dtype)

        testing.assert_array_equal(prefix_sum, expect)

    def test_check_1d_array(self):
        with self.assertRaises(TypeError):
            a = clpy.zeros((2, 2))
            clpy.core.core.scan(a)

    @testing.multi_gpu(2)
    def test_multi_gpu(self):
        with backend.Device(0):
            a = clpy.zeros((10,))
            clpy.core.core.scan(a)
        with backend.Device(1):
            a = clpy.zeros((10,))
            clpy.core.core.scan(a)

    @testing.for_all_dtypes()
    def test_scan_out(self, dtype):
        element_num = 10000

        if dtype in {clpy.int8, clpy.uint8, clpy.float16}:
            element_num = 100

        a = clpy.ones((element_num,), dtype=dtype)
        b = clpy.zeros_like(a)
        clpy.core.core.scan(a, b)
        expect = clpy.arange(start=1, stop=element_num + 1).astype(dtype)

        testing.assert_array_equal(b, expect)

        clpy.core.core.scan(a, a)
        testing.assert_array_equal(a, expect)
