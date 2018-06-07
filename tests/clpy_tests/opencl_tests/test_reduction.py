# -*- coding: utf-8 -*-

import unittest

import numpy as np

import clpy as cp


class TestSimpleReductionFunction(unittest.TestCase):
    def setUp(self):
        self.my_func = cp.core.create_reduction_func(
            'my_sum', ('b->b',), ('in0', 'a + b', 'out0 = a', None))

    def test_add(self):
        x_np = np.array([5, 7], dtype='int8')
        x = cp.array(x_np)

        y = self.my_func(x)
        actual = np.array(y.get())
        expected = x_np.sum()  # 12
        self.assertEqual(actual, expected)

    def test_sum(self):
        n = 1024
        k = 1
        x_np = np.full([n], k, dtype='int8')
        x = cp.core.array(x_np)

        y = self.my_func(x)
        expected = np.array(n * k, dtype=np.int8)
        actual = np.array(y.get())
        self.assertEqual(actual, expected)

    def test_sum_square_matrix(self):
        x_np = np.array([[5, 7], [16, 18]], dtype='int8')
        x = cp.array(x_np)

        y = self.my_func(x, axis=1)
        actual = np.array(y.get())
        expected = x_np.sum(axis=1)  # [12, 34]
        self.assertTrue(np.all(actual == expected))

    def test_sum_matrix(self):
        x_np = np.array([[5, 7, 9], [6, 8, 10]], dtype='int8')
        x = cp.array(x_np)

        y = self.my_func(x, axis=1)
        actual = np.array(y.get())
        expected = x_np.sum(axis=1)  # [21, 24]
        self.assertTrue(np.all(actual == expected))


class TestReductionKernel(unittest.TestCase):
    def test_reduce_add(self):
        n = 1024
        k = 1
        x_np = np.full([n], k, dtype='int32')
        x = cp.core.array(x_np)

        kernel = cp.core.ReductionKernel(
            'T x', 'T y', 'x', 'a+b', 'y = a', '0', 'add_with0')
        y = kernel(x)

        expected = n * k
        actual = y.get()
        self.assertEqual(actual, expected)

    def test_reduce_pre(self):
        """Test with pre_map_expr"""
        n = 1024
        k = 1
        offset = 10
        x_np = np.full([n], k, dtype='int32')
        x = cp.core.array(x_np)

        kernel = cp.core.ReductionKernel(
            'T x', 'T y', 'x+' + str(offset), 'a+b', 'y = a', '0',
            'add_with0')
        y = kernel(x)

        expected = n * (k + offset)
        actual = y.get()
        self.assertEqual(actual, expected)

    def test_reduce_post(self):
        """Test with post_map_expr"""
        n = 1024
        k = 1
        x_np = np.full([n], k, dtype='int32')
        x = cp.core.array(x_np)

        kernel = cp.core.ReductionKernel(
            'T x', 'T y', 'x', 'a+b', 'y = a * a', '0', 'add_with0')
        y = kernel(x)

        expected = (n * k) * (n * k)
        actual = y.get()
        self.assertEqual(actual, expected)

    def test_reduce_matrix(self):
        """Test with multi dimensional input"""
        n = 128
        k = 1
        x_np = np.full([n, n], k, dtype='int32')
        x = cp.core.array(x_np)

        kernel = cp.core.ReductionKernel(
            'T x', 'T y', 'x', 'a+b', 'y = a', '0', 'add_with0')
        y = kernel(x)

        expected = n * n * k
        actual = y.get()
        self.assertEqual(actual, expected)

    def test_reduce_flag(self):
        x_np = np.array([1,    2,     3,    4], dtype='int32')
        f_np = np.array([True, True, False, True], dtype=np.bool)
        x = cp.array(x_np)
        f = cp.array(f_np)

        kernel = cp.core.ReductionKernel(
            'T x, raw bool f', 'T y', '(f[_j] ? x : T(0))', 'a+b', 'y = a',
            '0', 'reduce_flag')
        y = kernel(x, f)

        expected = 0
        for x, f in zip(x_np, f_np):
            if f:
                expected += x
        actual = y.get()
        self.assertEqual(actual, expected)

    def test_size(self):
        x_np = np.array([1,    2,     3,    4], dtype='int32')
        x = cp.array(x_np)

        kernel = cp.core.ReductionKernel(
            'T x', 'T y', '_type_reduce(x)', 'a+b',
            'y = a * _in_ind.size() * _out_ind.size()', '0', 'reduce_size')
        y = kernel(x)

        expected = x_np.sum() * len(x_np) * 1
        actual = y.get()
        self.assertEqual(actual, expected)


class TestReductionKernelwithChunk(unittest.TestCase):
    """test class of ReductionKernel with Chunk"""

    def setUp(self):
        # create chunk and free to prepare chunk in pool
        self.pool = cp.backend.memory.SingleDeviceMemoryPool()
        cp.backend.memory.set_allocator(self.pool.malloc)
        self.pooled_chunk_size = cp.backend.memory.subbuffer_alignment * 2
        self.tmp = self.pool.malloc(self.pooled_chunk_size)
        self.pool.free(self.tmp.buf, self.pooled_chunk_size, 0)

    def tearDown(self):
        cp.backend.memory.set_allocator()

    def test_raw_map_expr(self):
        dummy_size = 3
        dtype = np.float32
        coeff_np = np.array([1.0 / 3.0], dtype=dtype)
        x_np = np.array([1.0,  2.0,  3.0], dtype=dtype)

        # get chunk with offset = 0
        dummy = cp.empty(dummy_size, dtype)
        self.assertTrue(dummy.data.offset == 0)

        # fill dummy to detect error
        dummy.fill(0)

        # get chunk with offset != 0
        coeff = cp.core.array(coeff_np)
        self.assertTrue(coeff.data.mem.buf ==
                        dummy.data.mem.buf and coeff.data.mem.offset != 0)
        x = cp.core.array(x_np)

        kernel = cp.core.ReductionKernel(
            'T x, raw T coeff',  # in_params
            'T y',  # out_params
            'x * coeff[0];',  # map_expr
            'a + b',  # reduce_expr
            'y = 2 * a',  # post_map_expr
            '0',  # identity
            'reduction_sample')
        y = kernel(x, coeff)
        actual = y.get()

        expected = 2 * np.sum(x_np * coeff_np[0])
        self.assertTrue(np.allclose(actual, expected))

    def test_raw_post_map_expr(self):
        dummy_size = 3
        dtype = np.float32
        coeff_np = np.array([1.0 / 3.0], dtype=dtype)
        x_np = np.array([1.0,  2.0,  3.0], dtype=dtype)

        # get chunk with offset = 0
        dummy = cp.empty(dummy_size, dtype)
        self.assertTrue(dummy.data.offset == 0)

        # fill dummy to detect error
        dummy.fill(0)

        # get chunk with offset != 0
        coeff = cp.core.array(coeff_np)
        self.assertTrue(coeff.data.mem.buf ==
                        dummy.data.mem.buf and coeff.data.mem.offset != 0)

        x = cp.core.array(x_np)

        kernel = cp.core.ReductionKernel(
            'T x, raw T coeff',  # in_params
            'T y',  # out_params
            'x * 2;',  # map_expr
            'a + b',  # reduce_expr
            'y = a * coeff[0]',  # post_map_expr
            '0',  # identity
            'reduction_sample')
        y = kernel(x, coeff)

        actual = y.get()
        expected = coeff_np[0] * np.sum(x_np * 2)

        self.assertTrue(np.allclose(actual, expected))

    def test_raw_different_index(self):
        dummy_size = 3
        dtype = np.float32
        coeff_np = np.array([1.0 / 3.0, -1.0 / 7.0], dtype=dtype)
        x_np = np.array([1.0,  2.0,  3.0], dtype=dtype)

        # get chunk with offset = 0
        dummy = cp.empty(dummy_size, dtype)
        self.assertTrue(dummy.data.offset == 0)

        # fill dummy to detect error
        dummy.fill(0)
        dummy[1] = -100

        # get chunk with offset != 0
        coeff = cp.core.array(coeff_np)
        self.assertTrue(coeff.data.mem.buf ==
                        dummy.data.mem.buf and coeff.data.mem.offset != 0)
        x = cp.core.array(x_np)

        kernel = cp.core.ReductionKernel(
            'T x, raw T coeff',  # in_params
            'T y',  # out_params
            'x * coeff[0];',  # map_expr
            'a + b',  # reduce_expr
            'y = 2 * a + coeff[1]',  # post_map_expr
            '0',  # identity
            'reduction_sample')
        y = kernel(x, coeff)
        actual = y.get()

        expected = 2 * np.sum(x_np * coeff_np[0]) + coeff_np[1]
        self.assertTrue(np.allclose(actual, expected))

    def test_2_raw_array(self):
        dummy_size = 3
        dtype = np.float32
        coeff1_np = np.array([1.0 / 3.0], dtype=dtype)
        coeff2_np = np.array([1.0 / 7.0], dtype=dtype)
        x_np = np.array([1.0,  2.0,  3.0], dtype=dtype)

        # get chunk with offset = 0
        dummy = cp.empty(dummy_size, dtype)
        self.assertTrue(dummy.data.offset == 0)

        # fill dummy to detect error
        dummy.fill(0)

        # get chunk with offset != 0
        coeff1 = cp.core.array(coeff1_np)
        self.assertTrue(coeff1.data.mem.buf ==
                        dummy.data.mem.buf and coeff1.data.mem.offset != 0)

        # create another chunk
        self.tmp2 = self.pool.malloc(self.pooled_chunk_size)
        self.pool.free(self.tmp2.buf, self.pooled_chunk_size, 0)

        dummy2 = cp.empty(dummy_size, dtype)
        self.assertTrue(dummy2.data.offset == 0)

        # fill dummy to detect error
        dummy2.fill(0)

        # get chunk with offset != 0
        coeff2 = cp.core.array(coeff2_np)
        self.assertTrue(dummy2.data.mem.buf ==
                        coeff2.data.mem.buf and coeff2.data.mem.offset != 0)

        # chunk with offset == 0
        x = cp.core.array(x_np)
        self.assertTrue(x.data.mem.offset == 0)

        kernel = cp.core.ReductionKernel(
            'T x, raw T coeff1, raw T coeff2',  # in_params
            'T y',  # out_params
            'x * coeff1[0] + coeff2[0];',  # map_expr
            'a + b',  # reduce_expr
            'y = 2 * a',  # post_map_expr
            '0',  # identity
            'reduction_sample')
        y = kernel(x, coeff1, coeff2)
        actual = y.get()

        expected = 2 * np.sum(x_np * coeff1_np[0] + coeff2_np[0])
        self.assertTrue(np.allclose(actual, expected))


if __name__ == "__main__":
    unittest.main()
