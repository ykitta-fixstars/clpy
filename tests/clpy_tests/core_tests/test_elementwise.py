import unittest

import numpy
import six

import clpy
# from clpy import backend
from clpy.backend.opencl.exceptions import OpenCLProgramBuildError
from clpy.backend.ultima.exceptions import UltimaRuntimeError
# from clpy import core
from clpy import testing


@testing.gpu
class TestElementwise(unittest.TestCase):

    _multiprocess_can_split_ = True

    # TODO(LWisteria): Enable below if multi device is implemented
    # def check_copy(self, dtype, src_id, dst_id):
    #     with backend.Device(src_id):
    #         src = testing.shaped_arange((2, 3, 4), dtype=dtype)
    #     with backend.Device(dst_id):
    #         dst = clpy.empty((2, 3, 4), dtype=dtype)
    #     core.elementwise_copy(src, dst)
    #     testing.assert_allclose(src, dst)

    # @testing.for_all_dtypes()
    # def test_copy(self, dtype):
    #     device_id = backend.Device().id
    #     self.check_copy(dtype, device_id, device_id)

    # @testing.multi_gpu(2)
    # @testing.for_all_dtypes()
    # def test_copy_multigpu(self, dtype):
    #     with self.assertRaises(ValueError):
    #         self.check_copy(dtype, 0, 1)

    @testing.for_orders('CFAK')
    @testing.for_all_dtypes()
    @testing.numpy_clpy_allclose()
    def test_copy_zero_sized_array1(self, xp, dtype, order):
        src = xp.empty((0,), dtype=dtype)
        res = xp.copy(src, order=order)
        self.assertIsNot(src, res)
        return res

    @testing.for_orders('CFAK')
    @testing.for_all_dtypes()
    @testing.numpy_clpy_allclose()
    def test_copy_zero_sized_array2(self, xp, dtype, order):
        src = xp.empty((1, 0, 2), dtype=dtype)
        res = xp.copy(src, order=order)
        self.assertIsNot(src, res)
        return res

    @testing.for_orders('CFAK')
    def test_copy_orders(self, order):
        a = clpy.empty((2, 3, 4))
        b = clpy.copy(a, order)

        a_cpu = numpy.empty((2, 3, 4))
        b_cpu = numpy.copy(a_cpu, order)

        self.assertEqual(b.strides, b_cpu.strides)


@testing.gpu
class TestElementwiseInvalidArgument(unittest.TestCase):

    _multiprocess_can_split_ = True

    def test_invalid_kernel_name(self):
        with six.assertRaisesRegex(self, ValueError, 'Invalid kernel name'):
            clpy.ElementwiseKernel('T x', '', '', '1')


@testing.gpu
class TestElementwiseType(unittest.TestCase):

    @testing.for_int_dtypes(no_bool=True)
    @testing.numpy_clpy_array_equal()
    def test_large_int_upper_1(self, xp, dtype):
        a = xp.array([0], dtype=xp.int8)
        b = xp.iinfo(dtype).max
        return a + b

    @testing.for_int_dtypes(no_bool=True)
    @testing.numpy_clpy_array_equal()
    def test_large_int_upper_2(self, xp, dtype):
        a = xp.array([1], dtype=xp.int8)
        b = xp.iinfo(dtype).max - 1
        return a + b

    @testing.for_int_dtypes(no_bool=True)
    @testing.numpy_clpy_array_equal()
    def test_large_int_upper_3(self, xp, dtype):
        a = xp.array([xp.iinfo(dtype).max], dtype=dtype)
        b = xp.int8(0)
        return a + b

    @testing.for_int_dtypes(no_bool=True)
    @testing.numpy_clpy_array_equal()
    def test_large_int_upper_4(self, xp, dtype):
        a = xp.array([xp.iinfo(dtype).max - 1], dtype=dtype)
        b = xp.int8(1)
        return a + b

    @testing.for_int_dtypes(no_bool=True)
    @testing.numpy_clpy_array_equal()
    def test_large_int_lower_1(self, xp, dtype):
        a = xp.array([0], dtype=xp.int8)
        b = xp.iinfo(dtype).min
        return a + b

    @testing.for_int_dtypes(no_bool=True)
    @testing.numpy_clpy_array_equal()
    def test_large_int_lower_2(self, xp, dtype):
        a = xp.array([-1], dtype=xp.int8)
        b = xp.iinfo(dtype).min + 1
        return a + b

    @testing.for_int_dtypes(no_bool=True)
    @testing.numpy_clpy_array_equal()
    def test_large_int_lower_3(self, xp, dtype):
        a = xp.array([xp.iinfo(dtype).min], dtype=dtype)
        b = xp.int8(0)
        return a + b

    @testing.for_int_dtypes(no_bool=True)
    @testing.numpy_clpy_array_equal()
    def test_large_int_lower_4(self, xp, dtype):
        a = xp.array([xp.iinfo(dtype).min + 1], dtype=dtype)
        b = xp.int8(-1)
        return a + b


class TestClpyElementwiseKernel(unittest.TestCase):
    def test_vectoradd(self):
        x_np = numpy.array([0.1, 0.2, 0.3], dtype="float32")
        y_np = numpy.array([0.9, 1.8, 1.7], dtype="float32")

        x = clpy.core.array(x_np)
        y = clpy.core.array(y_np)

        kernel = clpy.core.ElementwiseKernel(
            'T x, T y',
            'T z',
            '''
                z = x + y;
            ''',
            'vectoradd')
        z = kernel(x, y)

        actual = z.get()
        expected = x_np + y_np
        self.assertTrue(numpy.allclose(actual, expected))

    def test_vectoraddScalar(self):
        x_np = numpy.array([0.1, 0.2, 0.3], dtype="float32")
        y_np = numpy.float32(1.0)

        x = clpy.core.array(x_np)
        y = y_np

        kernel = clpy.core.ElementwiseKernel(
            'T x, T y',
            'T z',
            '''
                z = x + y;
            ''',
            'vectoradd')
        z = kernel(x, y)

        actual = z.get()
        expected = x_np + y_np
        self.assertTrue(numpy.allclose(actual, expected))

    def test_vectoradd_int64(self):
        x_np = numpy.array([1,  2,  3], dtype="int64")
        y_np = numpy.array([9, 18, 17], dtype="int64")

        x = clpy.core.array(x_np)
        y = clpy.core.array(y_np)

        kernel = clpy.core.ElementwiseKernel(
            'T x, T y',
            'T z',
            '''
                z = x + y;
            ''',
            'vectoradd')
        z = kernel(x, y)

        actual = z.get()
        expected = x_np + y_np
        self.assertTrue(numpy.allclose(actual, expected))

    def test_elementwise3(self):
        a_np = numpy.array([0.1, 0.2, 0.3], dtype="float32")
        b_np = numpy.array([0.9, 1.8, 1.7], dtype="float32")
        c_np = numpy.array([0.2, 1.2, 2.2], dtype="float32")

        a = clpy.core.array(a_np)
        b = clpy.core.array(b_np)
        c = clpy.core.array(c_np)

        kernel = clpy.core.ElementwiseKernel(
            'T a, T b, T c',
            'T z',
            '''
                z = 2*a + 0.1*b / (c*c);
            ''',
            'elementwise3')
        z = kernel(a, b, c)

        actual = z.get()
        expected = 2 * a_np + 0.1 * b_np / (c_np * c_np)
        self.assertTrue(numpy.allclose(actual, expected))

    def test_ufunc(self):
        a_np = numpy.array([0.1, 0.2, 0.3], dtype="float32")

        a = clpy.core.array(a_np)

        kernel = clpy.core.create_ufunc(
            'negative_float32', [('f->f', 'out0 = -in0')])
        z = kernel(a)

        actual = z.get()
        expected = -a_np
        self.assertTrue(numpy.allclose(actual, expected))

    def test_vectoradd_raw(self):
        x_np = numpy.array([1.1,  2.1,  3.1], dtype="float64")
        y_np = numpy.array([9.1, 18.1, 17.1], dtype="float64")

        x = clpy.core.array(x_np)
        y = clpy.core.array(y_np)

        kernel = clpy.core.ElementwiseKernel(
            'raw T x, T y',
            'T z',
            '''
                z = x[i] + y;
            ''',
            'vectoradd')
        z = kernel(x, y)

        actual = z.get()
        expected = x_np + y_np
        self.assertTrue(numpy.allclose(actual, expected))

    def test_vectoradd_size(self):
        x_np = numpy.array([1.1,  2.1,  3.1], dtype="float64")
        y_np = numpy.array([9.1, 18.1, 17.1], dtype="float64")

        x = clpy.core.array(x_np)
        y = clpy.core.array(y_np)

        kernel = clpy.core.ElementwiseKernel(
            'raw T x, T y',
            'T z',
            '''
                z = x[i] + y * T(_ind.size())
            ''',
            'vectoradd')
        z = kernel(x, y)

        actual = z.get()
        expected = x_np + y_np * len(y_np)
        self.assertTrue(numpy.allclose(actual, expected))

    def test_access_raw_1d(self):
        dtype = numpy.float32
        npx = numpy.array([0, 1, 2, 3,  4,  5], dtype=dtype)
        npy = numpy.array([6, 7, 8, 9, 10, 11], dtype=dtype)

        cpx = clpy.array(npx)
        cpy = clpy.array(npy)

        kernel = clpy.core.ElementwiseKernel(
            'raw T x, T y',
            'T z',
            '''
                z = x[i] + y;
            ''',
            'vector_add')
        cpz = kernel(cpx, cpy)

        actual = cpz.get()
        expected = npx + npy

        self.assertTrue(numpy.allclose(actual, expected))

    def test_access_raw_2d(self):
        column = 3
        dtype = numpy.float32
        npx = numpy.array([[0, 1, 2], [3,  4,  5]], dtype=dtype)
        npy = numpy.array([[6, 7, 8], [9, 10, 11]], dtype=dtype)

        cpx = clpy.array(npx)
        cpy = clpy.array(npy)

        kernel = clpy.core.ElementwiseKernel(
            'raw T x, T y, int32 column',
            'T z',
            '''
                ptrdiff_t x_ind[] = {i / column, i % column};
                z = x[x_ind] + y;
            ''',
            'matrix_add')
        cpz = kernel(cpx, cpy, column)

        actual = cpz.get()
        expected = npx + npy

        self.assertTrue(numpy.allclose(actual, expected))

    def test_access_raw_3d(self):
        height = 2
        row = 3
        column = 4
        dtype = numpy.float32

        # array([[[ 0,  1,  2,  3],
        #         [ 4,  5,  6,  7],
        #         [ 8,  9, 10, 11]],
        #       [[12, 13, 14, 15],
        #        [16, 17, 18, 19],
        #        [20, 21, 22, 23]]])
        npx = numpy.arange(0, height * row * column,
                           dtype=dtype).reshape((height, row, column))
        npy = numpy.arange(height * row * column, height * row *
                           column * 2,
                           dtype=dtype).reshape((height, row, column))

        cpx = clpy.array(npx)
        cpy = clpy.array(npy)

        kernel = clpy.core.ElementwiseKernel(
            'raw T x, T y, int32 row, int32 column',
            'T z',
            '''
               ptrdiff_t x_ind[] = {i/(row * column),
                                    (i%(row * column))/column,
                                    i%column};
               z = x[x_ind] + y;
            ''',
            'tensor_add')
        cpz = kernel(cpx, cpy, row, column)

        actual = cpz.get()
        expected = npx + npy

        self.assertTrue(numpy.allclose(actual, expected))


class TestClpyElementwiseKernelwithChunk(unittest.TestCase):
    """test class of EelementwiseKernel with Chunk"""

    def setUp(self):
        # create chunk and free to prepare chunk in pool
        self.pool = clpy.backend.memory.SingleDeviceMemoryPool()
        clpy.backend.memory.set_allocator(self.pool.malloc)
        self.pooled_chunk_size = clpy.backend.memory.subbuffer_alignment * 2
        self.tmp = self.pool.malloc(self.pooled_chunk_size)
        self.pool.free(self.tmp.buf, self.pooled_chunk_size, 0)

    def tearDown(self):
        clpy.backend.memory.set_allocator()

    def test_vectoradd_raw(self):
        dummy_size = 3
        dtype = numpy.float32
        x_np = numpy.array([1.1,  2.1,  3.1], dtype=dtype)
        y_np = numpy.array([9.1, 18.1, 17.1], dtype=dtype)

        # get chunk with offset = 0
        dummy = clpy.empty(dummy_size, dtype)
        self.assertTrue(dummy.data.offset == 0)

        # fill dummy to detect error
        dummy.fill(0)

        # get chunk with offset != 0
        x = clpy.core.array(x_np)
        self.assertTrue(x.data.mem.buf ==
                        dummy.data.mem.buf and x.data.mem.offset != 0)

        # x and y are different chunks
        y = clpy.core.array(y_np)
        self.assertTrue(x.data.mem != y.data.mem and y.data.mem.offset == 0)

        kernel = clpy.core.ElementwiseKernel(
            'raw T x, T y',
            'T z',
            '''
                z = x[i] + y;
            ''',
            'vectoradd')
        z = kernel(x, y)

        actual = z.get()
        expected = x_np + y_np

        self.assertTrue(numpy.allclose(actual, expected))

    def test_vectoradd_2_raw_array(self):
        dummy_size = 3
        dtype = numpy.float32
        x_np = numpy.array([1.1,  2.1,  3.1], dtype=dtype)
        y_np = numpy.array([9.1, 18.1, 17.1], dtype=dtype)
        z_np = numpy.array([20.1, 21.1, 22.1], dtype=dtype)

        # get chunk with offset = 0
        dummy = clpy.empty(dummy_size, dtype)
        self.assertTrue(dummy.data.offset == 0)

        # fill dummy to detect error
        dummy.fill(0)

        # get chunk with offset != 0
        x = clpy.core.array(x_np)
        self.assertTrue(x.data.mem.buf ==
                        dummy.data.mem.buf and x.data.mem.offset != 0)

        # create another chunk
        self.tmp2 = self.pool.malloc(self.pooled_chunk_size)
        self.pool.free(self.tmp2.buf, self.pooled_chunk_size, 0)

        dummy2 = clpy.empty(dummy_size, dtype)
        self.assertTrue(dummy2.data.offset == 0)

        # fill dummy to detect error
        dummy2.fill(0)

        # get chunk with offset != 0
        y = clpy.core.array(y_np)
        self.assertTrue(dummy2.data.mem.buf ==
                        y.data.mem.buf and y.data.mem.offset != 0)

        # chunk with offset == 0
        z = clpy.core.array(z_np)
        self.assertTrue(z.data.mem.offset == 0)

        kernel = clpy.core.ElementwiseKernel(
            'raw T x, raw T y, T z',
            'T w',
            '''
                w = x[i] + y[i] + z;
            ''',
            'vectoradd')
        w = kernel(x, y, z)

        actual = w.get()
        expected = x_np + y_np + z_np

        print(actual)
        print(expected)

        self.assertTrue(numpy.allclose(actual, expected))

    def test_vectoradd_raw_different_index(self):
        dummy_size = 3
        size = 3
        dtype = numpy.float32
        x_np = numpy.array([1.1,  2.1,  3.1], dtype=dtype)
        y_np = numpy.array([9.1, 18.1, 17.1], dtype=dtype)

        # get chunk with offset = 0
        dummy = clpy.empty(dummy_size * 2, dtype)
        self.assertTrue(dummy.data.offset == 0)

        # fill dummy to detect error
        dummy.fill(0)

        # get chunk with offset != 0
        x = clpy.core.array(x_np)
        self.assertTrue(x.data.mem.buf ==
                        dummy.data.mem.buf and x.data.mem.offset != 0)

        y = clpy.core.array(y_np)

        kernel = clpy.core.ElementwiseKernel(
            'raw T x, T y, uint32 n',
            'T z',
            '''
                z = x[i] + x[ n-i-1 ] + y;
            ''',
            'vectoradd')
        z = kernel(x, y, size)

        actual = z.get()
        expected = x_np + x_np[::-1] + y_np

        self.assertTrue(numpy.allclose(actual, expected))

    def test_vectoradd_raw_output(self):
        dummy_size = 3
        dtype = numpy.float32
        x_np = numpy.array([1.1,  2.1,  3.1], dtype=dtype)
        y_np = numpy.array([9.1, 18.1, 17.1], dtype=dtype)
        z_np = numpy.empty(dummy_size, dtype=dtype)

        # get chunk with offset = 0
        dummy = clpy.empty(dummy_size, dtype)
        self.assertTrue(dummy.data.offset == 0)

        # fill dummy to detect error
        dummy.fill(0)

        z = clpy.core.array(z_np)
        self.assertTrue(z.data.mem.buf ==
                        dummy.data.mem.buf and z.data.mem.offset != 0)

        x = clpy.core.array(x_np)
        y = clpy.core.array(y_np)

        kernel = clpy.core.ElementwiseKernel(
            'T x, T y',
            'raw T z',
            '''
                z[i] = x + y;
            ''',
            'vectoradd')
        kernel(x, y, z)

        actual = z.get()
        expected = x_np + y_np

        self.assertTrue(numpy.allclose(actual, expected))

    def test_one_line_operation(self):
        dummy_size = 3
        dtype = numpy.float32
        size = 3
        x_np = numpy.array([1.1,  2.1,  3.1], dtype=dtype)
        y_np = numpy.array([9.1, 18.1, 17.1], dtype=dtype)

        # get chunk with offset = 0
        dummy = clpy.empty(dummy_size, dtype)
        self.assertTrue(dummy.data.offset == 0)

        # fill dummy to detect error
        dummy.fill(0)

        # get chunk with offset != 0
        x = clpy.core.array(x_np)
        self.assertTrue(x.data.mem.buf ==
                        dummy.data.mem.buf and x.data.mem.offset != 0)

        # x and y are different chunks
        y = clpy.core.array(y_np)
        self.assertTrue(x.data.mem != y.data.mem and y.data.mem.offset == 0)

        kernel = clpy.core.ElementwiseKernel(
            'raw T x, T y, int32 n',
            'T z',
            "size_t x_index = n-i-1;"  # first operation
            "z = x[x_index] * y;",     # second operation
            'vector_mul_add')
        z = kernel(x, y, size)

        actual = z.get()
        expected = numpy.flip(x_np, -1) * y_np

        self.assertTrue(numpy.allclose(actual, expected))

    def test_invalid_array_access(self):
        dummy_size = 3
        dtype = numpy.float32
        x_np = numpy.array([1.1,  2.1,  3.1], dtype=dtype)
        y_np = numpy.array([9.1, 18.1, 17.1], dtype=dtype)

        # get chunk with offset = 0
        dummy = clpy.empty(dummy_size, dtype)
        self.assertTrue(dummy.data.offset == 0)

        # fill dummy to detect error
        dummy.fill(0)

        # get chunk with offset != 0
        x = clpy.core.array(x_np)
        self.assertTrue(x.data.mem.buf ==
                        dummy.data.mem.buf and x.data.mem.offset != 0)

        # x and y are different chunks
        y = clpy.core.array(y_np)
        self.assertTrue(x.data.mem != y.data.mem and y.data.mem.offset == 0)

        with self.assertRaises(RuntimeError):
            clpy.core.ElementwiseKernel(
                'raw T x',
                'T z',
                "z = x[i * y;",     # second operation
                'vector_mul_add')


@testing.gpu
class TestElementwiseRaiseExceptions(unittest.TestCase):

    def test_undeclared_identifier(self):
        with six.assertRaisesRegex(self, UltimaRuntimeError,
                                   'undeclared identifier'):
            x = clpy.core.array(numpy.array([1], dtype="float32"))
            clpy.ElementwiseKernel(
                'T x',
                '',
                'undeclared_identifier',
                'use_of_undeclared_indentifier')(x)

    def test_assign_to_const_qualified_variable(self):
        with six.assertRaisesRegex(self, OpenCLProgramBuildError,
                                   'cannot assign|is not assignable'):
            x = clpy.core.array(numpy.array([1], dtype="float32"))
            clpy.ElementwiseKernel(
                'T x',
                'T y',
                'x = y',
                'assign_to_const_qualified_variable')(x)


if __name__ == "__main__":
    unittest.main()
