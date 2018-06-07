import unittest

import numpy

import clpy


class TestCArraywithChunk(unittest.TestCase):
    """test class of Carray with Chunk"""

    def setUp(self):
        # create chunk and free to prepare chunk in pool
        self.pool = clpy.backend.memory.SingleDeviceMemoryPool()
        clpy.backend.memory.set_allocator(self.pool.malloc)
        self.pooled_chunk_size = clpy.backend.memory.subbuffer_alignment * 2
        self.tmp = self.pool.malloc(self.pooled_chunk_size)
        self.pool.free(self.tmp.buf, self.pooled_chunk_size, 0)

    def tearDown(self):
        clpy.backend.memory.set_allocator()

    def test_get_CArrayIndexI_2(self):
        dummy_size = 3
        dtype = numpy.int32
        x_np = numpy.array([[1, 2], [3, 4]], dtype=dtype)
        y_np = numpy.array([[5, 6], [7, 8]], dtype=dtype)

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

        # y is used as dummy value to compile kernel
        kernel = clpy.core.ElementwiseKernel(
            'raw T x, T y',
            'T z',
            '''
                z = get_CArrayIndexI_2(&x_info, i)
            ''',
            'vectoradd')
        z = kernel(x, y)

        actual = z.get()
        expected = numpy.zeros(actual.shape, dtype=dtype)
        for i in range(expected.shape[0]):
            for j in range(expected.shape[1]):
                # data.mem.offset divided by size of int32 (= 4 byte)
                expected[i][j] = x.data.mem.offset / \
                    4 + i * expected.shape[1] + j

        self.assertTrue(numpy.allclose(actual, expected))


if __name__ == "__main__":
    unittest.main()
