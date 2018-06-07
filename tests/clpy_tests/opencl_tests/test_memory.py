# -*- coding: utf-8 -*-

import unittest

import numpy

import clpy
import clpy.testing


class TestSingleDeviceMemoryPool(unittest.TestCase):
    """test class of SingleDeviceMemoryPool"""

    def setUp(self):
        self.pool = clpy.backend.memory.SingleDeviceMemoryPool()

    def test_malloc(self):
        p = self.pool.malloc(1)
        self.assertFalse(p.buf.isNull())

    def test_scalar(self):
        for type in [numpy.int8, numpy.int16, numpy.int32, numpy.int64,
                     numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64,
                     numpy.float32, numpy.float64]:
            expected = type(12.5)
            buf = self.pool.malloc(expected.nbytes)

            host_buf = numpy.array(expected)
            hostptr = host_buf.ctypes.get_as_parameter().value
            clpy.testing.writebuf(buffer_to_write=buf.buf,
                                  n_bytes=expected.nbytes,
                                  host_ptr=hostptr)

            actual = numpy.empty(1, dtype=type)
            ahostptr = actual.ctypes.get_as_parameter().value
            clpy.testing.readbuf(buffer_to_read=buf.buf,
                                 n_bytes=expected.nbytes,
                                 host_ptr=ahostptr)

            self.assertEqual(expected, actual[0])

    def test_vector(self):
        for type in [numpy.int8, numpy.int16, numpy.int32, numpy.int64,
                     numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64,
                     numpy.float32, numpy.float64]:
            expected = numpy.array([12.5, 13.4, 14.3, 15.2, 16.1], dtype=type)
            buf = self.pool.malloc(expected.nbytes)

            hostptr = expected.ctypes.get_as_parameter().value
            clpy.testing.writebuf(buffer_to_write=buf.buf,
                                  n_bytes=expected.nbytes,
                                  host_ptr=hostptr)

            actual = numpy.empty(5, dtype=type)
            ahostptr = actual.ctypes.get_as_parameter().value
            clpy.testing.readbuf(buffer_to_read=buf.buf, n_bytes=actual.nbytes,
                                 host_ptr=ahostptr)

            self.assertTrue(numpy.all(expected == actual))

    def test_no_melt(self):
        expected0 = numpy.array(
            [[1.234, 56.78], [1.234, 56.78]], dtype="float64")
        expected1 = numpy.array(
            [[10234, 56078], [10234, 56078]], dtype="uint32")

        buf0 = self.pool.malloc(expected0.nbytes)
        buf1 = self.pool.malloc(expected1.nbytes)

        hostptr0 = expected0.ctypes.get_as_parameter().value
        hostptr1 = expected1.ctypes.get_as_parameter().value
        clpy.testing.writebuf(buffer_to_write=buf0.buf,
                              n_bytes=expected0.nbytes,
                              host_ptr=hostptr0)
        clpy.testing.writebuf(buffer_to_write=buf1.buf,
                              n_bytes=expected1.nbytes,
                              host_ptr=hostptr1)

        actual0 = numpy.empty(expected0.shape, dtype=expected0.dtype)
        actual1 = numpy.empty(expected1.shape, dtype=expected1.dtype)
        clpy.testing.readbuf(buffer_to_read=buf0.buf, n_bytes=actual0.nbytes,
                             host_ptr=actual0.ctypes.get_as_parameter().value)
        clpy.testing.readbuf(buffer_to_read=buf1.buf, n_bytes=actual1.nbytes,
                             host_ptr=actual1.ctypes.get_as_parameter().value)

        self.assertTrue(numpy.all(expected0 == actual0))
        self.assertTrue(numpy.all(expected1 == actual1))


class TestMemoryPointer(unittest.TestCase):
    """test class of MemoryPointer"""

    def test_offset_read(self):
        count = 128
        totaldata = numpy.arange(count, dtype="uint64")
        mem = clpy.backend.Memory(totaldata.nbytes)
        hostptr = totaldata.ctypes.get_as_parameter().value
        clpy.testing.writebuf(buffer_to_write=mem.buf,
                              n_bytes=totaldata.nbytes,
                              host_ptr=hostptr)

        itemsize = totaldata.itemsize

        for offset in range(0, count, 32):
            ptr = clpy.backend.MemoryPointer(mem, totaldata.itemsize * offset)

            actual = numpy.empty(count - offset, totaldata.dtype)
            hostptr = actual.ctypes.get_as_parameter().value
            clpy.testing.readbuf(buffer_to_read=ptr.buf,
                                 offset=offset * itemsize,
                                 n_bytes=actual.nbytes, host_ptr=hostptr)

            expected = totaldata[offset:]
            self.assertTrue(numpy.all(actual == expected))

    def test_offset_write(self):
        count = 256
        step = 32
        expected = numpy.empty(count, dtype="uint64")
        mem = clpy.backend.Memory(expected.nbytes)

        for offset in range(0, count // 2, step):
            ptr = clpy.backend.MemoryPointer(mem, expected.itemsize * offset)
            val = offset + 1

            data = numpy.array([val] * (count - offset * 2), expected.dtype)
            hostptr = data.ctypes.get_as_parameter().value
            clpy.testing.writebuf(buffer_to_write=ptr.buf, offset=ptr.offset,
                                  n_bytes=data.nbytes,
                                  host_ptr=hostptr)

            expected[offset:offset + step] = val
            if offset == 0:
                expected[-step:] = val
            else:
                expected[-offset - step:-offset] = val

        actual = numpy.empty(count, dtype="uint64")
        clpy.testing.readbuf(buffer_to_read=mem.buf, n_bytes=actual.nbytes,
                             host_ptr=actual.ctypes.get_as_parameter().value)
        self.assertTrue(numpy.all(actual == expected))

    def test_add_read(self):
        count = 128
        step = 32
        totaldata = numpy.arange(count, dtype="uint64")
        mem = clpy.backend.Memory(totaldata.nbytes)
        hostptr = totaldata.ctypes.get_as_parameter().value
        clpy.testing.writebuf(buffer_to_write=mem.buf,
                              n_bytes=totaldata.nbytes,
                              host_ptr=hostptr)

        ptr = clpy.backend.MemoryPointer(mem, 0)
        for offset in range(0, count, step):
            actual = numpy.empty(count - offset, totaldata.dtype)
            hostptr = actual.ctypes.get_as_parameter().value
            clpy.testing.readbuf(buffer_to_read=ptr.buf, offset=ptr.offset,
                                 n_bytes=actual.nbytes,
                                 host_ptr=hostptr)

            expected = totaldata[offset:]
            self.assertTrue(numpy.all(actual == expected))

            ptr = ptr + totaldata.itemsize * step

    def test_add_write(self):
        count = 256
        step = 32
        expected = numpy.empty(count, dtype="uint64")
        mem = clpy.backend.Memory(expected.nbytes)

        ptr = clpy.backend.MemoryPointer(mem, 0)
        for offset in range(0, count // 2, step):
            val = offset + 1

            data = numpy.array([val] * (count - offset * 2), expected.dtype)
            hostptr = data.ctypes.get_as_parameter().value
            clpy.testing.writebuf(buffer_to_write=ptr.buf, offset=ptr.offset,
                                  n_bytes=data.nbytes,
                                  host_ptr=hostptr)

            expected[offset:offset + step] = val
            if offset == 0:
                expected[-step:] = val
            else:
                expected[-offset - step:-offset] = val
            ptr = ptr + expected.itemsize * step

        actual = numpy.empty(count, dtype="uint64")
        clpy.testing.readbuf(buffer_to_read=mem.buf, n_bytes=actual.nbytes,
                             host_ptr=actual.ctypes.get_as_parameter().value)
        self.assertTrue(numpy.all(actual == expected))


class TestSingleDeviceMemoryPoolwithChunk(unittest.TestCase):
    """test class of SingleDeviceMemoryPool"""

    def setUp(self):
        # create chunk and free to prepare chunk in pool
        self.pool = clpy.backend.memory.SingleDeviceMemoryPool()
        clpy.backend.memory.set_allocator(self.pool.malloc)
        self.pooled_chunk_size = clpy.backend.memory.subbuffer_alignment * 2
        self.tmp = self.pool.malloc(self.pooled_chunk_size)
        self.pool.free(self.tmp.buf, self.pooled_chunk_size, 0)

    def tearDown(self):
        clpy.backend.memory.set_allocator()

    def test_chunk_copy_from_host(self):
        size = 2
        dtype = numpy.float32
        wrong_value = 0
        correct_value = 1

        dummy = clpy.empty(size, dtype)
        dummy.fill(wrong_value)

        # check offset != 0
        clpy_array = clpy.empty(size, dtype)
        self.assertTrue(clpy_array.data.mem.offset != 0)

        # write wrong_value to clpy_array
        tmp = numpy.full(shape=size, fill_value=wrong_value, dtype=dtype)
        nbytes = tmp.nbytes
        ptr = tmp.ctypes.get_as_parameter().value
        clpy.testing.writebuf(
            buffer_to_write=clpy_array.data.buf,
            # MemoryPointer.offset + Chunk.offset
            offset=clpy_array.data.offset + clpy_array.data.mem.offset,
            n_bytes=nbytes,
            host_ptr=ptr,
        )

        # write correct_value to clpy array by copy_from_host
        expected = numpy.full(
            shape=size, fill_value=correct_value, dtype=dtype)
        clpy_array.data.copy_from_host(
            expected.ctypes.get_as_parameter(), clpy_array.nbytes)

        # read clpy_array
        actual = numpy.empty(shape=size, dtype=dtype)
        nbytes = actual.nbytes
        ptr = actual.ctypes.get_as_parameter().value
        clpy.testing.readbuf(
            buffer_to_read=clpy_array.data.buf,
            # MemoryPointer offset + Chunked offset
            offset=clpy_array.data.offset + clpy_array.data.mem.offset,
            n_bytes=nbytes,
            host_ptr=ptr,
        )

        self.assertTrue(numpy.allclose(actual, expected))

    def test_chunk_copy_to_host(self):
        size = 2
        dtype = numpy.float32
        wrong_value = 0
        correct_value = 1

        dummy = clpy.empty(size, dtype)
        dummy.fill(wrong_value)

        # chunk offset != 0
        clpy_array = clpy.empty(size, dtype)
        self.assertTrue(clpy_array.data.mem.offset != 0)

        # write correct_value to clpy_array
        expected = numpy.full(
            shape=size, fill_value=correct_value, dtype=dtype)
        nbytes = expected.nbytes
        ptr = expected.ctypes.get_as_parameter().value
        clpy.testing.writebuf(
            buffer_to_write=clpy_array.data.buf,
            # MemoryPointer.offset + Chunk.offset
            offset=clpy_array.data.offset + clpy_array.data.mem.offset,
            n_bytes=nbytes,
            host_ptr=ptr,
        )

        # read clpy_array to ptr by copy_to_host
        actual = numpy.full(shape=size, fill_value=wrong_value, dtype=dtype)
        ptr = actual.ctypes.get_as_parameter()
        nbytes = actual.nbytes
        clpy_array.data.copy_to_host(ptr, nbytes)

        self.assertTrue(numpy.allclose(actual, expected))

    def test_chunk_function(self):
        size = 2
        dtype = numpy.float32
        wrong_value = 0
        correct_value = 1

        dummy = clpy.empty(size, dtype)
        dummy.fill(wrong_value)

        # chunk offset != 0
        clpy_array = clpy.empty(size, dtype)
        self.assertTrue(clpy_array.data.mem.offset != 0)

        # write wrong_value to clpy_array
        tmp = numpy.full(shape=size, fill_value=wrong_value, dtype=dtype)
        nbytes = tmp.nbytes
        ptr = tmp.ctypes.get_as_parameter().value
        clpy.testing.writebuf(
            buffer_to_write=clpy_array.data.buf,
            # MemoryPointer.offset + Chunk.offset
            offset=clpy_array.data.offset + clpy_array.data.mem.offset,
            n_bytes=nbytes,
            host_ptr=ptr,
        )

        # write correct_value to clpy_array by fill (function.pyx)
        clpy_array.fill(correct_value)

        # read clpy_array
        actual = numpy.empty(shape=size, dtype=dtype)
        nbytes = actual.nbytes
        ptr = actual.ctypes.get_as_parameter().value
        clpy.testing.readbuf(
            buffer_to_read=clpy_array.data.buf,
            # MemoryPointer offset + Chunked offset
            offset=clpy_array.data.offset + clpy_array.data.mem.offset,
            n_bytes=nbytes,
            host_ptr=ptr,
        )

        expected = numpy.full(size, fill_value=correct_value, dtype=dtype)

        self.assertTrue(numpy.allclose(actual, expected))

    def test_chunk_copy_from_device_src(self):
        size = 2
        dtype = numpy.float32
        wrong_value = 0
        correct_value = 1

        dummy = clpy.empty(size, dtype)
        dummy.fill(wrong_value)

        # chunk offset != 0
        src_array = clpy.empty(size, dtype)
        self.assertTrue(src_array.data.mem.offset != 0)

        # dst_array should be different Chunk from src_array
        # to avoid CL_MEM_COPY_OVERLAP
        # chunk offset == 0
        dst_array = clpy.empty(self.pooled_chunk_size, dtype)
        self.assertTrue(dst_array.data.mem.offset == 0)
        dst_array.fill(wrong_value)

        # write correct_value to src_array
        expected = numpy.full(
            shape=size, fill_value=correct_value, dtype=dtype)
        nbytes = expected.nbytes
        ptr = expected.ctypes.get_as_parameter().value
        clpy.testing.writebuf(
            buffer_to_write=src_array.data.buf,
            # MemoryPointer.offset + Chunk.offset
            offset=src_array.data.offset + src_array.data.mem.offset,
            n_bytes=nbytes,
            host_ptr=ptr,
        )

        # copy src_array to dst_array by copy_from_device
        dst_array.data.copy_from_device(src_array.data, src_array.nbytes)

        actual = dst_array.get()[0:2]

        self.assertTrue(numpy.allclose(actual, expected))

    def test_chunk_copy_from_device_dst(self):
        size = 2
        dtype = numpy.float32
        wrong_value = 0
        correct_value = 1

        dummy = clpy.empty(size, dtype)
        dummy.fill(wrong_value)

        # chunk offset != 0
        dst_array = clpy.empty(size, dtype)
        self.assertTrue(dst_array.data.mem.offset != 0)

        # write wrong_value to dst_array
        tmp = numpy.full(shape=size, fill_value=wrong_value, dtype=dtype)
        nbytes = tmp.nbytes
        ptr = tmp.ctypes.get_as_parameter().value
        clpy.testing.writebuf(
            buffer_to_write=dst_array.data.buf,
            # MemoryPointer.offset + Chunk.offset
            offset=dst_array.data.offset + dst_array.data.mem.offset,
            n_bytes=nbytes,
            host_ptr=ptr,
        )

        # src_array should be different Chunk from dst_array
        # to avoid CL_MEM_COPY_OVERLAP
        # chunk with offset == 0
        src_array = clpy.empty(self.pooled_chunk_size, dtype)
        self.assertTrue(src_array.data.mem.offset == 0)
        src_array.fill(correct_value)

        # copy src_array to dst_array by copy_from_device
        dst_array.data.copy_from_device(src_array.data, dst_array.nbytes)

        # read dst_array
        actual = numpy.empty(shape=size, dtype=dtype)
        nbytes = actual.nbytes
        ptr = actual.ctypes.get_as_parameter().value
        clpy.testing.readbuf(
            buffer_to_read=dst_array.data.buf,
            # MemoryPointer offset + Chunked offset
            offset=dst_array.data.offset + dst_array.data.mem.offset,
            n_bytes=nbytes,
            host_ptr=ptr,
        )

        expected = src_array.get()[0:2]

        self.assertTrue(numpy.allclose(actual, expected))


if __name__ == "__main__":
    unittest.main()
