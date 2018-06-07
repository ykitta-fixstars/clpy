# -*- coding: utf-8 -*-
import unittest

import numpy

import clpy
from clpy.backend.opencl import blas
from clpy.core import core


class TestBlas3Sgemm(unittest.TestCase):
    """test class of ClPy's BLAS-3 SGEMM function"""

    def test_row_matrix_row_matrix(self):
        npA = numpy.array([
            [1, 2, 3],
            [4, 5, 6]], dtype='float32')  # row-major
        npB = numpy.array([
            [10, 11],
            [13, 14],
            [16, 17]], dtype='float32')  # row-major
        transa = 0  # A is not transposed in c-style(row-major)
        transb = 0  # B is not transposed in c-style(row-major)

        m = npA.shape[0]  # op(A) rows = (A in row-major) rows = C rows
        n = npB.shape[1]  # op(B) cols = (B in row-major) cols = C cols
        k = npA.shape[1]  # op(A) cols = (A in row-major) cols = op(B) rows

        clpA = clpy.ndarray(npA.shape, dtype=numpy.dtype('float32'))
        clpA.set(npA)
        clpB = clpy.ndarray(npB.shape, dtype=numpy.dtype('float32'))
        clpB.set(npB)
        clpC = clpy.ndarray((n, m), dtype=numpy.dtype(
            'float32'))  # column-major

        clpA, transa, lda = core._mat_to_cublas_contiguous(
            clpA, transa)  # as cublas-style
        clpB, transb, ldb = core._mat_to_cublas_contiguous(
            clpB, transb)  # as cublas-style
        ldc = clpC.shape[1]

        blas.sgemm(transa, transb, m, n, k,
                   1.0, clpA, lda,
                   clpB, ldb,
                   0.0, clpC, ldc
                   )

        actualC = clpC.get().T  # as row-major
        expectedC = numpy.dot(npA, npB)  # row-major caluculation
        self.assertTrue(numpy.allclose(expectedC, actualC))

    def test_row_matrix_row_vector(self):
        npA = numpy.array([
            [1, 2],
            [4, 5],
            [7, 8]], dtype='float32')  # row-major
        npB = numpy.array([
            [10],
            [13]], dtype='float32')  # row-major
        transa = 0  # A is not transposed in c-style(row-major)
        transb = 0  # B is not transposed in c-style(row-major)

        m = npA.shape[0]  # op(A) rows = (A in row-major) rows = C rows
        n = npB.shape[1]  # op(B) cols = (B in row-major) cols = C cols
        k = npA.shape[1]  # op(A) cols = (A in row-major) cols = op(B) rows

        clpA = clpy.ndarray(npA.shape, dtype=numpy.dtype('float32'))
        clpA.set(npA)
        clpB = clpy.ndarray(npB.shape, dtype=numpy.dtype('float32'))
        clpB.set(npB)
        clpC = clpy.ndarray((n, m), dtype=numpy.dtype(
            'float32'))  # column-major

        clpA, transa, lda = core._mat_to_cublas_contiguous(
            clpA, transa)  # as cublas-style
        clpB, transb, ldb = core._mat_to_cublas_contiguous(
            clpB, transb)  # as cublas-style
        ldc = clpC.shape[1]

        blas.sgemm(transa, transb, m, n, k,
                   1.0, clpA, lda,
                   clpB, ldb,
                   0.0, clpC, ldc
                   )

        actualC = clpC.get().T  # as row-major
        expectedC = numpy.dot(npA, npB)  # row-major caluculation
        self.assertTrue(numpy.allclose(expectedC, actualC))
        m = npA.shape[0]  # op(A) rows = (A in row-major) rows = C rows

    def test_row_vector_row_matrix(self):
        npA = numpy.array([
            [10, 13, 16]
        ], dtype='float32')  # row-major
        npB = numpy.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]], dtype='float32')  # row-major
        transa = 0  # A is not transposed in c-style(row-major)
        transb = 0  # B is not transposed in c-style(row-major)

        m = npA.shape[0]  # op(A) rows = (A in row-major) rows = C rows
        n = npB.shape[1]  # op(B) cols = (B in row-major) cols = C cols
        k = npA.shape[1]  # op(A) cols = (A in row-major) cols = op(B) rows

        clpA = clpy.ndarray(npA.shape, dtype=numpy.dtype('float32'))
        clpA.set(npA)
        clpB = clpy.ndarray(npB.shape, dtype=numpy.dtype('float32'))
        clpB.set(npB)
        clpC = clpy.ndarray((n, m), dtype=numpy.dtype(
            'float32'))  # column-major

        clpA, transa, lda = core._mat_to_cublas_contiguous(
            clpA, transa)  # as cublas-style
        clpB, transb, ldb = core._mat_to_cublas_contiguous(
            clpB, transb)  # as cublas-style
        ldc = clpC.shape[1]

        blas.sgemm(transa, transb, m, n, k,
                   1.0, clpA, lda,
                   clpB, ldb,
                   0.0, clpC, ldc
                   )

        actualC = clpC.get().T  # as row-major
        expectedC = numpy.dot(npA, npB)  # row-major caluculation
        self.assertTrue(numpy.allclose(expectedC, actualC))

    def test_column_matrix_column_matrix(self):
        npA = numpy.array([
            [1, 2, 3],
            [4, 5, 6]], dtype='float32')  # column-major
        # 1 4
        # 2 5
        # 3 6
        npB = numpy.array([
            [10, 11],
            [13, 14],
            [16, 17]], dtype='float32')  # column-major
        # 10 13 16
        # 11 14 17
        transa = 1  # A is transposed in c-style(row-major)
        transb = 1  # B is transposed in c-style(row-major)

        m = npA.shape[1]  # op(A) rows = (A in column-major) rows = C rows
        n = npB.shape[0]  # op(B) cols = (B in column-major) cols = C cols
        k = npA.shape[0]  # op(A) cols = (A in column-major) cols = op(B) rows

        clpA = clpy.ndarray(npA.shape, dtype=numpy.dtype('float32'))
        clpA.set(npA)
        clpB = clpy.ndarray(npB.shape, dtype=numpy.dtype('float32'))
        clpB.set(npB)
        clpC = clpy.ndarray((n, m), dtype=numpy.dtype(
            'float32'))  # column-major

        clpA, transa, lda = core._mat_to_cublas_contiguous(
            clpA, transa)  # as cublas-style
        clpB, transb, ldb = core._mat_to_cublas_contiguous(
            clpB, transb)  # as cublas-style
        ldc = clpC.shape[1]

        blas.sgemm(transa, transb, m, n, k,
                   1.0, clpA, lda,
                   clpB, ldb,
                   0.0, clpC, ldc
                   )

        actualC = clpC.get().T  # as row-major
        expectedC = numpy.dot(npA.T, npB.T)  # row-major caluculation
        self.assertTrue(numpy.allclose(expectedC, actualC))

    def test_column_matrix_column_vector(self):
        npA = numpy.array([
            [1, 2, 3],
            [4, 5, 6]], dtype='float32')  # column-major
        # 1 4
        # 2 5
        # 3 6
        npB = numpy.array([
            [10, 11]], dtype='float32')  # column-major
        # 10
        # 11
        transa = 1  # A is transposed in c-style(row-major)
        transb = 1  # B is transposed in c-style(row-major)

        m = npA.shape[1]  # op(A) rows = (A in column-major) rows = C rows
        n = npB.shape[0]  # op(B) cols = (B in column-major) cols = C cols
        k = npA.shape[0]  # op(A) cols = (A in column-major) cols = op(B) rows

        clpA = clpy.ndarray(npA.shape, dtype=numpy.dtype('float32'))
        clpA.set(npA)
        clpB = clpy.ndarray(npB.shape, dtype=numpy.dtype('float32'))
        clpB.set(npB)
        clpC = clpy.ndarray((n, m), dtype=numpy.dtype(
            'float32'))  # column-major

        clpA, transa, lda = core._mat_to_cublas_contiguous(
            clpA, transa)  # as cublas-style
        clpB, transb, ldb = core._mat_to_cublas_contiguous(
            clpB, transb)  # as cublas-style
        ldc = clpC.shape[1]

        blas.sgemm(transa, transb, m, n, k,
                   1.0, clpA, lda,
                   clpB, ldb,
                   0.0, clpC, ldc
                   )

        actualC = clpC.get().T  # as row-major
        expectedC = numpy.dot(npA.T, npB.T)  # row-major caluculation
        self.assertTrue(numpy.allclose(expectedC, actualC))

    def test_column_vector_column_matrix(self):
        npA = numpy.array([
            [1],
            [4]], dtype='float32')  # column-major
        # 1 4
        npB = numpy.array([
            [10, 11],
            [13, 14],
            [16, 17]], dtype='float32')  # column-major
        # 10 13 16
        # 11 14 17
        transa = 1  # A is transposed in c-style(row-major)
        transb = 1  # B is transposed in c-style(row-major)

        m = npA.shape[1]  # op(A) rows = (A in column-major) rows = C rows
        n = npB.shape[0]  # op(B) cols = (B in column-major) cols = C cols
        k = npA.shape[0]  # op(A) cols = (A in column-major) cols = op(B) rows

        clpA = clpy.ndarray(npA.shape, dtype=numpy.dtype('float32'))
        clpA.set(npA)
        clpB = clpy.ndarray(npB.shape, dtype=numpy.dtype('float32'))
        clpB.set(npB)
        clpC = clpy.ndarray((n, m), dtype=numpy.dtype(
            'float32'))  # column-major

        clpA, transa, lda = core._mat_to_cublas_contiguous(
            clpA, transa)  # as cublas-style
        clpB, transb, ldb = core._mat_to_cublas_contiguous(
            clpB, transb)  # as cublas-style
        ldc = clpC.shape[1]

        blas.sgemm(transa, transb, m, n, k,
                   1.0, clpA, lda,
                   clpB, ldb,
                   0.0, clpC, ldc
                   )

        actualC = clpC.get().T  # as row-major
        expectedC = numpy.dot(npA.T, npB.T)  # row-major caluculation
        self.assertTrue(numpy.allclose(expectedC, actualC))

    def test_row_matrix_column_matrix(self):
        npA = numpy.array([
            [1, 2],
            [4, 5]], dtype='float32')  # row-major
        # 1 2
        # 4 5
        npB = numpy.array([
            [10, 11],
            [13, 14],
            [16, 17]], dtype='float32')  # column-major
        # 10 13 16
        # 11 14 17
        transa = 0  # A is not transposed in c-style(row-major)
        transb = 1  # B is transposed in c-style(row-major)

        m = npA.shape[0]  # op(A) rows = (A in row-major)    rows = C rows
        n = npB.shape[0]  # op(B) cols = (B in column-major) cols = C cols
        k = npA.shape[1]  # op(A) cols = (A in row-major)    cols = op(B) rows

        clpA = clpy.ndarray(npA.shape, dtype=numpy.dtype('float32'))
        clpA.set(npA)
        clpB = clpy.ndarray(npB.shape, dtype=numpy.dtype('float32'))
        clpB.set(npB)
        clpC = clpy.ndarray((n, m), dtype=numpy.dtype(
            'float32'))  # column-major

        clpA, transa, lda = core._mat_to_cublas_contiguous(
            clpA, transa)  # as cublas-style
        clpB, transb, ldb = core._mat_to_cublas_contiguous(
            clpB, transb)  # as cublas-style
        ldc = clpC.shape[1]

        blas.sgemm(transa, transb, m, n, k,
                   1.0, clpA, lda,
                   clpB, ldb,
                   0.0, clpC, ldc
                   )

        actualC = clpC.get().T  # as row-major
        expectedC = numpy.dot(npA, npB.T)  # row-major caluculation
        self.assertTrue(numpy.allclose(expectedC, actualC))

    def test_row_matrix_column_vector(self):
        npA = numpy.array([
            [1, 2],
            [4, 5]], dtype='float32')  # row-major
        # 1 2
        # 4 5
        npB = numpy.array([
            [10, 11]], dtype='float32')  # column-major
        # 10
        # 11
        transa = 0  # A is not transposed in c-style(row-major)
        transb = 1  # B is transposed in c-style(row-major)

        m = npA.shape[0]  # op(A) rows = (A in row-major)    rows = C rows
        n = npB.shape[0]  # op(B) cols = (B in column-major) cols = C cols
        k = npA.shape[1]  # op(A) cols = (A in row-major)    cols = op(B) rows

        clpA = clpy.ndarray(npA.shape, dtype=numpy.dtype('float32'))
        clpA.set(npA)
        clpB = clpy.ndarray(npB.shape, dtype=numpy.dtype('float32'))
        clpB.set(npB)
        clpC = clpy.ndarray((n, m), dtype=numpy.dtype(
            'float32'))  # column-major

        clpA, transa, lda = core._mat_to_cublas_contiguous(
            clpA, transa)  # as cublas-style
        clpB, transb, ldb = core._mat_to_cublas_contiguous(
            clpB, transb)  # as cublas-style
        ldc = clpC.shape[1]

        blas.sgemm(transa, transb, m, n, k,
                   1.0, clpA, lda,
                   clpB, ldb,
                   0.0, clpC, ldc
                   )

        actualC = clpC.get().T  # as row-major
        expectedC = numpy.dot(npA, npB.T)  # row-major caluculation
        self.assertTrue(numpy.allclose(expectedC, actualC))

    def test_row_vector_column_matrix(self):
        npA = numpy.array([
            [1, 2]], dtype='float32')  # row-major
        # 1 2
        # 4 5
        npB = numpy.array([
            [10, 11],
            [13, 14],
            [16, 17]], dtype='float32')  # column-major
        # 10 13 16
        # 11 14 17
        transa = 0  # A is not transposed in c-style(row-major)
        transb = 1  # B is transposed in c-style(row-major)

        m = npA.shape[0]  # op(A) rows = (A in row-major)    rows = C rows
        n = npB.shape[0]  # op(B) cols = (B in column-major) cols = C cols
        k = npA.shape[1]  # op(A) cols = (A in row-major)    cols = op(B) rows

        clpA = clpy.ndarray(npA.shape, dtype=numpy.dtype('float32'))
        clpA.set(npA)
        clpB = clpy.ndarray(npB.shape, dtype=numpy.dtype('float32'))
        clpB.set(npB)
        clpC = clpy.ndarray((n, m), dtype=numpy.dtype(
            'float32'))  # column-major

        clpA, transa, lda = core._mat_to_cublas_contiguous(
            clpA, transa)  # as cublas-style
        clpB, transb, ldb = core._mat_to_cublas_contiguous(
            clpB, transb)  # as cublas-style
        ldc = clpC.shape[1]

        blas.sgemm(transa, transb, m, n, k,
                   1.0, clpA, lda,
                   clpB, ldb,
                   0.0, clpC, ldc
                   )

        actualC = clpC.get().T  # as row-major
        expectedC = numpy.dot(npA, npB.T)  # row-major caluculation
        self.assertTrue(numpy.allclose(expectedC, actualC))

    def test_column_matrix_row_matrix(self):
        npA = numpy.array([
            [1, 2, 3],
            [4, 5, 6]], dtype='float32')  # column-major
        # 1 4
        # 2 5
        # 3 6
        npB = numpy.array([
            [10, 11],
            [13, 14]], dtype='float32')  # row-major
        # 10 11
        # 13 14
        transa = 1  # A is transposed in c-style(row-major)
        transb = 0  # B is not transposed in c-style(row-major)

        m = npA.shape[1]  # op(A) rows = (A in column-major) rows = C rows
        n = npB.shape[1]  # op(B) cols = (B in row-major)    cols = C cols
        k = npA.shape[0]  # op(A) cols = (A in column-major) cols = op(B) rows

        clpA = clpy.ndarray(npA.shape, dtype=numpy.dtype('float32'))
        clpA.set(npA)
        clpB = clpy.ndarray(npB.shape, dtype=numpy.dtype('float32'))
        clpB.set(npB)
        clpC = clpy.ndarray((n, m), dtype=numpy.dtype(
            'float32'))  # column-major

        clpA, transa, lda = core._mat_to_cublas_contiguous(
            clpA, transa)  # as cublas-style
        clpB, transb, ldb = core._mat_to_cublas_contiguous(
            clpB, transb)  # as cublas-style
        ldc = clpC.shape[1]

        blas.sgemm(transa, transb, m, n, k,
                   1.0, clpA, lda,
                   clpB, ldb,
                   0.0, clpC, ldc
                   )

        actualC = clpC.get().T  # as row-major
        expectedC = numpy.dot(npA.T, npB)  # row-major caluculation
        self.assertTrue(numpy.allclose(expectedC, actualC))

    def test_column_matrix_row_vector(self):
        npA = numpy.array([
            [1, 2, 3],
            [4, 5, 6]], dtype='float32')  # column-major
        # 1 4
        # 2 5
        # 3 6
        npB = numpy.array([
            [10],
            [13]], dtype='float32')  # row-major
        # 10
        # 13
        transa = 1  # A is transposed in c-style(row-major)
        transb = 0  # B is not transposed in c-style(row-major)

        m = npA.shape[1]  # op(A) rows = (A in column-major) rows = C rows
        n = npB.shape[1]  # op(B) cols = (B in row-major)    cols = C cols
        k = npA.shape[0]  # op(A) cols = (A in column-major) cols = op(B) rows

        clpA = clpy.ndarray(npA.shape, dtype=numpy.dtype('float32'))
        clpA.set(npA)
        clpB = clpy.ndarray(npB.shape, dtype=numpy.dtype('float32'))
        clpB.set(npB)
        clpC = clpy.ndarray((n, m), dtype=numpy.dtype(
            'float32'))  # column-major

        clpA, transa, lda = core._mat_to_cublas_contiguous(
            clpA, transa)  # as cublas-style
        clpB, transb, ldb = core._mat_to_cublas_contiguous(
            clpB, transb)  # as cublas-style
        ldc = clpC.shape[1]

        blas.sgemm(transa, transb, m, n, k,
                   1.0, clpA, lda,
                   clpB, ldb,
                   0.0, clpC, ldc
                   )

        actualC = clpC.get().T  # as row-major
        expectedC = numpy.dot(npA.T, npB)  # row-major caluculation
        self.assertTrue(numpy.allclose(expectedC, actualC))

    def test_column_vector_row_matrix(self):
        npA = numpy.array([
            [1],
            [2],
            [3]], dtype='float32')  # column-major
        # 1 2 3
        npB = numpy.array([
            [10, 11],
            [13, 14],
            [16, 17]], dtype='float32')  # row-major
        # 10 11
        # 13 14
        # 16 17
        transa = 1  # A is transposed in c-style(row-major)
        transb = 0  # B is not transposed in c-style(row-major)

        m = npA.shape[1]  # op(A) rows = (A in column-major) rows = C rows
        n = npB.shape[1]  # op(B) cols = (B in row-major)    cols = C cols
        k = npA.shape[0]  # op(A) cols = (A in column-major) cols = op(B) rows

        clpA = clpy.ndarray(npA.shape, dtype=numpy.dtype('float32'))
        clpA.set(npA)
        clpB = clpy.ndarray(npB.shape, dtype=numpy.dtype('float32'))
        clpB.set(npB)
        clpC = clpy.ndarray((n, m), dtype=numpy.dtype(
            'float32'))  # column-major

        clpA, transa, lda = core._mat_to_cublas_contiguous(
            clpA, transa)  # as cublas-style
        clpB, transb, ldb = core._mat_to_cublas_contiguous(
            clpB, transb)  # as cublas-style
        ldc = clpC.shape[1]

        blas.sgemm(transa, transb, m, n, k,
                   1.0, clpA, lda,
                   clpB, ldb,
                   0.0, clpC, ldc
                   )

        actualC = clpC.get().T  # as row-major
        expectedC = numpy.dot(npA.T, npB)  # row-major caluculation
        self.assertTrue(numpy.allclose(expectedC, actualC))

    def test_invalid_transa(self):
        npA = numpy.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype='float32')
        npB = numpy.array([[10, 11, 12], [13, 14, 15], [
                          16, 17, 18]], dtype='float32')

        clpA = clpy.ndarray(npA.shape, dtype=numpy.dtype('float32'))
        clpA.set(npA)
        clpB = clpy.ndarray(npB.shape, dtype=numpy.dtype('float32'))
        clpB.set(npB)

        expectedC = numpy.dot(npA, npB).T
        clpC = clpy.ndarray(expectedC.shape, dtype=numpy.dtype('float32'))

        m = npA.shape[0]
        n = npB.shape[1]
        k = npA.shape[1]
        with self.assertRaises(ValueError):
            blas.sgemm(transa='a', transb='t',
                       m=m, n=n, k=k,
                       alpha=1.0, A=clpA, lda=k,
                       B=clpB, ldb=n,
                       beta=0.0,
                       C=clpC, ldc=m
                       )

    def test_invalid_transb(self):
        npA = numpy.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype='float32')
        npB = numpy.array([[10, 11, 12], [13, 14, 15], [
                          16, 17, 18]], dtype='float32')

        clpA = clpy.ndarray(npA.shape, dtype=numpy.dtype('float32'))
        clpA.set(npA)
        clpB = clpy.ndarray(npB.shape, dtype=numpy.dtype('float32'))
        clpB.set(npB)

        expectedC = numpy.dot(npA, npB).T
        clpC = clpy.ndarray(expectedC.shape, dtype=numpy.dtype('float32'))

        m = npA.shape[0]
        n = npB.shape[1]
        k = npA.shape[1]
        with self.assertRaises(ValueError):
            blas.sgemm(transa='n', transb='a',
                       m=m, n=n, k=k,
                       alpha=1.0, A=clpA, lda=k,
                       B=clpB, ldb=n,
                       beta=0.0,
                       C=clpC, ldc=m
                       )

    def test_alpha_matrix_matrix(self):
        npA = numpy.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                          dtype='float32')  # row-major
        npB = numpy.array([[10, 11, 12], [13, 14, 15], [
                          16, 17, 18]], dtype='float32')  # row-major
        transa = 0  # A is not transposed in c-style(row-major)
        transb = 0  # B is not transposed in c-style(row-major)
        alpha = 2.0
        beta = 0.0

        m = npA.shape[0]
        n = npB.shape[1]
        k = npA.shape[1]

        clpA = clpy.ndarray(npA.shape, dtype=numpy.dtype(
            'float32'))  # col major in clpy
        clpA.set(npA)
        clpB = clpy.ndarray(npB.shape, dtype=numpy.dtype(
            'float32'))  # col major in clpy
        clpB.set(npB)

        expectedC = numpy.dot(npA, npB) * alpha
        clpC = clpy.ndarray(expectedC.shape, dtype=numpy.dtype('float32'))

        clpA, transa, lda = core._mat_to_cublas_contiguous(
            clpA, transa)  # as cublas-style
        clpB, transb, ldb = core._mat_to_cublas_contiguous(
            clpB, transb)  # as cublas-style
        ldc = clpC.shape[1]

        # alpha * (A^t x B^T) in col-major = alpha * AxB in row major
        blas.sgemm(transa, transb,
                   m, n, k, alpha,
                   clpA, lda,
                   clpB, ldb,
                   beta, clpC, ldc
                   )

        actualC = clpC.get().T  # as row-major

        self.assertTrue(numpy.allclose(expectedC, actualC))

    def test_beta_matrix_matrix(self):
        npA = numpy.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                          dtype='float32')  # row-major
        npB = numpy.array([[10, 11, 12], [13, 14, 15], [
                          16, 17, 18]], dtype='float32')  # row-major
        npC = numpy.array([[19, 20, 21], [22, 23, 24], [
                          25, 26, 27]], dtype='float32')  # row-major
        transa = 0  # A is not transposed in c-style(row-major)
        transb = 0  # B is not transposed in c-style(row-major)

        alpha = 1.0
        beta = 2.0

        m = npA.shape[0]
        n = npB.shape[1]
        k = npA.shape[1]

        clpA = clpy.ndarray(npA.shape, dtype=numpy.dtype(
            'float32'))  # col-major in clpy
        clpA.set(npA)
        clpB = clpy.ndarray(npB.shape, dtype=numpy.dtype(
            'float32'))  # col-major in clpy
        clpB.set(npB)

        clpC = clpy.ndarray(npC.shape, dtype=numpy.dtype(
            'float32'))  # col-major in clpy
        clpC.set(npC.T)  # transpose C

        clpA, transa, lda = core._mat_to_cublas_contiguous(
            clpA, transa)  # as cublas-style
        clpB, transb, ldb = core._mat_to_cublas_contiguous(
            clpB, transb)  # as cublas-style
        ldc = clpC.shape[1]

        # AxB + beta*C
        expectedC = numpy.add(numpy.dot(npA, npB), beta * npC)

        # (A^T x B^T) + C^T in col-major = A x B + C in row-major
        blas.sgemm(transa, transb,
                   m, n, k, alpha,
                   clpA, lda,
                   clpB, ldb,
                   beta, clpC, ldc
                   )

        actualC = clpC.get().T  # as row-major

        self.assertTrue(numpy.allclose(expectedC, actualC))

    def test_beta_0_matrix_matrix(self):
        npA = numpy.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                          dtype='float32')  # row-major
        npB = numpy.array([[10, 11, 12], [13, 14, 15], [
                          16, 17, 18]], dtype='float32')  # row-major
        transa = 0  # A is not transposed in c-style(row-major)
        transb = 0  # B is not transposed in c-style(row-major)
        alpha = 1.0
        beta = 0.0

        m = npA.shape[0]
        n = npB.shape[1]
        k = npA.shape[1]

        clpA = clpy.ndarray(npA.shape, dtype=numpy.dtype(
            'float32'))  # col major in clpy
        clpA.set(npA)
        clpB = clpy.ndarray(npB.shape, dtype=numpy.dtype(
            'float32'))  # col major in clpy
        clpB.set(npB)

        expectedC = numpy.dot(npA, npB) * alpha
        clpC = clpy.ndarray(expectedC.shape, dtype=numpy.dtype('float32'))
        clpC.fill(numpy.nan)

        clpA, transa, lda = core._mat_to_cublas_contiguous(
            clpA, transa)  # as cublas-style
        clpB, transb, ldb = core._mat_to_cublas_contiguous(
            clpB, transb)  # as cublas-style
        ldc = clpC.shape[1]

        # alpha * (A^t x B^T) in col-major = alpha * AxB in row major
        blas.sgemm(transa, transb,
                   m, n, k, alpha,
                   clpA, lda,
                   clpB, ldb,
                   beta, clpC, ldc
                   )

        actualC = clpC.get().T  # as row-major

        self.assertTrue(numpy.allclose(expectedC, actualC))

    def test_alpha_0_matrix_matrix(self):
        npC = numpy.array([[19, 20, 21], [22, 23, 24], [
                          25, 26, 27]], dtype='float32')  # row-major

        npA = numpy.ndarray(npC.shape, dtype=numpy.dtype('float32'))
        npA.fill(numpy.nan)
        npB = numpy.ndarray(npC.shape, dtype=numpy.dtype('float32'))
        npB.fill(numpy.nan)
        transa = 0  # A is not transposed in c-style(row-major)
        transb = 0  # B is not transposed in c-style(row-major)

        alpha = 0.0
        beta = 2.0

        m = npA.shape[0]
        n = npB.shape[1]
        k = npA.shape[1]

        clpA = clpy.ndarray(npA.shape, dtype=numpy.dtype(
            'float32'))  # col-major in clpy
        clpA.set(npA)
        clpB = clpy.ndarray(npB.shape, dtype=numpy.dtype(
            'float32'))  # col-major in clpy
        clpB.set(npB)

        clpC = clpy.ndarray(npC.shape, dtype=numpy.dtype(
            'float32'))  # col-major in clpy
        clpC.set(npC.T)  # transpose C

        clpA, transa, lda = core._mat_to_cublas_contiguous(
            clpA, transa)  # as cublas-style
        clpB, transb, ldb = core._mat_to_cublas_contiguous(
            clpB, transb)  # as cublas-style
        ldc = clpC.shape[1]

        # AxB + beta*C
        expectedC = beta * npC

        # (A^T x B^T) + C^T in col-major = A x B + C in row-major
        blas.sgemm(transa, transb,
                   m, n, k, alpha,
                   clpA, lda,
                   clpB, ldb,
                   beta, clpC, ldc
                   )

        actualC = clpC.get().T  # as row-major

        self.assertTrue(numpy.allclose(expectedC, actualC))

    def test_chunk_sgemm_A(self):
        # create chunk and free to prepare chunk in pool
        pool = clpy.backend.memory.SingleDeviceMemoryPool()
        clpy.backend.memory.set_allocator(pool.malloc)
        pooled_chunk_size = clpy.backend.memory.subbuffer_alignment * 2
        tmp = pool.malloc(pooled_chunk_size)
        pool.free(tmp.buf, pooled_chunk_size, 0)

        size = 3
        dtype = numpy.float32
        wrong_value = numpy.nan

        npA = numpy.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=dtype)
        npB = numpy.array([[10, 11, 12], [13, 14, 15],
                           [16, 17, 18]], dtype=dtype)
        transa = 0  # A is not transposed in c-style(row-major)
        transb = 0  # B is not transposed in c-style(row-major)
        alpha = 1.0
        beta = 0.0

        m = npA.shape[0]
        n = npB.shape[1]
        k = npA.shape[1]

        dummy = clpy.empty(size, dtype)
        dummy.fill(wrong_value)

        # clpA is chunk with offset != 0
        clpA = clpy.empty(npA.shape, dtype=dtype)
        self.assertTrue(clpA.data.mem.offset != 0)
        clpA.set(npA)

        # clpB is chunk with offset == 0
        clpB = clpy.empty(npB.shape, dtype=dtype)
        self.assertTrue(clpB.data.mem.offset == 0)
        clpB.set(npB)

        expectedC = numpy.dot(npA, npB)
        clpC = clpy.ndarray(expectedC.shape, dtype=dtype)

        clpA, transa, lda = core._mat_to_cublas_contiguous(
            clpA, transa)  # as cublas-style
        clpB, transb, ldb = core._mat_to_cublas_contiguous(
            clpB, transb)  # as cublas-style
        ldc = clpC.shape[1]

        blas.sgemm(transa, transb,
                   m, n, k, alpha,
                   clpA, lda,
                   clpB, ldb,
                   beta, clpC, ldc
                   )

        actualC = clpC.get().T

        clpy.backend.memory.set_allocator()

        self.assertTrue(numpy.allclose(expectedC, actualC))

    def test_chunk_sgemm_B(self):
        # create chunk and free to prepare chunk in pool
        pool = clpy.backend.memory.SingleDeviceMemoryPool()
        clpy.backend.memory.set_allocator(pool.malloc)
        pooled_chunk_size = clpy.backend.memory.subbuffer_alignment * 2
        tmp = pool.malloc(pooled_chunk_size)
        pool.free(tmp.buf, pooled_chunk_size, 0)

        size = 3
        dtype = numpy.float32
        wrong_value = numpy.nan

        npA = numpy.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=dtype)
        npB = numpy.array([[10, 11, 12], [13, 14, 15],
                           [16, 17, 18]], dtype=dtype)
        transa = 0  # A is not transposed in c-style(row-major)
        transb = 0  # B is not transposed in c-style(row-major)
        alpha = 1.0
        beta = 0.0

        m = npA.shape[0]
        n = npB.shape[1]
        k = npA.shape[1]

        dummy = clpy.empty(size, dtype)
        dummy.fill(wrong_value)

        # clpB is chunk with offset != 0
        clpB = clpy.empty(npB.shape, dtype=dtype)
        self.assertTrue(clpB.data.mem.offset != 0)
        clpB.set(npB)

        # clpA is chunk with offset == 0
        clpA = clpy.empty(npA.shape, dtype=dtype)
        self.assertTrue(clpA.data.mem.offset == 0)
        clpA.set(npA)

        expectedC = numpy.dot(npA, npB)
        clpC = clpy.ndarray(expectedC.shape, dtype=dtype)

        clpA, transa, lda = core._mat_to_cublas_contiguous(
            clpA, transa)  # as cublas-style
        clpB, transb, ldb = core._mat_to_cublas_contiguous(
            clpB, transb)  # as cublas-style
        ldc = clpC.shape[1]

        blas.sgemm(transa, transb,
                   m, n, k, alpha,
                   clpA, lda,
                   clpB, ldb,
                   beta, clpC, ldc
                   )

        actualC = clpC.get().T

        clpy.backend.memory.set_allocator()

        self.assertTrue(numpy.allclose(expectedC, actualC))

    def test_chunk_sgemm_C(self):
        # create chunk and free to prepare chunk in pool
        pool = clpy.backend.memory.SingleDeviceMemoryPool()
        clpy.backend.memory.set_allocator(pool.malloc)
        pooled_chunk_size = clpy.backend.memory.subbuffer_alignment * 2
        tmp = pool.malloc(pooled_chunk_size)
        pool.free(tmp.buf, pooled_chunk_size, 0)

        size = 3
        dtype = numpy.float32
        wrong_value = numpy.nan

        npA = numpy.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=dtype)
        npB = numpy.array([[10, 11, 12], [13, 14, 15],
                           [16, 17, 18]], dtype=dtype)
        npC = numpy.array([[19, 20, 21], [22, 23, 24],
                           [25, 26, 27]], dtype=dtype)
        transa = 0  # A is not transposed in c-style(row-major)
        transb = 0  # B is not transposed in c-style(row-major)
        alpha = 1.0
        beta = 1.0

        m = npA.shape[0]
        n = npB.shape[1]
        k = npA.shape[1]

        expectedC = numpy.add(numpy.dot(npA, npB), beta * npC)

        dummy = clpy.empty(size, dtype)
        dummy.fill(wrong_value)

        # clpC is chunk with offset != 0
        clpC = clpy.ndarray(expectedC.shape, dtype=dtype)
        self.assertTrue(clpC.data.mem.offset != 0)
        clpC.set(npC.T)

        # clpA is chunk with offset == 0
        clpA = clpy.empty(npA.shape, dtype=dtype)
        self.assertTrue(clpA.data.mem.offset == 0)
        clpA.set(npA)

        # clpB is chunk with offset == 0
        clpB = clpy.empty(npB.shape, dtype=dtype)
        self.assertTrue(clpB.data.mem.offset == 0)
        clpB.set(npB)

        clpA, transa, lda = core._mat_to_cublas_contiguous(
            clpA, transa)  # as cublas-style
        clpB, transb, ldb = core._mat_to_cublas_contiguous(
            clpB, transb)  # as cublas-style
        ldc = clpC.shape[1]

        blas.sgemm(transa, transb,
                   m, n, k, alpha,
                   clpA, lda,
                   clpB, ldb,
                   beta, clpC, ldc
                   )

        actualC = clpC.get().T

        clpy.backend.memory.set_allocator()

        self.assertTrue(numpy.allclose(expectedC, actualC))

    def test_strides_transpose_A(self):
        npA = numpy.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                          dtype='float32')  # row-major
        npB = numpy.array([[10, 11, 12], [13, 14, 15], [
                          16, 17, 18]], dtype='float32')  # row-major
        npC = numpy.array([[19, 20, 21], [22, 23, 24], [
                          25, 26, 27]], dtype='float32')  # row-major

        alpha = 1.1
        beta = 2.1

        m = npA.shape[1]
        n = npB.shape[1]
        k = npA.shape[0]

        clpA = clpy.ndarray(npA.shape, dtype=numpy.dtype(
            'float32'))  # col-major in clpy
        clpA.set(npA)
        clpB = clpy.ndarray(npB.shape, dtype=numpy.dtype(
            'float32'))  # col-major in clpy
        clpB.set(npB)

        # change A.strides
        clpA = clpA.transpose(1, 0)
        npA = npA.transpose(1, 0)

        clpC = clpy.ndarray(npC.shape, dtype=numpy.dtype(
            'float32'))  # col-major in clpy
        clpC.set(npC.T)  # transpose C

        transa = 0  # A is not transposed in c-style(row-major)
        transb = 0  # B is not transposed in c-style(row-major)
        clpA, transa, lda = core._mat_to_cublas_contiguous(
            clpA, transa)  # as cublas-style
        clpB, transb, ldb = core._mat_to_cublas_contiguous(
            clpB, transb)  # as cublas-style
        ldc = clpC.shape[1]

        # AxB + beta*C
        expectedC = numpy.add(alpha * numpy.dot(npA, npB), beta * npC)

        # (A^T x B^T) + C^T in col-major = A x B + C in row-major
        blas.sgemm(transa, transb,
                   m, n, k, alpha,
                   clpA, lda,
                   clpB, ldb,
                   beta, clpC, ldc
                   )

        actualC = clpC.get().T  # as row-major

        self.assertTrue(numpy.allclose(expectedC, actualC))


class TestBlas3Sgeam(unittest.TestCase):
    """test class of ClPy's BLAS-extension function"""

    def test_row_matrix_row_matrix(self):
        alpha = 1.1
        beta = 2.1
        npA = numpy.array([
            [1, 2, 3],
            [4, 5, 6]], dtype='float32')  # row-major
        npB = numpy.array([
            [10, 11, 12],
            [13, 14, 15]], dtype='float32')  # row-major
        transa = 0  # A is not transposed in c-style(row-major)
        transb = 0  # B is not transposed in c-style(row-major)

        # C rows = (A in row-major) rows = (B in row-major) rows
        m = npA.shape[0]
        # C cols = (A in row-major) cols = (B in row-major) cols
        n = npA.shape[1]

        clpA = clpy.ndarray(npA.shape, dtype=numpy.dtype('float32'))
        clpA.set(npA)
        clpB = clpy.ndarray(npB.shape, dtype=numpy.dtype('float32'))
        clpB.set(npB)
        clpC = clpy.ndarray((n, m), dtype=numpy.dtype(
            'float32'))  # column-major

        clpA, transa, lda = core._mat_to_cublas_contiguous(
            clpA, transa)  # as cublas-style
        clpB, transb, ldb = core._mat_to_cublas_contiguous(
            clpB, transb)  # as cublas-style
        ldc = clpC.shape[1]

        blas.sgeam(transa, transb, m, n,
                   alpha, clpA, lda,
                   beta, clpB, ldb,
                   clpC, ldc)

        actualC = clpC.get().T  # as row-major
        # row-major caluculation
        expectedC = numpy.add(alpha * npA, beta * npB)
        self.assertTrue(numpy.allclose(expectedC, actualC))

    def test_column_matrix_column_matrix(self):
        alpha = 1.2
        beta = 2.2
        npA = numpy.array([
            [1, 2, 3],
            [4, 5, 6]], dtype='float32')  # column-major
        npB = numpy.array([
            [10, 11, 12],
            [13, 14, 15]], dtype='float32')  # column-major
        transa = 1  # A is transposed in c-style(row-major)
        transb = 1  # B is transposed in c-style(row-major)

        # C rows = (A in column-major) rows = (B in column-major) rows
        m = npA.shape[1]
        # C cols = (A in column-major) cols = (B in column-major) cols
        n = npA.shape[0]

        clpA = clpy.ndarray(npA.shape, dtype=numpy.dtype('float32'))
        clpA.set(npA)
        clpB = clpy.ndarray(npB.shape, dtype=numpy.dtype('float32'))
        clpB.set(npB)
        clpC = clpy.ndarray((n, m), dtype=numpy.dtype(
            'float32'))  # column-major

        clpA, transa, lda = core._mat_to_cublas_contiguous(
            clpA, transa)  # as cublas-style
        clpB, transb, ldb = core._mat_to_cublas_contiguous(
            clpB, transb)  # as cublas-style
        ldc = clpC.shape[1]

        blas.sgeam(transa, transb, m, n,
                   alpha, clpA, lda,
                   beta, clpB, ldb,
                   clpC, ldc)

        actualC = clpC.get().T  # as row-major
        # col-major caluculation
        expectedC = numpy.add(alpha * npA.T, beta * npB.T)
        self.assertTrue(numpy.allclose(expectedC, actualC))

    def test_row_matrix_column_matrix(self):
        alpha = 1.1
        beta = 2.2
        npA = numpy.array([
            [1, 2, 3],
            [4, 5, 6]], dtype='float32')  # row-major
        npB = numpy.array([
            [10, 11],
            [13, 14],
            [16, 17]], dtype='float32')  # column-major
        transa = 0  # A is not transposed in c-style(row-major)
        transb = 1  # B is transposed in c-style(row-major)

        # C rows = (A in row-major) rows = (B in column-major) rows
        m = npA.shape[0]
        # C cols = (A in row-major) cols = (B in column-major) cols
        n = npA.shape[1]

        clpA = clpy.ndarray(npA.shape, dtype=numpy.dtype('float32'))
        clpA.set(npA)
        clpB = clpy.ndarray(npB.shape, dtype=numpy.dtype('float32'))
        clpB.set(npB)
        clpC = clpy.ndarray((n, m), dtype=numpy.dtype(
            'float32'))  # column-major

        clpA, transa, lda = core._mat_to_cublas_contiguous(
            clpA, transa)  # as cublas-style
        clpB, transb, ldb = core._mat_to_cublas_contiguous(
            clpB, transb)  # as cublas-style
        ldc = clpC.shape[1]

        blas.sgeam(transa, transb, m, n,
                   alpha, clpA, lda,
                   beta, clpB, ldb,
                   clpC, ldc)

        actualC = clpC.get().T  # as row-major
        # col-major caluculation
        expectedC = numpy.add(alpha * npA, beta * npB.T)
        self.assertTrue(numpy.allclose(expectedC, actualC))

    def test_column_matrix_row_matrix(self):
        alpha = 1.2
        beta = 2.1
        npA = numpy.array([
            [1, 2],
            [4, 5],
            [7, 8]], dtype='float32')  # column-major
        npB = numpy.array([
            [10, 11, 12],
            [13, 14, 15]], dtype='float32')  # row-major
        transa = 1  # A is transposed in c-style(row-major)
        transb = 0  # B is not transposed in c-style(row-major)

        # C rows = (A in column-major) rows = (B in row-major) rows
        m = npA.shape[1]
        # C cols = (A in column-major) cols = (B in row-major) cols
        n = npA.shape[0]

        clpA = clpy.ndarray(npA.shape, dtype=numpy.dtype('float32'))
        clpA.set(npA)
        clpB = clpy.ndarray(npB.shape, dtype=numpy.dtype('float32'))
        clpB.set(npB)
        clpC = clpy.ndarray((n, m), dtype=numpy.dtype(
            'float32'))  # column-major

        clpA, transa, lda = core._mat_to_cublas_contiguous(
            clpA, transa)  # as cublas-style
        clpB, transb, ldb = core._mat_to_cublas_contiguous(
            clpB, transb)  # as cublas-style
        ldc = clpC.shape[1]

        blas.sgeam(transa, transb, m, n,
                   alpha, clpA, lda,
                   beta, clpB, ldb,
                   clpC, ldc)

        actualC = clpC.get().T  # as row-major
        # col-major caluculation
        expectedC = numpy.add(alpha * npA.T, beta * npB)
        self.assertTrue(numpy.allclose(expectedC, actualC))

    def test_alpha_zero(self):
        alpha = 0.0
        beta = 2.3
        npA = numpy.array([
            [1, 2, 3],
            [4, 5, 6]], dtype='float32')  # row-major
        npB = numpy.array([
            [10, 11, 12],
            [13, 14, 15]], dtype='float32')  # row-major
        transa = 0  # A is not transposed in c-style(row-major)
        transb = 0  # B is not transposed in c-style(row-major)

        # C rows = (A in row-major) rows = (B in row-major) rows
        m = npA.shape[0]
        # C cols = (A in row-major) cols = (B in row-major) cols
        n = npA.shape[1]

        clpA = clpy.ndarray(npA.shape, dtype=numpy.dtype('float32'))
        clpA.set(npA)
        clpB = clpy.ndarray(npB.shape, dtype=numpy.dtype('float32'))
        clpB.set(npB)
        clpC = clpy.ndarray((n, m), dtype=numpy.dtype(
            'float32'))  # column-major

        clpA, transa, lda = core._mat_to_cublas_contiguous(
            clpA, transa)  # as cublas-style
        clpB, transb, ldb = core._mat_to_cublas_contiguous(
            clpB, transb)  # as cublas-style
        ldc = clpC.shape[1]

        blas.sgeam(transa, transb, m, n,
                   alpha, clpA, lda,
                   beta, clpB, ldb,
                   clpC, ldc)

        actualC = clpC.get().T  # as row-major
        expectedC = beta * npB  # row-major caluculation
        self.assertTrue(numpy.allclose(expectedC, actualC))

    def test_beta_zero(self):
        alpha = 1.3
        beta = 0.0
        npA = numpy.array([
            [1, 2, 3],
            [4, 5, 6]], dtype='float32')  # row-major
        npB = numpy.array([
            [10, 11, 12],
            [13, 14, 15]], dtype='float32')  # row-major
        transa = 0  # A is not transposed in c-style(row-major)
        transb = 0  # B is not transposed in c-style(row-major)

        # C rows = (A in row-major) rows = (B in row-major) rows
        m = npA.shape[0]
        # C cols = (A in row-major) cols = (B in row-major) cols
        n = npA.shape[1]

        clpA = clpy.ndarray(npA.shape, dtype=numpy.dtype('float32'))
        clpA.set(npA)
        clpB = clpy.ndarray(npB.shape, dtype=numpy.dtype('float32'))
        clpB.set(npB)
        clpC = clpy.ndarray((n, m), dtype=numpy.dtype(
            'float32'))  # column-major

        clpA, transa, lda = core._mat_to_cublas_contiguous(
            clpA, transa)  # as cublas-style
        clpB, transb, ldb = core._mat_to_cublas_contiguous(
            clpB, transb)  # as cublas-style
        ldc = clpC.shape[1]

        blas.sgeam(transa, transb, m, n,
                   alpha, clpA, lda,
                   beta, clpB, ldb,
                   clpC, ldc)

        actualC = clpC.get().T  # as row-major
        expectedC = alpha * npA  # row-major caluculation
        self.assertTrue(numpy.allclose(expectedC, actualC))

    def test_alpha_beta_zero(self):
        alpha = 0.0
        beta = 0.0
        npA = numpy.array([
            [1, 2, 3],
            [4, 5, 6]], dtype='float32')  # row-major
        npB = numpy.array([
            [10, 11, 12],
            [13, 14, 15]], dtype='float32')  # row-major
        transa = 0  # A is not transposed in c-style(row-major)
        transb = 0  # B is not transposed in c-style(row-major)

        # C rows = (A in row-major) rows = (B in row-major) rows
        m = npA.shape[0]
        # C cols = (A in row-major) cols = (B in row-major) cols
        n = npA.shape[1]

        clpA = clpy.ndarray(npA.shape, dtype=numpy.dtype('float32'))
        clpA.set(npA)
        clpB = clpy.ndarray(npB.shape, dtype=numpy.dtype('float32'))
        clpB.set(npB)
        clpC = clpy.ndarray((n, m), dtype=numpy.dtype(
            'float32'))  # column-major

        clpA, transa, lda = core._mat_to_cublas_contiguous(
            clpA, transa)  # as cublas-style
        clpB, transb, ldb = core._mat_to_cublas_contiguous(
            clpB, transb)  # as cublas-style
        ldc = clpC.shape[1]

        blas.sgeam(transa, transb, m, n,
                   alpha, clpA, lda,
                   beta, clpB, ldb,
                   clpC, ldc)

        actualC = clpC.get().T  # as row-major
        expectedC = numpy.zeros((m, n))  # row-major caluculation
        self.assertTrue(numpy.allclose(expectedC, actualC))

    def test_chunk_sgeam_A(self):
        # create chunk and free to prepare chunk in pool
        pool = clpy.backend.memory.SingleDeviceMemoryPool()
        clpy.backend.memory.set_allocator(pool.malloc)
        pooled_chunk_size = clpy.backend.memory.subbuffer_alignment * 2
        tmp = pool.malloc(pooled_chunk_size)
        pool.free(tmp.buf, pooled_chunk_size, 0)

        size = 3
        dtype = numpy.float32
        wrong_value = numpy.nan

        npA = numpy.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=dtype)
        npB = numpy.array([[10, 11, 12], [13, 14, 15],
                           [16, 17, 18]], dtype=dtype)
        transa = 0  # A is not transposed in c-style(row-major)
        transb = 0  # B is not transposed in c-style(row-major)
        alpha = 1.0
        beta = 1.0

        m = npA.shape[0]
        n = npB.shape[1]

        dummy = clpy.empty(size, dtype)
        dummy.fill(wrong_value)

        # clpA is chunk with offset != 0
        clpA = clpy.empty(npA.shape, dtype=dtype)
        self.assertTrue(clpA.data.mem.offset != 0)
        clpA.set(npA)

        # clpB is chunk with offset == 0
        clpB = clpy.empty(npB.shape, dtype=dtype)
        self.assertTrue(clpB.data.mem.offset == 0)
        clpB.set(npB)

        expectedC = numpy.add(npA, npB)
        clpC = clpy.ndarray(expectedC.shape, dtype=dtype)

        clpA, transa, lda = core._mat_to_cublas_contiguous(
            clpA, transa)  # as cublas-style
        clpB, transb, ldb = core._mat_to_cublas_contiguous(
            clpB, transb)  # as cublas-style
        ldc = clpC.shape[1]

        blas.sgeam(transa, transb,
                   m, n,
                   alpha, clpA, lda,
                   beta, clpB, ldb,
                   clpC, ldc
                   )

        actualC = clpC.get().T

        clpy.backend.memory.set_allocator()

        self.assertTrue(numpy.allclose(expectedC, actualC))

    def test_chunk_sgeam_B(self):
        # create chunk and free to prepare chunk in pool
        pool = clpy.backend.memory.SingleDeviceMemoryPool()
        clpy.backend.memory.set_allocator(pool.malloc)
        pooled_chunk_size = clpy.backend.memory.subbuffer_alignment * 2
        tmp = pool.malloc(pooled_chunk_size)
        pool.free(tmp.buf, pooled_chunk_size, 0)

        size = 3
        dtype = numpy.float32
        wrong_value = numpy.nan

        npA = numpy.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=dtype)
        npB = numpy.array([[10, 11, 12], [13, 14, 15],
                           [16, 17, 18]], dtype=dtype)
        transa = 0  # A is not transposed in c-style(row-major)
        transb = 0  # B is not transposed in c-style(row-major)
        alpha = 1.0
        beta = 1.0

        m = npA.shape[0]
        n = npB.shape[1]

        dummy = clpy.empty(size, dtype)
        dummy.fill(wrong_value)

        # clpB is chunk with offset != 0
        clpB = clpy.empty(npB.shape, dtype=dtype)
        self.assertTrue(clpB.data.mem.offset != 0)
        clpB.set(npB)

        # clpA is chunk with offset == 0
        clpA = clpy.empty(npA.shape, dtype=dtype)
        self.assertTrue(clpA.data.mem.offset == 0)
        clpA.set(npA)

        expectedC = numpy.add(npA, npB)
        clpC = clpy.ndarray(expectedC.shape, dtype=dtype)

        clpA, transa, lda = core._mat_to_cublas_contiguous(
            clpA, transa)  # as cublas-style
        clpB, transb, ldb = core._mat_to_cublas_contiguous(
            clpB, transb)  # as cublas-style
        ldc = clpC.shape[1]

        blas.sgeam(transa, transb,
                   m, n,
                   alpha, clpA, lda,
                   beta, clpB, ldb,
                   clpC, ldc
                   )

        actualC = clpC.get().T

        clpy.backend.memory.set_allocator()

        self.assertTrue(numpy.allclose(expectedC, actualC))

    def test_chunk_sgeam_C(self):
        # create chunk and free to prepare chunk in pool
        pool = clpy.backend.memory.SingleDeviceMemoryPool()
        clpy.backend.memory.set_allocator(pool.malloc)
        pooled_chunk_size = clpy.backend.memory.subbuffer_alignment * 2
        tmp = pool.malloc(pooled_chunk_size)
        pool.free(tmp.buf, pooled_chunk_size, 0)

        size = 3
        dtype = numpy.float32
        wrong_value = numpy.nan

        npA = numpy.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=dtype)
        npB = numpy.array([[10, 11, 12], [13, 14, 15],
                           [16, 17, 18]], dtype=dtype)
        transa = 0  # A is not transposed in c-style(row-major)
        transb = 0  # B is not transposed in c-style(row-major)
        alpha = 1.0
        beta = 1.0

        m = npA.shape[0]
        n = npB.shape[1]

        dummy = clpy.empty(size, dtype)
        dummy.fill(wrong_value)

        # clpC is chunk with offset != 0
        expectedC = numpy.add(npA, npB)
        clpC = clpy.ndarray(expectedC.shape, dtype=dtype)
        self.assertTrue(clpC.data.mem.offset != 0)

        # clpB is chunk with offset == 0
        clpB = clpy.empty(npB.shape, dtype=dtype)
        self.assertTrue(clpB.data.mem.offset == 0)
        clpB.set(npB)

        # clpA is chunk with offset == 0
        clpA = clpy.empty(npA.shape, dtype=dtype)
        self.assertTrue(clpA.data.mem.offset == 0)
        clpA.set(npA)

        clpA, transa, lda = core._mat_to_cublas_contiguous(
            clpA, transa)  # as cublas-style
        clpB, transb, ldb = core._mat_to_cublas_contiguous(
            clpB, transb)  # as cublas-style
        ldc = clpC.shape[1]

        blas.sgeam(transa, transb,
                   m, n,
                   alpha, clpA, lda,
                   beta, clpB, ldb,
                   clpC, ldc
                   )

        actualC = clpC.get().T

        clpy.backend.memory.set_allocator()

        self.assertTrue(numpy.allclose(expectedC, actualC))

    def test_strides_transpose_A(self):
        alpha = 1.1
        beta = 2.1

        npA = numpy.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]], dtype='float32')  # col-major, will be transposed
        npB = numpy.array([
            [10, 11, 12],
            [13, 14, 15],
            [16, 17, 18]], dtype='float32')  # row-major

        clpA = clpy.ndarray(npA.shape, dtype=numpy.dtype('float32'))
        clpA.set(npA)
        clpB = clpy.ndarray(npB.shape, dtype=numpy.dtype('float32'))
        clpB.set(npB)

        # change A.strides
        clpA = clpA.transpose(1, 0)
        npA = npA.transpose(1, 0)

        transa = 0  # A is not transposed in c-style(row-major)
        transb = 0  # B is not transposed in c-style(row-major)

        clpA, transa, lda = core._mat_to_cublas_contiguous(
            clpA, transa)  # as cublas-style
        clpB, transb, ldb = core._mat_to_cublas_contiguous(
            clpB, transb)  # as cublas-style

        m = clpA.shape[1 - transa]
        n = clpA.shape[transa]

        clpC = clpy.ndarray((n, m), dtype=numpy.dtype('float32'))
        ldc = m

        blas.sgeam(transa, transb,
                   m, n,
                   alpha, clpA, lda,
                   beta, clpB, ldb,
                   clpC, ldc
                   )

        actualC = clpC.get().T  # as row-major
        # row-major caluculation
        expectedC = numpy.add(alpha * npA, beta * npB)

        self.assertTrue(numpy.allclose(expectedC, actualC))


if __name__ == '__main__':
    unittest.main()
