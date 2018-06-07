"""CUBLAS interface"""

# from unsupport/cublas.pxd

###############################################################################
# Context
###############################################################################


def create():
    raise NotImplementedError("clpy does not support this")


def destroy(handle):
    raise NotImplementedError("clpy does not support this")


def getVersion(handle):
    raise NotImplementedError("clpy does not support this")


def getPointerMode(handle):
    raise NotImplementedError("clpy does not support this")


def setPointerMode(handle, mode):
    raise NotImplementedError("clpy does not support this")


###############################################################################
# Stream
###############################################################################

def setStream(handle, stream):
    raise NotImplementedError("clpy does not support this")


def getStream(handle):
    raise NotImplementedError("clpy does not support this")


###############################################################################
# BLAS Level 1
###############################################################################

def isamax(handle, n, x, incx):
    raise NotImplementedError("clpy does not support this")


def isamin(handle, n, x, incx):
    raise NotImplementedError("clpy does not support this")


def sasum(handle, n, x, incx):
    raise NotImplementedError("clpy does not support this")


def saxpy(handle, n, alpha, x, incx, y, incy):
    raise NotImplementedError("clpy does not support this")


def daxpy(handle, n, alpha, x, incx, y, incy):
    raise NotImplementedError("clpy does not support this")


def sdot(handle, n, x, incx, y, incy, result):
    raise NotImplementedError("clpy does not support this")


def ddot(handle, n, x, incx, y, incy, result):
    raise NotImplementedError("clpy does not support this")


def cdotu(handle, n, x, incx, y, incy, result):
    raise NotImplementedError("clpy does not support this")


def cdotc(handle, n, x, incx, y, incy, result):
    raise NotImplementedError("clpy does not support this")


def zdotu(handle, n, x, incx, y, incy, result):
    raise NotImplementedError("clpy does not support this")


def zdotc(handle, n, x, incx, y, incy, result):
    raise NotImplementedError("clpy does not support this")


def snrm2(handle, n, x, incx):
    raise NotImplementedError("clpy does not support this")


def sscal(handle, n, alpha, x, incx):
    raise NotImplementedError("clpy does not support this")


###############################################################################
# BLAS Level 2
###############################################################################

def sgemv(handle, trans, m, n, alpha, A,
          lda, x, incx, beta, y, incy):
    raise NotImplementedError("clpy does not support this")


def dgemv(handle, trans, m, n, alpha, A,
          lda, x, incx, beta, y, incy):
    raise NotImplementedError("clpy does not support this")


def cgemv(handle, trans, m, n, alpha,
          A, lda, x, incx, beta,
          y, incy):
    raise NotImplementedError("clpy does not support this")


def zgemv(handle, trans, m, n, alpha,
          A, lda, x, incx, beta,
          y, incy):
    raise NotImplementedError("clpy does not support this")


def sger(handle, m, n, alpha, x, incx,
         y, incy, A, lda):
    raise NotImplementedError("clpy does not support this")


def dger(handle, m, n, alpha, x, incx,
         y, incy, A, lda):
    raise NotImplementedError("clpy does not support this")


def cgeru(handle, m, n, alpha, x,
          incx, y, incy, A, lda):
    raise NotImplementedError("clpy does not support this")


def cgerc(handle, m, n, alpha, x,
          incx, y, incy, A, lda):
    raise NotImplementedError("clpy does not support this")


def zgeru(handle, m, n, alpha, x,
          incx, y, incy, A, lda):
    raise NotImplementedError("clpy does not support this")


def zgerc(handle, m, n, alpha, x,
          incx, y, incy, A, lda):
    raise NotImplementedError("clpy does not support this")


###############################################################################
# BLAS Level 3
###############################################################################

def sgemm(handle, transa, transb,
          m, n, k, alpha, A, lda,
          B, ldb, beta, C, ldc):
    raise NotImplementedError("clpy does not support this")


def dgemm(handle, transa, transb,
          m, n, k, alpha, A, lda,
          B, ldb, beta, C, ldc):
    raise NotImplementedError("clpy does not support this")


def cgemm(handle, transa, transb,
          m, n, k, alpha, A, lda,
          B, ldb, beta, C, ldc):
    raise NotImplementedError("clpy does not support this")


def zgemm(handle, transa, transb,
          m, n, k, alpha, A, lda,
          B, ldb, beta, C, ldc):
    raise NotImplementedError("clpy does not support this")


def sgemmBatched(handle, transa, transb,
                 m, n, k, alpha, Aarray, lda,
                 Barray, ldb, beta, Carray, ldc,
                 batchCount):
    raise NotImplementedError("clpy does not support this")


def dgemmBatched(handle, transa, transb,
                 m, n, k, alpha, Aarray, lda,
                 Barray, ldb, beta, Carray, ldc,
                 batchCount):
    raise NotImplementedError("clpy does not support this")


def cgemmBatched(handle, transa, transb,
                 m, n, k, alpha, Aarray,
                 lda, Barray, ldb, beta,
                 Carray, ldc, batchCount):
    raise NotImplementedError("clpy does not support this")


def zgemmBatched(handle, transa, transb,
                 m, n, k, alpha, Aarray,
                 lda, Barray, ldb, beta,
                 Carray, ldc, batchCount):
    raise NotImplementedError("clpy does not support this")


def strsm(handle, side, uplo, trans, diag,
          m, n, alpha, Aarray, lda,
          Barray, ldb):
    raise NotImplementedError("clpy does not support this")


def dtrsm(handle, side, uplo, trans, diag,
          m, n, alpha, Aarray, lda,
          Barray, ldb):
    raise NotImplementedError("clpy does not support this")

###############################################################################
# BLAS extension
###############################################################################


def sgeam(handle, transa, transb, m, n,
          alpha, A, lda, beta, B, ldb,
          C, ldc):
    raise NotImplementedError("clpy does not support this")


def dgeam(handle, transa, transb, m, n,
          alpha, A, lda, beta, B, ldb,
          C, ldc):
    raise NotImplementedError("clpy does not support this")


def sdgmm(handle, mode, m, n, A, lda,
          x, incx, C, ldc):
    raise NotImplementedError("clpy does not support this")


def sgemmEx(handle, transa, transb, m, n, k,
            alpha, A, Atype, lda, B,
            Btype, ldb, beta, C, Ctype,
            ldc):
    raise NotImplementedError("clpy does not support this")


def sgetrfBatched(handle, n, Aarray, lda,
                  PivotArray, infoArray, batchSize):
    raise NotImplementedError("clpy does not support this")


def sgetriBatched(handle, n, Aarray, lda,
                  PivotArray, Carray, ldc,
                  infoArray, batchSize):
    raise NotImplementedError("clpy does not support this")
