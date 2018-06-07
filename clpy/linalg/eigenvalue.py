import clpy  # NOQA
from clpy import backend  # NOQA
# from clpy.backend import cublas
from clpy.backend import device  # NOQA


'''
def _syevd(a, UPLO, with_eigen_vector):
    if UPLO not in ('L', 'U'):
        raise ValueError("UPLO argument must be 'L' or 'U'")

    if a.dtype == 'f' or a.dtype == 'e':
        dtype = 'f'
        ret_type = a.dtype
    else:
        # NumPy uses float64 when an input is not floating point number.
        dtype = 'd'
        ret_type = 'd'

    # Note that cuSolver assumes fortran array
    v = a.astype(dtype, order='F', copy=True)

    m, lda = a.shape
    w = clpy.empty(m, dtype)
    dev_info = clpy.empty((), 'i')
    handle = device.Device().cusolver_handle

    if with_eigen_vector:
        jobz = cusolver.CUSOLVER_EIG_MODE_VECTOR
    else:
        jobz = cusolver.CUSOLVER_EIG_MODE_NOVECTOR

    if UPLO == 'L':
        uplo = cublas.CUBLAS_FILL_MODE_LOWER
    else:  # UPLO == 'U'
        uplo = cublas.CUBLAS_FILL_MODE_UPPER

    if dtype == 'f':
        buffer_size = clpy.cuda.cusolver.ssyevd_bufferSize
        syevd = clpy.cuda.cusolver.ssyevd
    elif dtype == 'd':
        buffer_size = clpy.cuda.cusolver.dsyevd_bufferSize
        syevd = clpy.cuda.cusolver.dsyevd
    else:
        raise RuntimeError('Only float and double are supported')

    work_size = buffer_size(
        handle, jobz, uplo, m, v.data.ptr, lda, w.data.ptr)
    work = clpy.empty(work_size, dtype)
    syevd(
        handle, jobz, uplo, m, v.data.ptr, lda,
        w.data.ptr, work.data.ptr, work_size, dev_info.data.ptr)

    return w.astype(ret_type, copy=False), v.astype(ret_type, copy=False)
'''


# TODO(okuta): Implement eig


def eigh(a, UPLO='L'):
    """Eigenvalues and eigenvectors of a symmetric matrix.

    This method calculates eigenvalues and eigenvectors of a given
    symmetric matrix.

    .. note::

       Currenlty only 2-D matrix is supported.

    .. note::

       CUDA >=8.0 is required.

    Args:
        a (clpy.ndarray): A symmetric 2-D square matrix.
        UPLO (str): Select from ``'L'`` or ``'U'``. It specifies which
            part of ``a`` is used. ``'L'`` uses the lower triangular part of
            ``a``, and ``'U'`` uses the upper triangular part of ``a``.
    Returns:
        tuple of :class:`~clpy.ndarray`:
            Returns a tuple ``(w, v)``. ``w`` contains eigenvalues and
            ``v`` contains eigenvectors. ``v[:, i]`` is an eigenvector
            corresponding to an eigenvalue ``w[i]``.

    .. seealso:: :func:`numpy.linalg.eigh`
    """
    raise NotImplementedError("clpy does not support this")


# TODO(okuta): Implement eigvals


def eigvalsh(a, UPLO='L'):
    """Calculates eigenvalues of a symmetric matrix.

    This method calculates eigenvalues a given symmetric matrix.
    Note that :func:`clpy.linalg.eigh` calculates both eigenvalues and
    eigenvectors.

    .. note::

       Currenlty only 2-D matrix is supported.

    .. note::

       CUDA >=8.0 is required.

    Args:
        a (clpy.ndarray): A symmetric 2-D square matrix.
        UPLO (str): Select from ``'L'`` or ``'U'``. It specifies which
            part of ``a`` is used. ``'L'`` uses the lower triangular part of
            ``a``, and ``'U'`` uses the upper triangular part of ``a``.
    Returns:
        clpy.ndarray:
            Returns eigenvalues as a vector.

    .. seealso:: :func:`numpy.linalg.eigvalsh`
    """
    raise NotImplementedError("clpy does not support this")
