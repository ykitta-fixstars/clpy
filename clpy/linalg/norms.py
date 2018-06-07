import numpy
from numpy import linalg  # NOQA

import clpy
from clpy import backend  # NOQA
from clpy.backend import device  # NOQA
from clpy.linalg import decomposition
from clpy.linalg import util  # NOQA


def norm(x, ord=None, axis=None, keepdims=False):
    """Returns one of matrix norms specified by ``ord`` parameter.

    Complex valued matrices and vectors are not supported.
    See numpy.linalg.norm for more detail.

    Args:
        x (clpy.ndarray): Array to take norm. If ``axis`` is None,
            ``x`` must be 1-D or 2-D.
        ord (non-zero int, inf, -inf, 'fro'): Norm type.
        axis (int, 2-tuple of ints, None): 1-D or 2-D norm is cumputed over
            ``axis``.
        keepdims (bool): If this is set ``True``, the axes which are normed
            over are left.

    Returns:
        clpy.ndarray

    """
    if not issubclass(x.dtype.type, numpy.inexact):
        x = x.astype(float)

    # Immediately handle some default, simple, fast, and common cases.
    if axis is None:
        ndim = x.ndim
        if (ord is None or (ndim == 1 and ord == 2) or
                (ndim == 2 and ord in ('f', 'fro'))):
            ret = clpy.sqrt(clpy.sum(x.ravel() ** 2))
            if keepdims:
                ret = ret.reshape((1,) * ndim)
            return ret

    # Normalize the `axis` argument to a tuple.
    nd = x.ndim
    if axis is None:
        axis = tuple(range(nd))
    elif not isinstance(axis, tuple):
        try:
            axis = int(axis)
        except Exception:
            raise TypeError(
                "'axis' must be None, an integer or a tuple of integers")
        axis = (axis,)

    if len(axis) == 1:
        if ord == numpy.Inf:
            return abs(x).max(axis=axis, keepdims=keepdims)
        elif ord == -numpy.Inf:
            return abs(x).min(axis=axis, keepdims=keepdims)
        elif ord == 0:
            # Zero norm
            # Convert to Python float in accordance with NumPy
            return (x != 0).sum(axis=axis, keepdims=keepdims, dtype='d')
        elif ord == 1:
            # special case for speedup
            return abs(x).sum(axis=axis, keepdims=keepdims)
        elif ord is None or ord == 2:
            # special case for speedup
            s = x ** 2
            return clpy.sqrt(s.sum(axis=axis, keepdims=keepdims))
        else:
            try:
                float(ord)
            except TypeError:
                raise ValueError("Invalid norm order for vectors.")
            absx = abs(x).astype('d')
            absx **= ord
            ret = absx.sum(axis=axis, keepdims=keepdims)
            ret **= (1.0 / ord)
            return ret
    elif len(axis) == 2:
        row_axis, col_axis = axis
        if row_axis < 0:
            row_axis += nd
        if col_axis < 0:
            col_axis += nd
        if not (0 <= row_axis < nd and 0 <= col_axis < nd):
            raise ValueError('Invalid axis %r for an array with shape %r' %
                             (axis, x.shape))
        if row_axis == col_axis:
            raise ValueError('Duplicate axes given.')
        if ord == 1:
            if col_axis > row_axis:
                col_axis -= 1
            ret = abs(x).sum(axis=row_axis).max(axis=col_axis)
        elif ord == numpy.Inf:
            if row_axis > col_axis:
                row_axis -= 1
            ret = abs(x).sum(axis=col_axis).max(axis=row_axis)
        elif ord == -1:
            if col_axis > row_axis:
                col_axis -= 1
            ret = abs(x).sum(axis=row_axis).min(axis=col_axis)
        elif ord == -numpy.Inf:
            if row_axis > col_axis:
                row_axis -= 1
            ret = abs(x).sum(axis=col_axis).min(axis=row_axis)
        elif ord in [None, 'fro', 'f']:
            ret = clpy.sqrt((x ** 2).sum(axis=axis))
        else:
            raise ValueError("Invalid norm order for matrices.")
        if keepdims:
            ret_shape = list(x.shape)
            ret_shape[axis[0]] = 1
            ret_shape[axis[1]] = 1
            ret = ret.reshape(ret_shape)
        return ret
    else:
        raise ValueError("Improper number of dimensions to norm.")


# TODO(okuta): Implement cond


def det(a):
    """Retruns the deteminant of an array.

    Args:
        a (clpy.ndarray): The input matrix with dimension ``(..., N, N)``.

    Returns:
        clpy.ndarray: Determinant of ``a``. Its shape is ``a.shape[:-2]``.

    .. seealso:: :func:`numpy.linalg.det`
    """
    sign, logdet = slogdet(a)
    return sign * clpy.exp(logdet)


def matrix_rank(M, tol=None):
    """Return matrix rank of array using SVD method

    Args:
        M (clpy.ndarray): Input array. Its `ndim` must be less than or equal to
            2.
        tol (None or float): Threshold of singular value of `M`.
            When `tol` is `None`, and `eps` is the epsilon value for datatype
            of `M`, then `tol` is set to `S.max() * max(M.shape) * eps`,
            where `S` is the singular value of `M`.
            It obeys :func:`numpy.linalg.matrix_rank`.

    Returns:
        clpy.ndarray: Rank of `M`.

    .. seealso:: :func:`numpy.linalg.matrix_rank`
    """
    if M.ndim < 2:
        return (M != 0).any().astype('l')
    S = decomposition.svd(M, compute_uv=False)
    if tol is None:
        tol = (S.max(axis=-1, keepdims=True) * max(M.shape[-2:]) *
               numpy.finfo(S.dtype).eps)
    return (S > tol).sum(axis=-1)


def slogdet(a):
    """Returns sign and logarithm of the determinat of an array.

    It calculates the natural logarithm of the deteminant of a given value.

    Args:
        a (clpy.ndarray): The input matrix with dimension ``(..., N, N)``.

    Returns:
        tuple of :class:`~clpy.ndarray`:
            It returns a tuple ``(sign, logdet)``. ``sign`` represents each
            sign of the deteminant as a real number ``0``, ``1`` or ``-1``.
            'logdet' represents the natural logarithm of the absolute of the
            deteminant.
            If the deteninant is zero, ``sign`` will be ``0`` and ``logdet``
            will be ``-inf``.
            The shapes of both ``sign`` and ``logdet`` are equal to
            ``a.shape[:-2]``.

    .. seealso:: :func:`numpy.linalg.slogdet`
    """
    raise NotImplementedError("clpy does not support this")


'''
    if a.ndim < 2:
        msg = ('%d-dimensional array given. '
               'Array must be at least two-dimensional' % a.ndim)
        raise linalg.LinAlgError(msg)

    dtype = numpy.find_common_type((a.dtype.char, 'f'), ())
    shape = a.shape[:-2]
    sign = clpy.empty(shape, dtype)
    logdet = clpy.empty(shape, dtype)

    a = a.astype(dtype)
    for index in numpy.ndindex(*shape):
        s, l = _slogdet_one(a[index])
        sign[index] = s
        logdet[index] = l
    return sign, logdet
'''


'''
def _slogdet_one(a):
    util._assert_rank2(a)
    util._assert_nd_squareness(a)
    dtype = a.dtype

    handle = device.get_cusolver_handle()
    m = len(a)
    ipiv = clpy.empty(m, 'i')
    info = clpy.empty((), 'i')

    # Need to make a copy because getrf works inplace
    a_copy = a.copy(order='F')

    if dtype == 'f':
        getrf_bufferSize = cusolver.sgetrf_bufferSize
        getrf = cusolver.sgetrf
    else:
        getrf_bufferSize = cusolver.dgetrf_bufferSize
        getrf = cusolver.dgetrf

    buffersize = getrf_bufferSize(handle, m, m, a_copy.data.ptr, m)
    workspace = clpy.empty(buffersize, dtype=dtype)
    getrf(handle, m, m, a_copy.data.ptr, m, workspace.data.ptr,
          ipiv.data.ptr, info.data.ptr)

    if info[()] == 0:
        diag = clpy.diag(a_copy)
        # ipiv is 1-origin
        non_zero = (clpy.count_nonzero(ipiv != clpy.arange(1, m + 1)) +
                    clpy.count_nonzero(diag < 0))
        # Note: sign == -1 ** (non_zero % 2)
        sign = (non_zero % 2) * -2 + 1
        logdet = clpy.log(abs(diag)).sum()
    else:
        sign = clpy.array(0.0, dtype=dtype)
        logdet = clpy.array(float('-inf'), dtype)

    return sign, logdet


def trace(a, offset=0, axis1=0, axis2=1, dtype=None, out=None):
    """Returns the sum along the diagonals of an array.

    It computes the sum along the diagonals at ``axis1`` and ``axis2``.

    Args:
        a (clpy.ndarray): Array to take trace.
        offset (int): Index of diagonals. Zero indicates the main diagonal, a
            positive value an upper diagonal, and a negative value a lower
            diagonal.
        axis1 (int): The first axis along which the trace is taken.
        axis2 (int): The second axis along which the trace is taken.
        dtype: Data type specifier of the output.
        out (clpy.ndarray): Output array.

    Returns:
        clpy.ndarray: The trace of ``a`` along axes ``(axis1, axis2)``.

    .. seealso:: :func:`numpy.trace`

    """
    # TODO(okuta): check type
    return a.trace(offset, axis1, axis2, dtype, out)
'''
