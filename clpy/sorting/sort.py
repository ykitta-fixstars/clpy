import numpy  # NOQA

import clpy  # NOQA


def sort(a, axis=-1):
    """Returns a sorted copy of an array with a stable sorting algorithm.

    Args:
        a (clpy.ndarray): Array to be sorted.
        axis (int or None): Axis along which to sort. Default is -1, which
            means sort along the last axis. If None is supplied, the array is
            flattened before sorting.

    Returns:
        clpy.ndarray: Array of the same type and shape as ``a``.

    .. note::
       For its implementation reason, ``clpy.sort`` currently does not support
       ``kind`` and ``order`` parameters that ``numpy.sort`` does
       support.

    .. seealso:: :func:`numpy.sort`

    """
    if axis is None:
        ret = a.flatten()
        axis = -1
    else:
        ret = a.copy()
    ret.sort(axis=axis)
    return ret


def lexsort(keys):
    """Perform an indirect sort using an array of keys.

    Args:
        keys (clpy.ndarray): ``(k, N)`` array containing ``k`` ``(N,)``-shaped
            arrays. The ``k`` different "rows" to be sorted. The last row is
            the primary sort key.

    Returns:
        clpy.ndarray: Array of indices that sort the keys.

    .. note::
        For its implementation reason, ``clpy.lexsort`` currently supports only
        keys with their rank of one or two and does not support ``axis``
        parameter that ``numpy.lexsort`` supports.

    .. seealso:: :func:`numpy.lexsort`

    """

    # TODO(takagi): Support axis argument.

    # if not clpy.cuda.thrust_enabled:
    raise NotImplementedError("clpy does not support this")


'''
    if keys.ndim == ():
        # as numpy.lexsort() raises
        raise TypeError('need sequence of keys with len > 0 in lexsort')

    if keys.ndim == 1:
        return 0

    # TODO(takagi): Support ranks of three or more.
    if keys.ndim > 2:
        raise NotImplementedError('Keys with the rank of three or more is not '
                                  'supported in lexsort')

    idx_array = clpy.ndarray(keys._shape[1:], dtype=numpy.intp)
    k = keys._shape[0]
    n = keys._shape[1]
    thrust.lexsort(keys.dtype, idx_array.data.ptr, keys.data.ptr, k, n)

    return idx_array
'''


def argsort(a, axis=-1):
    """Returns the indices that would sort an array with a stable sorting.

    Args:
        a (clpy.ndarray): Array to sort.
        axis (int or None): Axis along which to sort. Default is -1, which
            means sort along the last axis. If None is supplied, the array is
            flattened before sorting.

    Returns:
        clpy.ndarray: Array of indices that sort ``a``.

    .. note::
        For its implementation reason, ``clpy.argsort`` does not support
        ``kind`` and ``order`` parameters.

    .. seealso:: :func:`numpy.argsort`

    """
    return a.argsort(axis=axis)


def msort(a):
    """Returns a copy of an array sorted along the first axis.

    Args:
        a (clpy.ndarray): Array to be sorted.

    Returns:
        clpy.ndarray: Array of the same type and shape as ``a``.

    .. note:
        ``clpy.msort(a)``, the CuPy counterpart of ``numpy.msort(a)``, is
        equivalent to ``clpy.sort(a, axis=0)``.

    .. seealso:: :func:`numpy.msort`

    """

    # TODO(takagi): Support float16 and bool.
    return sort(a, axis=0)


# TODO(okuta): Implement sort_complex


def partition(a, kth, axis=-1):
    """Returns a partially sorted copy of an array.

    Creates a copy of the array whose elements are rearranged such that the
    value of the element in k-th position would occur in that position in a
    sorted array. All of the elements before the new k-th element are less
    than or equal to the elements after the new k-th element.

    Args:
        a (clpy.ndarray): Array to be sorted.
        kth (int or sequence of ints): Element index to partition by. If
            supplied with a sequence of k-th it will partition all elements
            indexed by k-th of them into their sorted position at once.
        axis (int or None): Axis along which to sort. Default is -1, which
            means sort along the last axis. If None is supplied, the array is
            flattened before sorting.

    Returns:
        clpy.ndarray: Array of the same type and shape as ``a``.

    .. note::
       For its implementation reason, :func:`clpy.partition` fully sorts the
       given array as :func:`clpy.sort` does. It also does not support
       ``kind`` and ``order`` parameters that :func:`numpy.partition` supports.

    .. seealso:: :func:`numpy.partition`

    """
    if axis is None:
        ret = a.flatten()
        axis = -1
    else:
        ret = a.copy()
    ret.partition(kth, axis=axis)
    return ret


def argpartition(a, kth, axis=-1):
    """Returns the indices that would partially sort an array.

    Args:
        a (clpy.ndarray): Array to be sorted.
        kth (int or sequence of ints): Element index to partition by. If
            supplied with a sequence of k-th it will partition all elements
            indexed by k-th of them into their sorted position at once.
        axis (int or None): Axis along which to sort. Default is -1, which
            means sort along the last axis. If None is supplied, the array is
            flattened before sorting.

    Returns:
        clpy.ndarray: Array of the same type and shape as ``a``.

    .. note::
        For its implementation reason, `clpy.argpartition` fully sorts the
        given array as `clpy.argsort` does. It also does not support ``kind``
        and ``order`` parameters that ``numpy.argpartition`` supports.

    .. seealso:: :func:`numpy.argpartition`

    """
    return a.argpartition(kth, axis=axis)
