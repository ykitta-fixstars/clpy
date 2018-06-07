import numpy

import clpy


def diag(v, k=0):
    """Returns a diagonal or a diagonal array.

    Args:
        v (array-like): Array or array-like object.
        k (int): Index of diagonals. Zero indicates the main diagonal, a
            positive value an upper diagonal, and a negative value a lower
            diagonal.

    Returns:
        clpy.ndarray: If ``v`` indicates a 1-D array, then it returns a 2-D
        array with the specified diagonal filled by ``v``. If ``v`` indicates a
        2-D array, then it returns the specified diagonal of ``v``. In latter
        case, if ``v`` is a :class:`clpy.ndarray` object, then its view is
        returned.

    .. seealso:: :func:`numpy.diag`

    """
    if isinstance(v, clpy.ndarray):
        if v.ndim == 1:
            size = v.size + abs(k)
            ret = clpy.zeros((size, size), dtype=v.dtype)
            ret.diagonal(k)[:] = v
            return ret
        else:
            return v.diagonal(k)
    else:
        return clpy.array(numpy.diag(v, k))


def diagflat(v, k=0):
    """Creates a diagonal array from the flattened input.

    Args:
        v (array-like): Array or array-like object.
        k (int): Index of diagonals. See :func:`clpy.diag` for detail.

    Returns:
        clpy.ndarray: A 2-D diagonal array with the diagonal copied from ``v``.

    """
    if isinstance(v, clpy.ndarray):
        return clpy.diag(v.ravel(), k)
    else:
        return clpy.diag(numpy.ndarray(v).ravel(), k)


# TODO(okuta): Implement tri


# TODO(okuta): Implement tril


# TODO(okuta): Implement triu


# TODO(okuta): Implement vander


# TODO(okuta): Implement mat


# TODO(okuta): Implement bmat
