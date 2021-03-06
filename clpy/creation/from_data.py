from clpy import core


def array(obj, dtype=None, copy=True, order='K', subok=False, ndmin=0):
    """Creates an array on the current device.

    This function currently does not support the ``order`` and ``subok``
    options.

    Args:
        obj: :class:`clpy.ndarray` object or any other object that can be
            passed to :func:`numpy.array`.
        dtype: Data type specifier.
        copy (bool): If ``False``, this function returns ``obj`` if possible.
            Otherwise this function always returns a new array.
        order ({'C', 'F', 'A', 'K'}): Row-major (C-style) or column-major
            (Fortran-style) order.
            When ``order`` is 'A', it uses 'F' if ``a`` is column-major and
            uses 'C' otherwise.
            And when ``order`` is 'K', it keeps strides as closely as
            possible.
            If ``obj`` is :class:`numpy.ndarray`, the function returns 'C' or
            'F' order array.
        subok (bool): If True, then sub-classes will be passed-through,
            otherwise the returned array will be forced to be a base-class
            array (default).
        ndmin (int): Minimum number of dimensions. Ones are inserted to the
            head of the shape if needed.

    Returns:
        clpy.ndarray: An array on the current device.



    .. note::
       This method currently does not support ``subok`` argument.

    .. seealso:: :func:`numpy.array`

    """
    return core.array(obj, dtype, copy, order, subok, ndmin)


def asarray(a, dtype=None):
    """Converts an object to array.

    This is equivalent to ``array(a, dtype, copy=False)``.
    This function currently does not support the ``order`` option.

    Args:
        a: The source object.
        dtype: Data type specifier. It is inferred from the input by default.

    Returns:
        clpy.ndarray: An array on the current device. If ``a`` is already on
        the device, no copy is performed.

    .. seealso:: :func:`numpy.asarray`

    """
    return core.array(a, dtype, False)


def asanyarray(a, dtype=None):
    """Converts an object to array.

    This is currently equivalent to :func:`~clpy.asarray`, since there is no
    subclass of ndarray in CuPy. Note that the original
    :func:`numpy.asanyarray` returns the input array as is if it is an instance
    of a subtype of :class:`numpy.ndarray`.

    .. seealso:: :func:`clpy.asarray`, :func:`numpy.asanyarray`

    """
    return core.array(a, dtype, False)


def ascontiguousarray(a, dtype=None):
    """Returns a C-contiguous array.

    Args:
        a (clpy.ndarray): Source array.
        dtype: Data type specifier.

    Returns:
        clpy.ndarray: If no copy is required, it returns ``a``. Otherwise, it
        returns a copy of ``a``.

    .. seealso:: :func:`numpy.ascontiguousarray`

    """
    return core.ascontiguousarray(a, dtype)


# TODO(okuta): Implement asmatrix


def copy(a, order='K'):
    """Creates a copy of a given array on the current device.

    This function allocates the new array on the current device. If the given
    array is allocated on the different device, then this function tries to
    copy the contents over the devices.

    Args:
        a (clpy.ndarray): The source array.
        order ({'C', 'F', 'A', 'K'}): Row-major (C-style) or column-major
            (Fortran-style) order.
            When `order` is 'A', it uses 'F' if `a` is column-major and
            uses `C` otherwise.
            And when `order` is 'K', it keeps strides as closely as
            possible.

    Returns:
        clpy.ndarray: The copy of ``a`` on the current device.

    See: :func:`numpy.copy`, :meth:`clpy.ndarray.copy`

    """
    # If the current device is different from the device of ``a``, then this
    # function allocates a new array on the current device, and copies the
    # contents over the devices.
    return a.copy(order=order)


# TODO(okuta): Implement frombuffer


# TODO(okuta): Implement fromfile


# TODO(okuta): Implement fromfunction


# TODO(okuta): Implement fromiter


# TODO(okuta): Implement fromstring


# TODO(okuta): Implement loadtxt
