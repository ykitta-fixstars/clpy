import clpy


def tile(A, reps):
    """Construct an array by repeating A the number of times given by reps.

    Args:
        A (clpy.ndarray): Array to transform.
        reps (int or tuple): The number of repeats.

    Returns:
        clpy.ndarray: Transformed array with repeats.

    .. seealso:: :func:`numpy.tile`

    """
    try:
        tup = tuple(reps)
    except TypeError:
        tup = (reps,)
    d = len(tup)
    if tup.count(1) == len(tup) and isinstance(A, clpy.ndarray):
        # Fixes the problem that the function does not make a copy if A is a
        # array and the repetitions are 1 in all dimensions
        return clpy.array(A, copy=True, ndmin=d)
    else:
        # Note that no copy of zero-sized arrays is made. However since they
        # have no data there is no risk of an inadvertent overwrite.
        c = clpy.array(A, copy=False, ndmin=d)
    if d < c.ndim:
        tup = (1,) * (c.ndim - d) + tup
    shape_out = tuple(s * t for s, t in zip(c.shape, tup))
    if c.size == 0:
        return clpy.empty(shape_out, dtype=c.dtype)
    c_shape = []
    ret_shape = []
    for dim_in, nrep in zip(c.shape, tup):
        if nrep == 1:
            c_shape.append(dim_in)
            ret_shape.append(dim_in)
        elif dim_in == 1:
            c_shape.append(dim_in)
            ret_shape.append(nrep)
        else:
            c_shape.append(1)
            c_shape.append(dim_in)
            ret_shape.append(nrep)
            ret_shape.append(dim_in)
    ret = clpy.empty(ret_shape, dtype=c.dtype)
    if ret.size:
        ret[...] = c.reshape(c_shape)
    return ret.reshape(shape_out)


def repeat(a, repeats, axis=None):
    """Repeat arrays along an axis.

    Args:
        a (clpy.ndarray): Array to transform.
        repeats (int, list or tuple): The number of repeats.
        axis (int): The axis to repeat.

    Returns:
        clpy.ndarray: Transformed array with repeats.

    .. seealso:: :func:`numpy.repeat`

    """
    return a.repeat(repeats, axis)
