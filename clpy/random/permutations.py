from clpy.random import generator

# TODO(okuta): Implement permutation


def shuffle(a):
    """Shuffles an array.

    Args:
        a (clpy.ndarray): The array to be shuffled.

    .. seealso:: :func:`numpy.random.shuffle`

    """
    rs = generator.get_random_state()
    return rs.shuffle(a)
