import unittest

import clpy
from clpy import testing


@testing.gpu
class TestPermutations(unittest.TestCase):

    _multiprocess_can_split_ = True


@testing.gpu
class TestShuffle(unittest.TestCase):

    _multiprocess_can_split_ = True

    # Test ranks

    @testing.numpy_clpy_raises()
    def test_shuffle_zero_dim(self, xp):
        a = testing.shaped_random((), xp)
        xp.random.shuffle(a)

    # Test same values

    @testing.for_all_dtypes(no_float16=True, no_bool=True, no_complex=True)
    def test_shuffle_sort_1dim(self, dtype):
        a = clpy.arange(10, dtype=dtype)
        b = clpy.copy(a)
        clpy.random.shuffle(a)
        testing.assert_allclose(clpy.sort(a), b)

    @testing.for_all_dtypes(no_float16=True, no_bool=True, no_complex=True)
    def test_shuffle_sort_ndim(self, dtype):
        a = clpy.arange(15, dtype=dtype).reshape(5, 3)
        b = clpy.copy(a)
        clpy.random.shuffle(a)
        testing.assert_allclose(clpy.sort(a, axis=0), b)

    # Test seed

    @testing.for_all_dtypes()
    def test_shuffle_seed1(self, dtype):
        a = testing.shaped_random((10,), clpy, dtype)
        b = clpy.copy(a)
        clpy.random.seed(0)
        clpy.random.shuffle(a)
        clpy.random.seed(0)
        clpy.random.shuffle(b)
        testing.assert_allclose(a, b)
