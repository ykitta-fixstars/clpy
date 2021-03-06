import unittest

import numpy

from clpy import testing


@testing.gpu
class TestExplog(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.for_all_dtypes()
    @testing.numpy_clpy_allclose(atol=1e-5)
    def check_unary(self, name, xp, dtype, no_complex=False):
        if no_complex:
            if numpy.dtype(dtype).kind == 'c':
                return xp.array(True)
        a = testing.shaped_arange((2, 3), xp, dtype)
        return getattr(xp, name)(a)

    @testing.for_all_dtypes()
    @testing.numpy_clpy_allclose(atol=1e-5)
    def check_binary(self, name, xp, dtype, no_complex=False):
        if no_complex:
            if numpy.dtype(dtype).kind == 'c':
                return xp.array(True)
        a = testing.shaped_arange((2, 3), xp, dtype)
        b = testing.shaped_reverse_arange((2, 3), xp, dtype)
        return getattr(xp, name)(a, b)

    def test_exp(self):
        self.check_unary('exp')

    def test_expm1(self):
        self.check_unary('expm1', no_complex=True)

    def test_exp2(self):
        self.check_unary('exp2')

    def test_log(self):
        with testing.NumpyError(divide='ignore'):
            self.check_unary('log')

    def test_log10(self):
        with testing.NumpyError(divide='ignore'):
            self.check_unary('log10')

    def test_log2(self):
        with testing.NumpyError(divide='ignore'):
            self.check_unary('log2', no_complex=True)

    def test_log1p(self):
        self.check_unary('log1p', no_complex=True)

    def test_logaddexp(self):
        self.check_binary('logaddexp', no_complex=True)

    def test_logaddexp2(self):
        self.check_binary('logaddexp2', no_complex=True)
