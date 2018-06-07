import unittest

import numpy

from clpy import random


class TestRandomSample(unittest.TestCase):

    def test_rand_valid_range(self):
        n = 10000
        ar = random.rand(n, dtype=numpy.float32)
        result = ar.get()

        ones = numpy.ones(n)
        zeros = numpy.zeros(n)
        self.assertTrue(numpy.greater_equal(
            zeros, result).all and numpy.less(result, ones).all())

    def test_rand_call_twice(self):
        # diffrent length of array than other test case is required
        n = 10
        random.rand(n, dtype=numpy.float32)
        ar = random.rand(n, dtype=numpy.float32)

        result = ar.get()

        ones = numpy.ones(n)
        zeros = numpy.zeros(n)

        self.assertTrue(numpy.greater_equal(
            zeros, result).all and numpy.less(result, ones).all())

    def test_rand_generate_different_result(self):
        n = 100
        a = random.rand(n, dtype=numpy.float32)
        b = random.rand(n, dtype=numpy.float32)

        self.assertFalse(numpy.allclose(a.get(), b.get()))


if __name__ == "__main__":
    unittest.main()
