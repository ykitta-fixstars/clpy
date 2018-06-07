import unittest

from clpy import testing


@testing.gpu
class TestCorrelation(unittest.TestCase):

    _multiprocess_can_split_ = True
