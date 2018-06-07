import unittest

from clpy import testing


@testing.gpu
class TestAddRemove(unittest.TestCase):

    _multiprocess_can_split_ = True
