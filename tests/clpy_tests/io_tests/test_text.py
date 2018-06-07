import unittest

from clpy import testing


@testing.gpu
class TestText(unittest.TestCase):

    _multiprocess_can_split_ = True
