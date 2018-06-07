import unittest

import numpy

import clpy
# from clpy.cuda import compiler
from clpy import testing


# def _compile_func(kernel_name, code):
#     mod = compiler.compile_with_cache(code)
#     return mod.get_function(kernel_name)


@testing.gpu
class TestFunction(unittest.TestCase):

    def test_python_scalar(self):
        code = '''
extern "C" __global__ void test_kernel(const double* a, double b, double* x) {
  int i = threadIdx.x;
  x[i] = a[i] + b;
}
'''

        a_cpu = numpy.arange(24, dtype=numpy.float64).reshape((4, 6))
        a = clpy.array(a_cpu)
        b = float(2)
        x = clpy.empty_like(a)

        func = _compile_func('test_kernel', code)  # NOQA

        func.linear_launch(a.size, (a, b, x))

        expected = a_cpu + b
        testing.assert_array_equal(x, expected)

    def test_numpy_scalar(self):
        code = '''
extern "C" __global__ void test_kernel(const double* a, double b, double* x) {
  int i = threadIdx.x;
  x[i] = a[i] + b;
}
'''

        a_cpu = numpy.arange(24, dtype=numpy.float64).reshape((4, 6))
        a = clpy.array(a_cpu)
        b = numpy.float64(2)
        x = clpy.empty_like(a)

        func = _compile_func('test_kernel', code)  # NOQA

        func.linear_launch(a.size, (a, b, x))

        expected = a_cpu + b
        testing.assert_array_equal(x, expected)
