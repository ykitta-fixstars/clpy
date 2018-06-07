from clpy import core
from clpy.math import ufunc


exp = ufunc.create_math_ufunc(
    'exp', 1, 'clpy_exp',
    '''Elementwise exponential function.

    .. seealso:: :data:`numpy.exp`

    ''')


expm1 = ufunc.create_math_ufunc(
    'expm1', 1, 'clpy_expm1',
    '''Computes ``exp(x) - 1`` elementwise.

    .. seealso:: :data:`numpy.expm1`

    ''')


exp2 = core.create_ufunc(
    'clpy_exp2',
    ('f->f', 'd->d', 'F->F', 'D->D'),
    'out0 = pow((in0_type)2, in0)',
    doc='''Elementwise exponentiation with base 2.

    .. seealso:: :data:`numpy.exp2`

    ''')


log = ufunc.create_math_ufunc(
    'log', 1, 'clpy_log',
    '''Elementwise natural logarithm function.

    .. seealso:: :data:`numpy.log`

    ''')


log10 = ufunc.create_math_ufunc(
    'log10', 1, 'clpy_log10',
    '''Elementwise common logarithm function.

    .. seealso:: :data:`numpy.log10`

    ''')


log2 = ufunc.create_math_ufunc(
    'log2', 1, 'clpy_log2',
    '''Elementwise binary logarithm function.

    .. seealso:: :data:`numpy.log2`

    ''')


log1p = ufunc.create_math_ufunc(
    'log1p', 1, 'clpy_log1p',
    '''Computes ``log(1 + x)`` elementwise.

    .. seealso:: :data:`numpy.log1p`

    ''')


logaddexp = core.create_ufunc(
    'clpy_logaddexp',
    ('ff->f', 'dd->d'),
    'out0 = fmax(in0, in1) + log1p(exp(-fabs(in0 - in1)))',
    doc='''Computes ``log(exp(x1) + exp(x2))`` elementwise.

    .. seealso:: :data:`numpy.logaddexp`

    ''')


logaddexp2 = core.create_ufunc(
    'clpy_logaddexp2',
    ('ff->f', 'dd->d'),
    'out0 = fmax(in0, in1) + log2(1 + exp2(-fabs(in0 - in1)))',
    doc='''Computes ``log2(exp2(x1) + exp2(x2))`` elementwise.

    .. seealso:: :data:`numpy.logaddexp2`

    ''')
