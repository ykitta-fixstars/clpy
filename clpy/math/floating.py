from clpy import core
from clpy.math import ufunc


signbit = core.create_ufunc(
    'clpy_signbit',
    ('f->?', 'd->?'),
    'out0 = signbit(in0)',
    doc='''Tests elementwise if the sign bit is set (i.e. less than zero).

    .. seealso:: :data:`numpy.signbit`

    ''')


copysign = ufunc.create_math_ufunc(
    'copysign', 2, 'clpy_copysign',
    '''Returns the first argument with the sign bit of the second elementwise.

    .. seealso:: :data:`numpy.copysign`

    ''')


ldexp = core.create_ufunc(
    'clpy_ldexp',
    ('fi->f', 'fl->f', 'di->d', 'dq->d'),
    'out0 = ldexp(in0, in1)',
    doc='''Computes ``x1 * 2 ** x2`` elementwise.

    .. seealso:: :data:`numpy.ldexp`

    ''')


frexp = core.create_ufunc(
    'clpy_frexp',
    ('f->fi', 'd->di'),
    'int nptr; out0 = frexp(in0, &nptr); out1 = nptr',
    doc='''Decomposes each element to mantissa and two's exponent.

    This ufunc outputs two arrays of the input dtype and the ``int`` dtype.

    .. seealso:: :data:`numpy.frexp`

    ''')


nextafter = ufunc.create_math_ufunc(
    'nextafter', 2, 'clpy_nextafter',
    '''Computes the nearest neighbor float values towards the second argument.

    .. seealso:: :data:`numpy.nextafter`

    ''')
