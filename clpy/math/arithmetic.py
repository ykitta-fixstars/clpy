from clpy import core


add = core.add


reciprocal = core.create_ufunc(
    'clpy_reciprocal',
    ('b', 'B', 'h', 'H', 'i', 'I', 'l', 'L', 'q', 'Q',
     ('e', 'out0 = 1 / in0'),
     ('f', 'out0 = 1 / in0'),
     ('d', 'out0 = 1 / in0'),
     ('F', 'out0 = in0_type(1) / in0'),
     ('D', 'out0 = in0_type(1) / in0')),
    'out0 = in0 == 0 ? 0 : (1 / in0)',
    doc='''Computes ``1 / x`` elementwise.

    .. seealso:: :data:`numpy.reciprocal`

    ''')


negative = core.negative


conj = core.conj


angle = core.angle


real = core.real


imag = core.imag


multiply = core.multiply


divide = core.divide


power = core.power


subtract = core.subtract


true_divide = core.true_divide


floor_divide = core.floor_divide


fmod = core.create_ufunc(
    'clpy_fmod',
    ('bb->b', 'BB->B', 'hh->h', 'HH->H', 'ii->i', 'II->I', 'll->l', 'LL->L',
     'qq->q', 'QQ->Q',
     ('ff->f', 'out0 = fmod(in0, in1)'),
     ('dd->d', 'out0 = fmod(in0, in1)')),
    'out0 = in1 == 0 ? 0 : fmod((double)in0, (double)in1)',
    doc='''Computes the remainder of C division elementwise.

    .. seealso:: :data:`numpy.fmod`

    ''')


modf = core.create_ufunc(
    'clpy_modf',
    (
        ('f->ff', 'float  iptr; out0 = modf(in0, &iptr); out1 = iptr'),
        ('d->dd', 'double iptr; out0 = modf(in0, &iptr); out1 = iptr'),
    ),
    doc='''Extracts the fractional and integral parts of an array elementwise.

    This ufunc returns two arrays.

    .. seealso:: :data:`numpy.modf`

    ''')


remainder = core.remainder
