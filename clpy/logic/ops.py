from clpy import core

logical_and = core.create_comparison(
    'logical_and', '&&',
    '''Computes the logical AND of two arrays.

    .. seealso:: :data:`numpy.logical_and`

    ''')


logical_or = core.create_comparison(
    'logical_or', '||',
    '''Computes the logical OR of two arrays.

    .. seealso:: :data:`numpy.logical_or`

    ''')


logical_not = core.create_ufunc(
    'clpy_logical_not',
    ('?->?', 'b->?', 'B->?', 'h->?', 'H->?', 'i->?', 'I->?', 'l->?', 'L->?',
     'q->?', 'Q->?', 'f->?', 'd->?'),
    'out0 = !in0',
    doc='''Computes the logical NOT of an array.

    .. seealso:: :data:`numpy.logical_not`

    ''')


logical_xor = core.create_ufunc(
    'clpy_logical_xor',
    ('??->?', 'bb->?', 'BB->?', 'hh->?', 'HH->?', 'ii->?', 'II->?', 'll->?',
     'LL->?', 'qq->?', 'QQ->?', 'ff->?', 'dd->?'),
    'out0 = !in0 != !in1',
    doc='''Computes the logical XOR of two arrays.

    .. seealso:: :data:`numpy.logical_xor`

    ''')
