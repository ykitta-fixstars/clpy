# Functions from the following NumPy document
# http://docs.scipy.org/doc/numpy/reference/routines.linalg.html

# "NOQA" to suppress flake8 warning
from clpy.linalg import decomposition  # NOQA
from clpy.linalg import eigenvalue  # NOQA
from clpy.linalg import einsum  # NOQA
from clpy.linalg import norms  # NOQA
from clpy.linalg.norms import det  # NOQA
from clpy.linalg.norms import matrix_rank  # NOQA
from clpy.linalg.norms import norm  # NOQA
from clpy.linalg.norms import slogdet  # NOQA
from clpy.linalg import product  # NOQA
from clpy.linalg import solve  # NOQA

from clpy.linalg.decomposition import cholesky  # NOQA
from clpy.linalg.decomposition import qr  # NOQA
from clpy.linalg.decomposition import svd  # NOQA

from clpy.linalg.eigenvalue import eigh  # NOQA
from clpy.linalg.eigenvalue import eigvalsh  # NOQA

from clpy.linalg.solve import inv  # NOQA
from clpy.linalg.solve import pinv  # NOQA
from clpy.linalg.solve import solve  # NOQA
from clpy.linalg.solve import tensorsolve  # NOQA
