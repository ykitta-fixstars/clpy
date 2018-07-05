import os

from clpy import backend

from clpy.backend cimport function
# from clpy.backend cimport runtime
cimport clpy.backend.opencl.api
cimport clpy.backend.opencl.utility
import clpy.backend.opencl.env
cimport clpy.backend.opencl.env

import warnings
import functools
import operator
import subprocess
import tempfile
import time

cdef class Indexer:
    def __init__(self, tuple shape):
        cdef Py_ssize_t size = 1
        for s in shape:
            size *= s
        self.shape = shape
        self.size = size

    @property
    def ndim(self):
        return len(self.shape)

    def get_size(self):
        return cython.sizeof(Py_ssize_t) * (1 + self.ndim * 2)  # size + shape_and_stride

cdef class Size_t:
    def __init__(self, size_t val):
        self.val = val

    @property
    def ndim(self):
        return 1

cdef class LocalMem:
    @property
    def ndim(self):
        return 1

cdef list _clpy_header_list = [
# TODO(LWisteria): implement complex
#    'clpy/complex.cuh',
    'clpy/carray.clh',
]
cdef str _clpy_header = ''.join(
    ['#include <%s>\n' % i for i in _clpy_header_list])

# This is indirect include header list.
# These header files are subject to a hash key.
cdef list _clpy_extra_header_list = [
# TODO(LWisteria): implement complex
#    'clpy/complex/complex.h',
#    'clpy/complex/math_private.h',
#    'clpy/complex/complex_inl.h',
#    'clpy/complex/arithmetic.h',
#    'clpy/complex/cproj.h',
#    'clpy/complex/cexp.h',
#    'clpy/complex/cexpf.h',
#    'clpy/complex/clog.h',
#    'clpy/complex/clogf.h',
#    'clpy/complex/cpow.h',
#    'clpy/complex/ccosh.h',
#    'clpy/complex/ccoshf.h',
#    'clpy/complex/csinh.h',
#    'clpy/complex/csinhf.h',
#    'clpy/complex/ctanh.h',
#    'clpy/complex/ctanhf.h',
#    'clpy/complex/csqrt.h',
#    'clpy/complex/csqrtf.h',
#    'clpy/complex/catrig.h',
#    'clpy/complex/catrigf.h',
]

cdef str _header_path_cache = None
cdef str _header_source = None


cpdef str _get_header_dir_path():
    global _header_path_cache
    if _header_path_cache is None:
        # Cython cannot use __file__ in global scope
        _header_path_cache = os.path.abspath(
            os.path.join(os.path.dirname(__file__), 'include'))
    return _header_path_cache


cpdef str _get_header_source():
    global _header_source
    if _header_source is None:
        source = []
        base_path = _get_header_dir_path()
        for file_path in _clpy_header_list + _clpy_extra_header_list:
            header_path = os.path.join(base_path, file_path)
            with open(header_path) as header_file:
                source.append(header_file.read())
        _header_source = '\n'.join(source)
    return _header_source


cdef str _cuda_path = None

cpdef str _get_cuda_path():
    global _cuda_path
    if _cuda_path is None:
        _cuda_path = os.getenv('CUDA_PATH', None)
        if _cuda_path is not None:
            return _cuda_path

        for p in os.getenv('PATH', '').split(os.pathsep):
            for cmd in ('nvcc', 'nvcc.exe'):
                nvcc_path = os.path.join(p, cmd)
                if not os.path.exists(nvcc_path):
                    continue
                nvcc_dir = os.path.dirname(
                    os.path.abspath(nvcc_path))
                _cuda_path = os.path.normpath(
                    os.path.join(nvcc_dir, '..'))
                return _cuda_path

        if os.path.exists('/usr/local/cuda'):
            _cuda_path = '/usr/local/cuda'

    return _cuda_path

class TempFile:
    def __init__(self, filename, source):
        self.fn = filename
        self.s = source
    def __enter__(self):
        with open(self.fn, 'w') as f:
            f.write(self.s)
    def __exit__(self, exception_type, exception_value, traceback):
        if os.getenv("CLPY_SAVE_PRE_KERNEL_SOURCE") != "1":
            os.remove(self.fn)

cpdef function.Module compile_with_cache(
        str source, tuple options=(), arch=None, cachd_dir=None):
    source = _clpy_header + '\nstatic void __clpy_begin_print_out() __attribute__((annotate("clpy_begin_print_out")));\n' + source + '\nstatic void __clpy_end_print_out() __attribute__((annotate("clpy_end_print_out")));\n'

    filename = tempfile.gettempdir() + "/" + str(time.monotonic()) + ".cpp"

    with TempFile(filename, source) as tf:
        proc = subprocess.Popen('{0}/ultima/ultima {1} -- -I {0}/clpy/core/include'.format(clpy.__path__[0]+"/../", filename).strip().split(" "), stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        try:
            source, errstream = proc.communicate(timeout=15)
            proc.wait()
        except subprocess.TimeoutExpired:
            proc.kill()
            source, errstream = proc.communicate()

        if proc.returncode != 0 and len(errstream) > 0:
            raise clpy.backend.ultima.exceptions.UltimaRuntimeError(proc.returncode, errstream)

    extra_source = _get_header_source()
    options += ('-I%s' % _get_header_dir_path(),)
    options += (' -cl-fp32-correctly-rounded-divide-sqrt', )
    optionStr = functools.reduce(operator.add, options)

    program = clpy.backend.opencl.utility.CreateProgram([source.encode('utf-8')], clpy.backend.opencl.env.get_context(), clpy.backend.opencl.env.num_devices, clpy.backend.opencl.env.get_devices_ptrs(), optionStr.encode('utf-8'))
    cdef function.Module module = function.Module()
    module.set(program)
    return module
