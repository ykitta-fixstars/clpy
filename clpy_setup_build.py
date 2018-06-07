from __future__ import print_function
from distutils import ccompiler
from distutils import sysconfig
import os
from os import path
import sys

import pkg_resources
import setuptools
from setuptools.command import build_ext
from setuptools.command import sdist

from install import build
from install import utils


required_cython_version = pkg_resources.parse_version('0.24.0')
ignore_cython_versions = [
    pkg_resources.parse_version('0.27.0'),
]

MODULES = [
    {
        'name': 'opencl',
        'file': [
            'clpy.core.core',
            'clpy.core.flags',
            'clpy.core.internal',
            # 'clpy.backend.cublas',
            # 'clpy.backend.curand',
            # 'clpy.backend.cusparse',
            'clpy.backend.device',
            # 'clpy.backend.driver',
            'clpy.backend.memory',
            # 'clpy.backend.memory_hook',
            # 'clpy.backend.nvrtc',
            'clpy.backend.pinned_memory',
            # 'clpy.backend.profiler',
            # 'clpy.backend.nvtx',
            'clpy.backend.function',
            # 'clpy.backend.runtime',
            'clpy.backend.opencl.api',
            'clpy.backend.opencl.blas',
            'clpy.backend.opencl.env',
            'clpy.backend.opencl.types',
            'clpy.backend.opencl.utility',
            'clpy.backend.opencl.exceptions',
            'clpy.util',
            'clpy.testing.bufio',
        ],
        'include': [
            'CL/cl.h'
        ],
        'libraries': [
            'OpenCL'
        ],
        'check_method': build.check_opencl_version,
    },
]


def ensure_module_file(file):
    if isinstance(file, tuple):
        return file
    else:
        return (file, [])


def module_extension_name(file):
    return ensure_module_file(file)[0]


def module_extension_sources(file, use_cython):
    pyx, others = ensure_module_file(file)
    ext = '.pyx' if use_cython else '.cpp'
    pyx = path.join(*pyx.split('.')) + ext

    return [pyx] + others


def check_readthedocs_environment():
    return os.environ.get('READTHEDOCS', None) == 'True'


def check_library(compiler, includes=(), libraries=(),
                  include_dirs=(), library_dirs=()):

    source = ''.join(['#include <%s>\n' % header for header in includes])
    source += 'int main(int argc, char* argv[]) {return 0;}'
    try:
        # We need to try to build a shared library because distutils
        # uses different option to build an executable and a shared library.
        # Especially when a user build an executable, distutils does not use
        # LDFLAGS environment variable.
        build.build_shlib(compiler, source, libraries,
                          include_dirs, library_dirs)
    except Exception as e:
        print(e)
        return False
    return True


def make_extensions(options, compiler, use_cython):
    """Produce a list of Extension instances which passed to cythonize()."""

    settings = build.get_compiler_setting()

    include_dirs = settings['include_dirs']

    settings['include_dirs'] = [
        x for x in include_dirs if path.exists(x)]
    settings['library_dirs'] = [
        x for x in settings['library_dirs'] if path.exists(x)]
    if sys.platform != 'win32':
        settings['runtime_library_dirs'] = settings['library_dirs']
    if sys.platform == 'darwin':
        args = settings.setdefault('extra_link_args', [])
        args.append(
            '-Wl,' + ','.join('-rpath,' + p
                              for p in settings['library_dirs']))
        # -rpath is only supported when targetting Mac OS X 10.5 or later
        args.append('-mmacosx-version-min=10.5')

    # This is a workaround for Anaconda.
    # Anaconda installs libstdc++ from GCC 4.8 and it is not compatible
    # with GCC 5's new ABI.
    settings['define_macros'].append(('_GLIBCXX_USE_CXX11_ABI', '0'))

    # In the environment with CUDA 7.5 on Ubuntu 16.04, gcc5.3 does not
    # automatically deal with memcpy because string.h header file has
    # been changed. This is a workaround for that environment.
    # See details in the below discussions:
    # https://github.com/BVLC/caffe/issues/4046
    # https://groups.google.com/forum/#!topic/theano-users/3ihQYiTRG4E
    settings['define_macros'].append(('_FORCE_INLINES', '1'))

    if options['linetrace']:
        settings['define_macros'].append(('CYTHON_TRACE', '1'))
        settings['define_macros'].append(('CYTHON_TRACE_NOGIL', '1'))

    ret = []
    for module in MODULES:
        print('Include directories:', settings['include_dirs'])
        print('Library directories:', settings['library_dirs'])

        err = False
        if not check_library(compiler,
                             includes=module['include'],
                             include_dirs=settings['include_dirs']):
            utils.print_warning(
                'Include files not found: %s' % module['include'],
                'Skip installing %s support' % module['name'],
                'Check your CFLAGS environment variable')
            err = True
        elif not check_library(compiler,
                               libraries=module['libraries'],
                               library_dirs=settings['library_dirs']):
            utils.print_warning(
                'Cannot link libraries: %s' % module['libraries'],
                'Skip installing %s support' % module['name'],
                'Check your LDFLAGS environment variable')
            err = True
        elif('check_method' in module and
             not module['check_method'](compiler, settings)):
            err = True

        if err:
            if module['name'] == 'opencl':
                raise Exception('Your OpenCL environment is invalid. '
                                'Please check above error log.')
            else:
                # Other modules are optional. They are skipped.
                continue

        s = settings.copy()
        s['libraries'] = module['libraries']

        for f in module['file']:
            name = module_extension_name(f)
            sources = module_extension_sources(f, use_cython)
            extension = setuptools.Extension(name, sources, **s)
            ret.append(extension)

    return ret


def parse_args():
    clpy_profile = '--clpy-profile' in sys.argv
    if clpy_profile:
        sys.argv.remove('--clpy-profile')
    clpy_coverage = '--clpy-coverage' in sys.argv
    if clpy_coverage:
        sys.argv.remove('--clpy-coverage')

    arg_options = {
        'profile': clpy_profile,
        'linetrace': clpy_coverage,
        'annotate': clpy_coverage,
    }
    return arg_options


clpy_setup_options = parse_args()
print('Options:', clpy_setup_options)

try:
    import Cython
    import Cython.Build
    cython_version = pkg_resources.parse_version(Cython.__version__)
    cython_available = (
        cython_version >= required_cython_version and
        cython_version not in ignore_cython_versions)
except ImportError:
    cython_available = False


def cythonize(extensions, arg_options):
    directive_keys = ('linetrace', 'profile')
    directives = {key: arg_options[key] for key in directive_keys}

    cythonize_option_keys = ('annotate',)
    cythonize_options = {key: arg_options[key]
                         for key in cythonize_option_keys}

    return Cython.Build.cythonize(
        extensions, verbose=True,
        compiler_directives=directives, **cythonize_options)


def check_extensions(extensions):
    for x in extensions:
        for f in x.sources:
            if not path.isfile(f):
                raise RuntimeError(
                    'Missing file: %s\n' % f +
                    'Please install Cython %s. ' % required_cython_version +
                    'Please also check the version of Cython.\n' +
                    'See ' +
                    'https://docs-clpy.chainer.org/en/stable/install.html')


def get_ext_modules(use_cython=False):
    arg_options = clpy_setup_options

    # We need to call get_config_vars to initialize _config_vars in distutils
    # see #1849
    sysconfig.get_config_vars()
    compiler = ccompiler.new_compiler()
    sysconfig.customize_compiler(compiler)

    extensions = make_extensions(arg_options, compiler, use_cython)

    return extensions


class sdist_with_cython(sdist.sdist):

    """Custom `sdist` command with cyhonizing."""

    def __init__(self, *args, **kwargs):
        if not cython_available:
            raise RuntimeError('Cython is required to make sdist.')
        ext_modules = get_ext_modules(True)  # get .pyx modules
        cythonize(ext_modules, clpy_setup_options)
        sdist.sdist.__init__(self, *args, **kwargs)


class custom_build_ext(build_ext.build_ext):

    """Custom `build_ext` command to include CUDA C source files."""

    def run(self):
        if cython_available:
            ext_modules = get_ext_modules(True)  # get .pyx modules
            cythonize(ext_modules, clpy_setup_options)
        check_extensions(self.extensions)
        build_ext.build_ext.run(self)
