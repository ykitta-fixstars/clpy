#!/usr/bin/env python

import imp
import os
from setuptools import setup
import sys

import clpy_setup_build


if sys.version_info[:3] == (3, 5, 0):
    if not int(os.getenv('CUPY_PYTHON_350_FORCE', '0')):
        msg = """
CuPy does not work with Python 3.5.0.

We strongly recommend to use another version of Python.
If you want to use CuPy with Python 3.5.0 at your own risk,
set 1 to CUPY_PYTHON_350_FORCE environment variable."""
        print(msg)
        sys.exit(1)

generate_cupy_alias = (os.getenv('CLPY_GENERATE_CUPY_ALIAS') == '1')

setup_requires = [
    'fastrlock>=0.3',
]
install_requires = [
    'numpy>=1.9.0',
    'six>=1.9.0',
    'fastrlock>=0.3',
]

ext_modules = clpy_setup_build.get_ext_modules()
build_ext = clpy_setup_build.custom_build_ext
sdist = clpy_setup_build.sdist_with_cython

here = os.path.abspath(os.path.dirname(__file__))
__version__ = imp.load_source(
    '_version', os.path.join(here, 'clpy', '_version.py')).__version__

packages_clpy = [
    'clpy',
    'clpy.binary',
    'clpy.core',
    'clpy.creation',
    'clpy.backend',
    #   'clpy.backend.memory_hooks',
    #   'clpy.ext',
    'clpy.indexing',
    #   'clpy.io',
    'clpy.linalg',
    'clpy.logic',
    'clpy.manipulation',
    'clpy.math',
    'clpy.backend.opencl',
    #   'clpy.padding',
    #   'clpy.prof',
    'clpy.random',
    'clpy.sorting',
    'clpy.sparse',
    'clpy.statistics',
    'clpy.testing'
]

packages_cupy_aliasing = [
    'cupy',
    'cupy.binary',
    'cupy.core',
    'cupy.creation',
    'cupy.backend',
    #   'cupy.backend.memory_hooks',
    #   'cupy.ext',
    'cupy.indexing',
    #   'cupy.io',
    'cupy.linalg',
    'cupy.logic',
    'cupy.manipulation',
    'cupy.math',
    'cupy.backend.opencl',
    'cupy.cuda',
    #   'cupy.padding',
    #   'cupy.prof',
    'cupy.random',
    'cupy.sorting',
    'cupy.sparse',
    'cupy.statistics',
    'cupy.testing'
]

if generate_cupy_alias:
    packages = packages_clpy + packages_cupy_aliasing
else:
    packages = packages_clpy

setup(
    name='clpy',
    version=__version__,
    description='ClPy: OpenCL backend for CuPy',
    url='https://github.com/fixstars/clpy',
    author='The University of Tokyo, '
           'National Institute of Advanced Industrial Science and Technology, '
           'Fixstars Corporation',
    maintainer='Fixstars Corporation',
    maintainer_email='clpy@fixstars.com',
    license='MIT License',
    packages=packages,
    package_data={
        'clpy': [
            # 'core/include/clpy/complex/arithmetic.h',
            # 'core/include/clpy/complex/catrig.h',
            # 'core/include/clpy/complex/catrigf.h',
            # 'core/include/clpy/complex/ccosh.h',
            # 'core/include/clpy/complex/ccoshf.h',
            # 'core/include/clpy/complex/cexp.h',
            # 'core/include/clpy/complex/cexpf.h',
            # 'core/include/clpy/complex/clog.h',
            # 'core/include/clpy/complex/clogf.h',
            # 'core/include/clpy/complex/complex.h',
            # 'core/include/clpy/complex/complex_inl.h',
            # 'core/include/clpy/complex/cpow.h',
            # 'core/include/clpy/complex/cproj.h',
            # 'core/include/clpy/complex/csinh.h',
            # 'core/include/clpy/complex/csinhf.h',
            # 'core/include/clpy/complex/csqrt.h',
            # 'core/include/clpy/complex/csqrtf.h',
            # 'core/include/clpy/complex/ctanh.h',
            # 'core/include/clpy/complex/ctanhf.h',
            # 'core/include/clpy/complex/math_private.h',
            'core/include/clpy/carray.clh',
        ],
    },
    zip_safe=False,
    setup_requires=setup_requires,
    install_requires=install_requires,
    tests_require=['mock',
                   'pytest'],
    ext_modules=ext_modules,
    cmdclass={'build_ext': build_ext,
              'sdist': sdist},
)
