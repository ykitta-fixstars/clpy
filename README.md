# ClPy: OpenCL backend for CuPy

*ClPy* is an implementation of [CuPy](https://cupy.chainer.org/)'s OpenCL backend.
In other words, ClPy enables softwares written in CuPy to work also on OpenCL devices, not only on CUDA (NVIDIA) devices.

## Current status

Current ClPy is alpha version, forked from [CuPy v2.1.0](https://github.com/cupy/cupy/releases/tag/v2.1.0).
ClPy is still under development and works on only limited APIs.

* Basic [ndarray](https://docs-cupy.chainer.org/en/stable/reference/ndarray.html) are supported, but not perfectly
* Basic [universal functions](https://docs-cupy.chainer.org/en/stable/reference/ufunc.html) are supported, but not perfectly
* Simple [custom kernels](https://docs-cupy.chainer.org/en/stable/reference/kernel.html) are supported, but some custom kernel codes might be fail to compile and/or run
* Only SGEMM is supported in BLAS library
* Sparse matrix, dnn, rand libraries are not supported
* half and complex are not supported
* Works on only a single device
* No multiple command queue (Stream on CUDA)
* Dockerfile and some other files are just neglected thus don't work well

Original CuPy's tests are not passed perfectly. See current [CuPy's test and example results](https://github.com/fixstars/ClPy/wiki/cupy_test_example_results).

[Chainer](https://chainer.org/) works with limited situation.
Very simple and small examples are confirmed to work. See current [Chainer's test and example results](https://github.com/fixstars/ClPy/wiki/chainer_test_example_results).

## Recommended system

We develop and test ClPy in following environments.

* Primary machine
	* OS: Ubuntu 16.04.4 LTS
	* CPU: Core i7-4790
	* GPU: AMD Radeon R9 Fury X
	* SDK: AMD APP SDK 3.0
* Secondary machine
	* OS: CentOS 7.2.1511
	* CPU: Core i7-4790
	* GPU: NVIDIA GeForce GTX 1060
	* SDK: CUDA 9.2

We recommend those environments to all ClPy users. However, reports on other environments are welcome.

## Installation

First, install and setup OpenCL environment.
`cl.h` and OpenCL libs (`libOpenCL.so`) must be able to be included and linked without any special path settings.

For example with AMD APP SDK, you should set following environment variables:

```sh
export C_INCLUDE_PATH=${C_INCLUDE_PATH}:${AMDAPPSDKROOT}/include
export CPLUS_INCLUDE_PATH=${CPLUS_INCLUDE_PATH}:${AMDAPPSDKROOT}/include
export LIBRARY_PATH=${LIBRARY_PATH}:${AMDAPPSDKROOT}/lib/x86_64
```

and add ldconfig on `/etc/ldconf.so.d/` and `$ sudo ldconfig`.

After OpenCL is successfully installed, install ClPy.

```console
$ pip install cython
$ python setup.py install
```

If you want `cupy` alias, set `export CLPY_GENERATE_CUPY_ALIAS=1` before install.

## How to use

Just replace `cupy` to `clpy` in your python codes and run it (e.g. `import cupy` -> `import clpy`).
You don't need to replace codes if you install with `CLPY_GENERATE_CUPY_ALIAS=1`.

### Woking with Chainer

It's confirmed that ClPy works with [Chainer v3.3.0](https://github.com/chainer/chainer/tree/v3.3.0).

### Tests

```console
$ pip install pytest
$ cd tests/you/want
$ python -m pytest test_you_want.py
```

## Development

1. All source codes (including comments) and commit messages should be written in English.
2. Issues and pull requests are welcome in any languages (recommended in English or Japanese).
3. Detailed coding styles are same as [CuPy's](https://docs-cupy.chainer.org/en/stable/contribution.html#coding-guidelines). Read and follow the guidelines before submitting PRs.

## Future plan

We are developing v0.2.1beta for next release.

* Support Chainer's ImageNet example
* OpenCL version check and auto generated `api.pxd` from `cl.h` in the system
* Map buffer for host memory
* Support all BLAS API
* Improve `cupy` aliasing mechanism
* Update recommended system (Vega and Volta)
* -- and other functions and/or bug fixes that someone develops and/or requests..

We also plan to update CuPy's base version to v4 or v5 after above beta release.

Check github's issues and pull requests to get latest status.

## License

MIT License (see `LICENSE` file).
