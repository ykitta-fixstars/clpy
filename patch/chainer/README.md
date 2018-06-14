# Patches on Chainer codes

The current version of ClPy requires some patches on Chainer codes.
Instead of installing Chainer through `pip`, please follow these instructions.

If you have already installed Chainer, please uninstall it:

```console
$ pip uninstall chainer
```

Fetch **v3.3.0** of Chainer, apply patches on it and install it:
```console
$ git clone --depth=1 -b v3.3.0 https://github.com/chainer/chainer.git
$ cd chainer
$ patch -p1 < /path/to/clpy/patch/chainer/lstm.patch
$ patch -p1 < /path/to/clpy/patch/chainer/hierarchical_softmax.patch
$ python setup.py install
```
