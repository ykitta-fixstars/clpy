#!/bin/bash
# Detailed output
set -x
# Exit immediately if an error has occurred
set -e

# Specify a python environment
export PATH="~/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
pyenv shell 3.6.5

# Set up a new temporary python modules environment
python -m venv venv
source venv/bin/activate


# Install dependencies
pip install -U pip
pip install Cython pytest

# Install clpy
python setup.py develop


# Run pytests
TEST_DIRS="
tests/clpy_tests/opencl_tests/
tests/clpy_tests/binary_tests/
"

TEST_FILES="
tests/clpy_tests/core_tests/test_core.py
tests/clpy_tests/core_tests/test_elementwise.py
tests/clpy_tests/core_tests/test_flags.py
tests/clpy_tests/core_tests/test_internal.py
tests/clpy_tests/core_tests/test_ndarray.py
tests/clpy_tests/core_tests/test_ndarray_contiguity.py
tests/clpy_tests/core_tests/test_ndarray_copy_and_view.py
tests/clpy_tests/core_tests/test_ndarray_elementwise_op.py
tests/clpy_tests/core_tests/test_ndarray_get.py
tests/clpy_tests/core_tests/test_ndarray_indexing.py
tests/clpy_tests/core_tests/test_ndarray_owndata.py
tests/clpy_tests/core_tests/test_ndarray_reduction.py
tests/clpy_tests/core_tests/test_ndarray_unary_op.py
tests/clpy_tests/core_tests/test_reduction.py
tests/clpy_tests/creation_tests/test_basic.py
tests/clpy_tests/creation_tests/test_matrix.py
tests/clpy_tests/creation_tests/test_ranges.py
"

# Ignore occurred errors below
set +e
# A filename to record stderrs
ERRORS_FILENAME=$WORKSPACE/erros.log
ERROR_HAS_OCCURRED=0

for d in $TEST_DIRS; do
  pushd $d
  python -m pytest >temporary_log 2>&1
  if [[ $? -ne 0 ]]; then
    cat temporary_log >> $ERRORS_FILENAME
    ERROR_HAS_OCCURRED=1
  fi
  popd
done

for f in $TEST_FILES; do
  pushd $(dirname $f)
  python -m pytest $(basename $f) >temporary_log 2>&1
  if [[ $? -ne 0 ]]; then
    cat temporary_log >> $ERRORS_FILENAME
    ERROR_HAS_OCCURRED=1
  fi
  popd
done

exit $ERROR_HAS_OCCURRED
