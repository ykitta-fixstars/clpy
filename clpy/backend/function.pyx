# distutils: language = c++

import numpy
import six

cimport cpython
from libcpp cimport vector
import cython

# from clpy.cuda cimport driver
from clpy.core cimport core
import clpy.core
import clpy.backend.opencl
cimport clpy.backend.opencl.api
cimport clpy.backend.opencl.utility
import clpy.backend.opencl.env
cimport clpy.backend.opencl.env
import clpy.backend.opencl.types
from clpy.backend.opencl.types cimport cl_event

cdef inline size_t _get_stream(strm) except *:
    return 0 if strm is None else strm.ptr


DEF MAX_NDIM = 25

cdef struct _CIndexer:
    Py_ssize_t size
    Py_ssize_t shape_and_index[MAX_NDIM * 2]

cdef struct _CArray:
    Py_ssize_t offset
    Py_ssize_t shape_and_index[MAX_NDIM * 2]

cdef struct _CArray0:
    char unused

cdef void _launch(clpy.backend.opencl.types.cl_kernel kernel, global_work_size, local_work_size, args, Py_ssize_t local_mem) except *:
    global_dim = len(global_work_size)
    local_dim = len(local_work_size)
    if global_dim < 1 or 3 < global_dim:
        raise ValueError("Global workitem dimension should be 1,2,3 but {0} was given".format(global_dim))
    elif local_dim < 0 or 3 < local_dim:
        raise ValueError("Local workitem dimension should be 0,1,2,3 but {0} was given".format(local_dim))
    elif local_dim > 0 and global_dim != local_dim:
        raise ValueError("global_work_size dim is {0} but local is {1}".format(global_dim, local_dim))

    cdef size_t i = 0
    cdef _CIndexer indexer  # to keep lifetime until SetKernelArg
    cdef _CArray arrayInfo  # to keep lifetime until SetKernelArg
    cdef size_t ptr = 0
    cdef size_t buffer_object = 0
    for a in args:
        if isinstance(a, core.ndarray):
            buffer_object = a.data.buf.get()
            clpy.backend.opencl.api.SetKernelArg(kernel, i, sizeof(void*), <void*>&buffer_object)
            i+=1

            ndim = len(a.strides)
            for d in range(ndim):
                if a.strides[d] % a.itemsize != 0:
                    raise ValueError("Stride of dim {0} = {1}, but item size is {2}".format(d, a.strides[d], a.itemsize))
                arrayInfo.shape_and_index[d] = a.shape[d]
                arrayInfo.shape_and_index[d + ndim] = a.strides[d] // a.itemsize
            arrayInfo.offset = a.data.cl_mem_offset() // a.itemsize
            clpy.backend.opencl.api.SetKernelArg(kernel, i, cython.sizeof(Py_ssize_t)*(2*ndim+1), <void*>&arrayInfo)
        elif isinstance(a, clpy.core.core.LocalMem):
            clpy.backend.opencl.utility.SetKernelArgLocalMemory(kernel, i, local_mem)
        else:
            if isinstance(a, core.Indexer):
                for d in range(a.ndim):
                    indexer.shape_and_index[d] = a.shape[d]
                ptr = <size_t>&indexer
                indexer.size = a.size
                size = a.get_size()
            else:
                if isinstance(a, clpy.core.core.Size_t):
                    if clpy.backend.opencl.types.device_typeof_size == 'uint':
                        a = numpy.uint32(a.val)
                    elif clpy.backend.opencl.types.device_typeof_size == 'ulong':
                        a = numpy.uint64(a.val)
                    else:
                        raise "api_sizeof_size is illegal"
                elif isinstance(a, int):
                    a = numpy.int_(a)
                elif isinstance(a, float):
                    a = numpy.float_(a)
                elif isinstance(a, bool):
                    a = numpy.bool_(a)

                if numpy.issctype(type(a)):
                    ptr = <size_t>numpy.array(a).ctypes.get_as_parameter().value
                    size = a.nbytes
                else:
                    raise TypeError('Unsupported type %s' % type(a))
            clpy.backend.opencl.api.SetKernelArg(kernel, i, size, <void*>ptr)
        i+=1

    cdef size_t gws[3]
    for i in range(global_dim):
        gws[i] = global_work_size[i]

    # local_work_size is a python list with the size either 0 or 1.
    cdef size_t lws[1]
    cdef size_t* lws_ptr
    if len(local_work_size) > 0:
        lws[0] = local_work_size[0]
        lws_ptr = &lws[0]
    else:
        lws_ptr = <size_t*>NULL

    clpy.backend.opencl.utility.RunNDRangeKernel(
        command_queue=clpy.backend.opencl.env.get_command_queue(),
        kernel=kernel,
        work_dim=global_dim,
        global_work_offset=<size_t*>NULL,
        global_work_size=&gws[0],
        local_work_size=lws_ptr,
        num_events_in_wait_list=0,
        event_wait_list=<cl_event*>NULL)


cdef class Function:

    """CUDA kernel function."""

    def __init__(self, Module module, str funcname):
        self.module = module  # to keep module loaded
        self.kernel = clpy.backend.opencl.api.CreateKernel(module.program, funcname.encode('utf-8'))

    def __call__(self, tuple grid, tuple block, args, size_t shared_mem=0,
                 stream=None):
        raise NotImplementedError("clpy does not support this")
#        grid = (grid + (1, 1))[:3]
#        block = (block + (1, 1))[:3]
#        s = _get_stream(stream)
#        _launch(
#            self.ptr,
#            max(1, grid[0]), max(1, grid[1]), max(1, grid[2]),
#            max(1, block[0]), max(1, block[1]), max(1, block[2]),
#            args, shared_mem, s)

    cpdef linear_launch(self, size_t size, args, size_t local_mem=0, size_t local_size=0):
        # TODO(beam2d): Tune it
        if local_size == 0:
            local_work_size = []
        else:
            local_work_size = [local_size, ]
        _launch(self.kernel, [size, ], local_work_size, args, local_mem)


cdef class Module:

    """CUDA kernel module."""

    def __init__(self):
        pass

    def __del__(self):
        pass

    cpdef load_file(self, str filename):
        raise NotImplementedError("clpy does not support this")
#        self.ptr = driver.moduleLoad(filename)

    cpdef load(self, bytes cubin):
        raise NotImplementedError("clpy does not support this")
#        self.ptr = driver.moduleLoadData(cubin)

    cpdef get_global_var(self, str name):
        raise NotImplementedError("clpy does not support this")
#        return driver.moduleGetGlobal(self.ptr, name)

    cdef set(self, clpy.backend.opencl.types.cl_program program):
        self.program = program

    cpdef get_function(self, str name):
        return Function(self, name)


cdef class LinkState:

    """CUDA link state."""

    def __init__(self):
        raise NotImplementedError("clpy does not support this")
#        self.ptr = driver.linkCreate()

    def __del__(self):
        if self.ptr:
            raise NotImplementedError("clpy does not support this")
#            driver.linkDestroy(self.ptr)
            self.ptr = 0

    cpdef add_ptr_data(self, unicode data, unicode name):
        cdef bytes data_byte = data.encode()
        raise NotImplementedError("clpy does not support this")
#        driver.linkAddData(self.ptr, driver.CU_JIT_INPUT_PTX, data_byte, name)

    cpdef bytes complete(self):
        raise NotImplementedError("clpy does not support this")
#        cubin = driver.linkComplete(self.ptr)
        return cubin
