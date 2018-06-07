cimport clpy.backend.opencl.api
cimport clpy.backend.opencl.types

cdef class CPointer:
    cdef void* ptr


cdef class Function:

    cdef:
        Module module
        clpy.backend.opencl.types.cl_kernel kernel

    cpdef linear_launch(self, size_t size, args, size_t local_mem=*, size_t local_size=*)


cdef class Module:

    cdef:
        clpy.backend.opencl.types.cl_program program

    cpdef load_file(self, str filename)
    cpdef load(self, bytes cubin)
    cpdef get_global_var(self, str name)
    cpdef get_function(self, str name)
    cdef set(self, clpy.backend.opencl.types.cl_program program)


cdef class LinkState:

    cdef:
        public size_t ptr

    cpdef add_ptr_data(self, unicode data, unicode name)
    cpdef bytes complete(self)
