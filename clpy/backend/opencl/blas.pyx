# -*- coding: utf-8 -*-
from string import Template
import sys

import numpy

import clpy.backend.opencl
cimport clpy.backend.opencl.api
cimport clpy.backend.opencl.utility
import clpy.backend.opencl.env
cimport clpy.backend.opencl.env
import clpy.backend.opencl.types
cimport clpy.backend.opencl.types
from clpy.backend.opencl.types cimport cl_event

_local_work_size = 16  # TODO(tomoya.sakai): GetDeviceInfo
_work_per_thread = 2

cdef void SetKernelArgWithScalarValue(clpy.backend.opencl.types.cl_kernel kernel, arg_index, _arg_value):
    ptr = _arg_value
    # Wrap _arg_value with NumPy container if _arg_value is a scalar value.
    if isinstance(_arg_value, int):
        if clpy.backend.opencl.types.device_typeof_size == 'uint':
            arg_value = numpy.uint32(_arg_value)
        elif clpy.backend.opencl.types.device_typeof_size == 'ulong':
            arg_value = numpy.uint64(_arg_value)
        else:
            raise "api_sizeof_size is illegal"
    elif isinstance(_arg_value, float):
        arg_value = numpy.float_(_arg_value)
    else:
        arg_value = _arg_value

    # Pass the pointer to arg_value if it is a NumPy-wrapped scalar.
    cdef size_t temporary = 0
    if numpy.issctype(type(arg_value)):
        size = arg_value.nbytes
        temporary = numpy.array(arg_value).ctypes.get_as_parameter().value
        clpy.backend.opencl.api.SetKernelArg(kernel, arg_index, size, <void*>temporary)
    else:
        raise ValueError("Type {0} is not a scalar value.".format(type(arg_value)))

cdef computeGlobalWorkItemSize(n, local_work_size):
    if n % local_work_size == 0:
        return n
    else:
        return n + (local_work_size - n % local_work_size)

cpdef sgemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc):
    '''
    A, B, C: ndarray
    '''

    m = numpy.intp(m)
    n = numpy.intp(n)
    k = numpy.intp(k)
    alpha = numpy.float32(alpha)
    beta = numpy.float32(beta)

    if transa == 'n' or transa == 0:
        name = 'n'
    elif transa == 't' or transa == 1:
        name = 't'
    else:
        raise ValueError("transa should be n(0) or t(1)")

    if transb == 'n' or transb == 0:
        name = name + 'n'
    elif transb == 't' or transb == 1:
        name = name + 't'
    else:
        raise ValueError("transb should be n(0) or t(1)")

    cdef size_t program_sizet = _sgemm_kernel[(name, alpha == 0.0, beta == 0.0)]
    cdef clpy.backend.opencl.types.cl_program program = <clpy.backend.opencl.types.cl_program>program_sizet
    cdef clpy.backend.opencl.types.cl_kernel kernel = clpy.backend.opencl.api.CreateKernel(
            program=program,
            kernel_name=b'dot_kernel'
            )
    SetKernelArgWithScalarValue(kernel, 0, m)
    SetKernelArgWithScalarValue(kernel, 1, n)
    SetKernelArgWithScalarValue(kernel, 2, k)
    SetKernelArgWithScalarValue(kernel, 3, alpha)
    cdef size_t Aptrtmp = A.data.buf.get()
    clpy.backend.opencl.api.SetKernelArg(kernel, 4, sizeof(void*), <void*>&Aptrtmp)
    SetKernelArgWithScalarValue(kernel, 5, lda)
    cdef size_t Bptrtmp = B.data.buf.get()
    clpy.backend.opencl.api.SetKernelArg(kernel, 6, sizeof(void*), <void*>&Bptrtmp)
    SetKernelArgWithScalarValue(kernel, 7, ldb)
    SetKernelArgWithScalarValue(kernel, 8, beta)
    cdef size_t Cptrtmp = C.data.buf.get()
    clpy.backend.opencl.api.SetKernelArg(kernel, 9, sizeof(void*), <void*>&Cptrtmp)
    SetKernelArgWithScalarValue(kernel, 10, ldc)
    SetKernelArgWithScalarValue(kernel, 11, A.data.cl_mem_offset() // A.itemsize)
    SetKernelArgWithScalarValue(kernel, 12, B.data.cl_mem_offset() // B.itemsize)
    SetKernelArgWithScalarValue(kernel, 13, C.data.cl_mem_offset() // C.itemsize)

    cdef size_t lws[2]
    lws[0] = _local_work_size
    lws[1] = _local_work_size / _work_per_thread

    cdef size_t gws[2]
    gws[0] = computeGlobalWorkItemSize(m, _local_work_size)
    gws[1] = computeGlobalWorkItemSize(n, _local_work_size) / _work_per_thread

    clpy.backend.opencl.utility.RunNDRangeKernel(
            command_queue=clpy.backend.opencl.env.get_command_queue(),
            kernel=kernel,
            work_dim=2,  # in 1, 2, 3
            global_work_offset=<size_t*>NULL,  # fixed
            global_work_size=&gws[0],
            local_work_size=&lws[0],
            num_events_in_wait_list=0,
            event_wait_list=<cl_event*>NULL
            )

cpdef size_t _generate_sgemm_kernel(transa, transb, alphaIsZero, betaIsZero):
    if transa == 'n' or transa == 0:
        indexA = 'I + (j + w * local_stride) * ld_a'
        name = 'n'
    elif transa == 't' or transa == 1:
        indexA = 'I * ld_a + (j + w * local_stride)'
        name = 't'

    if transb == 'n' or transb == 0:
        indexB = 'i + (J + w * local_stride) * ld_b'
        name = name + 'n'
    elif transb == 't' or transb == 1:
        indexB = 'i * ld_b + (J + w * local_stride)'
        name = name + 't'

    if betaIsZero:
        beta_expr = "acc[w]"
    elif not betaIsZero:
        beta_expr = "acc[w] + beta * C[C_offset + I + (J + w * local_stride)* ld_c]"

    if alphaIsZero:
        alpha_expr = ""
    elif not alphaIsZero:
        alpha_expr = Template('''
            const int numTiles = (mA  + local_work_size - 1)/ local_work_size;
            for(int t = 0;t < numTiles;t++){
                barrier(CLK_LOCAL_MEM_FENCE);
                for(int w = 0;w < work_per_thread;w++){
                    const size_t i = local_work_size * t + iSub;//tiled row
                    const size_t j = local_work_size * t + jSub;//tiled col
                    if ((I < nA) && (j + w * local_stride < mA)){
                        Asub[jSub + w * local_stride][iSub] = A[A_offset + ${indexA}]; // Asub(iSub, jSub) = A(I, j)
                    } else {
                        Asub[jSub + w * local_stride][iSub] = 0;
                    }
                    if ((i < nB) && (J + w * local_stride < mB)){
                        Bsub[jSub + w * local_stride][iSub] = B[B_offset + ${indexB}]; // Bsub(iSub, jSub) = B(i , J)
                    } else {
                        Bsub[jSub + w * local_stride][iSub] = 0;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                for(int k = 0;k < local_work_size;k++){
                    for(int w = 0;w < work_per_thread;w++){
                        acc[w] += Asub[k][iSub] * Bsub[jSub + w * local_stride][k];
                    }
                }
            }
        ''').substitute(indexA=indexA, indexB=indexB)
        beta_expr = "alpha * " + beta_expr

    dot_kernel_source = Template('''
    typedef ${typeof_size} kernel_arg_size_t;
    __kernel void dot_kernel(
        const kernel_arg_size_t nC, const kernel_arg_size_t mC, const kernel_arg_size_t mA,
        const float alpha,
        __global const float * const restrict A, const kernel_arg_size_t ld_a,
        __global const float * const restrict B, const kernel_arg_size_t ld_b,
        const float beta,
        __global float * const restrict C, const kernel_arg_size_t ld_c,
        const kernel_arg_size_t A_offset,
        const kernel_arg_size_t B_offset,
        const kernel_arg_size_t C_offset
        )
        {
            // na = M, mB = N, mA = K
            // A(nA = nC, mA     ) =  A(M, K)
            // B(nB = mA, mB = mC) =  B(K, N)
            // C(nC     , mC     ) =  C(M, N)
            const size_t local_work_size = ${local_work_size};
            const size_t iSub = get_local_id(0);
            const size_t jSub = get_local_id(1);
            const size_t I = local_work_size * get_group_id(0) + iSub;
            const size_t J = local_work_size * get_group_id(1) + jSub;
            const size_t nA = nC;
            const size_t nB = mA;
            const size_t mB = mC;
            const size_t work_per_thread = ${work_per_thread};
            const size_t local_stride= local_work_size / work_per_thread;

            __local float Asub[${local_work_size}][${local_work_size}];
            __local float Bsub[${local_work_size}][${local_work_size}];

            float acc[${work_per_thread}];
            for(int w = 0;w < work_per_thread;w++){
                acc[w] = 0.0f;
            }

            ${alpha_expr}

            for(int w = 0;w < work_per_thread;w++){
            if ((I < nC) && (J + w * local_stride < mC)){
                C[C_offset + I + (J + w * local_stride)* nC] = ${beta_expr};
            }
            }
        }
    ''').substitute(alpha_expr=alpha_expr, beta_expr=beta_expr, typeof_size=clpy.backend.opencl.types.device_typeof_size, local_work_size=str(_local_work_size), work_per_thread=str(_work_per_thread)).encode('utf-8')

    cdef clpy.backend.opencl.types.cl_program program = clpy.backend.opencl.utility.CreateProgram(
            sources=[dot_kernel_source],
            context=clpy.backend.opencl.env.get_context(),
            num_devices=clpy.backend.opencl.env.num_devices,
            devices_ptrs=clpy.backend.opencl.env.get_devices_ptrs()
            )
    return <size_t>program

_sgemm_kernel = {
    (name, alphaIsZero, betaIsZero):
        _generate_sgemm_kernel(name[0], name[1], alphaIsZero, betaIsZero)
    for name in ['nn', 'nt', 'tn', 'tt']
    for betaIsZero in [False, True]
    for alphaIsZero in [False, True]
}


def sgeam(transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc):
    '''
    A, B, C: ndarray
    '''

    m = numpy.intp(m)
    n = numpy.intp(n)
    alpha = numpy.float32(alpha)
    beta = numpy.float32(beta)

    if transa == 'n' or transa == 0:
        name = 'n'
    elif transa == 't' or transa == 1:
        name = 't'
    else:
        raise ValueError("transa should be n(0) or t(1)")

    if transb == 'n' or transb == 0:
        name = name + 'n'
    elif transb == 't' or transb == 1:
        name = name + 't'
    else:
        raise ValueError("transb should be n(0) or t(1)")

    cdef size_t program_sizet = _sgeam_kernel[(name, alpha == 0.0, beta == 0.0)]
    cdef clpy.backend.opencl.types.cl_program program = <clpy.backend.opencl.types.cl_program>program_sizet
    cdef clpy.backend.opencl.types.cl_kernel kernel = clpy.backend.opencl.api.CreateKernel(
            program=program,
            kernel_name=b'geam_kernel'
            )
    SetKernelArgWithScalarValue(kernel, 0, m)
    SetKernelArgWithScalarValue(kernel, 1, n)
    SetKernelArgWithScalarValue(kernel, 2, alpha)
    cdef size_t Aptrtmp = A.data.buf.get()
    clpy.backend.opencl.api.SetKernelArg(kernel, 3, sizeof(void*), <void*>&Aptrtmp)
    SetKernelArgWithScalarValue(kernel, 4, lda)
    SetKernelArgWithScalarValue(kernel, 5, beta)
    cdef size_t Bptrtmp = B.data.buf.get()
    clpy.backend.opencl.api.SetKernelArg(kernel, 6, sizeof(void*), <void*>&Bptrtmp)
    SetKernelArgWithScalarValue(kernel, 7, ldb)
    cdef size_t Cptrtmp = C.data.buf.get()
    clpy.backend.opencl.api.SetKernelArg(kernel, 8, sizeof(void*), <void*>&Cptrtmp)
    SetKernelArgWithScalarValue(kernel, 9, ldc)
    SetKernelArgWithScalarValue(kernel, 10, A.data.cl_mem_offset() // A.itemsize)
    SetKernelArgWithScalarValue(kernel, 11, B.data.cl_mem_offset() // B.itemsize)
    SetKernelArgWithScalarValue(kernel, 12, C.data.cl_mem_offset() // C.itemsize)

    cdef size_t gws[2]
    gws[0] = m
    gws[1] = n

    clpy.backend.opencl.utility.RunNDRangeKernel(
            command_queue=clpy.backend.opencl.env.get_command_queue(),
            kernel=kernel,
            work_dim=2,  # in 1, 2, 3
            global_work_offset=<size_t*>NULL,  # fixed
            global_work_size=&gws[0],
            local_work_size=<size_t*>NULL,
            num_events_in_wait_list=0,
            event_wait_list=<cl_event*>NULL
            )

cpdef size_t _generate_sgeam_kernel(transa, transb, alphaIsZero, betaIsZero):
    if transa == 'n' or transa == 0:
        indexA = 'A_offset + i + j*ld_a'  # column-major
        name = 'n'
    elif transa == 't' or transa == 1:
        indexA = 'A_offset + i*ld_a + j'  # row-major
        name = 't'

    if transb == 'n' or transb == 0:
        indexB = 'B_offset + i + j*ld_b'  # column-major
        name = name + 'n'
    elif transb == 't' or transb == 1:
        indexB = 'B_offset + i*ld_b + j'  # row-major
        name = name + 't'

    if alphaIsZero:
        alpha_expr = ""
    elif not alphaIsZero:
        alpha_expr = Template('''
            c_ij = alpha * A[${indexA}];
        ''').substitute(indexA=indexA)

    if betaIsZero:
        beta_expr = "c_ij"
    elif not betaIsZero:
        beta_expr = Template('''
            c_ij + beta * B[${indexB}]
        ''').substitute(indexB=indexB)

    geam_kernel_source = Template('''
        typedef ${typeof_size} kernel_arg_size_t;
        __kernel void geam_kernel(
        const kernel_arg_size_t n_c, const kernel_arg_size_t m_c,
        const float alpha,
        __global const float A[], const kernel_arg_size_t ld_a,
        const float beta,
        __global const float B[], const kernel_arg_size_t ld_b,
        __global       float C[], const kernel_arg_size_t ld_c,
        const kernel_arg_size_t A_offset,
        const kernel_arg_size_t B_offset,
        const kernel_arg_size_t C_offset
        )
        {
            const size_t i = get_global_id(0); // row number of C
            const size_t j = get_global_id(1); // col number of C
            if( i >= n_c ) return;
            if( j >= m_c ) return;

            float c_ij = 0.0;
            ${alpha_expr}
            C[C_offset + i + j*ld_c] = ${beta_expr}; // column-major
        }
    ''').substitute(alpha_expr=alpha_expr, beta_expr=beta_expr, typeof_size=clpy.backend.opencl.types.device_typeof_size).encode('utf-8')
    cdef clpy.backend.opencl.types.cl_program program = clpy.backend.opencl.utility.CreateProgram(
            sources=[geam_kernel_source],
            context=clpy.backend.opencl.env.get_context(),
            num_devices=clpy.backend.opencl.env.num_devices,
            devices_ptrs=clpy.backend.opencl.env.get_devices_ptrs()
            )

    return <size_t>program

_sgeam_kernel = {
    (name, alphaIsZero, betaIsZero):
        _generate_sgeam_kernel(name[0], name[1], alphaIsZero, betaIsZero)
    for name in ['nn', 'nt', 'tn', 'tt']
    for betaIsZero in [False, True]
    for alphaIsZero in [False, True]
}
