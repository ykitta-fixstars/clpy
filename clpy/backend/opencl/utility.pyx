cimport api
cimport env
import re
import os
import time
import tempfile
from cpython cimport array
from exceptions cimport check_status
from libc.stdlib cimport malloc
from libc.string cimport memcpy

################################################################################
# helpers

cdef cl_uint GetDeviceMemBaseAddrAlign(cl_device_id device):
    cdef cl_uint[1] valptrs
    cdef size_t[1] retptrs
    cdef cl_int status = api.clGetDeviceInfo(
            device,
            <cl_device_info>CL_DEVICE_MEM_BASE_ADDR_ALIGN,
            <size_t>sizeof(cl_uint),
            <void *>&valptrs[0],
            <size_t *>&retptrs[0]
            )
    check_status(status)

    ret = valptrs[0]
    return ret

cdef GetDeviceAddressBits(cl_device_id device):
    cdef cl_uint[1] valptrs
    cdef size_t[1] retptrs
    cdef cl_int status = api.clGetDeviceInfo(
            device,
            <cl_device_info>CL_DEVICE_ADDRESS_BITS,
            <size_t>sizeof(cl_uint),
            <void *>&valptrs[0],
            <size_t *>&retptrs[0]
            )
    check_status(status)

    ret = valptrs[0]
    return ret


################################################################################
# utility

cdef void SetKernelArgLocalMemory(cl_kernel kernel, arg_index, size_t size):
    api.SetKernelArg(kernel, arg_index, size, <void*>NULL)

cdef is_valid_kernel_name(name):
    return re.match('^[a-zA-Z_][a-zA-Z_0-9]*$', name) is not None

cdef cl_program CreateProgram(sources, cl_context context, num_devices, cl_device_id* devices_ptrs, options=b"") except *:
    cdef size_t length = len(sources)
    cdef char** src
    cdef size_t* src_size
    src = <char**>malloc(sizeof(const char*)*length)
    src_size = <size_t*>malloc(sizeof(size_t)*length)
    for i, s in enumerate(sources):
        src_size[i] = len(s)
        src[i] = <char*>malloc(sizeof(char)*src_size[i])
        memcpy(src[i], <char*>s, src_size[i])

    cdef bytes py_string
    if os.getenv("CLPY_SAVE_CL_KERNEL_SOURCE") == "1":
        for i in range(length):
            with open(tempfile.gettempdir() + "/" + str(time.monotonic()) + ".cl", 'w') as f:
                py_string = sources[i]
                f.write(py_string.decode('utf-8'))

    program = api.CreateProgramWithSource(context=context, count=length, strings=src, lengths=src_size)
    options = options + b'\0'
    cdef char* options_cstr = options

    from exceptions import OpenCLRuntimeError, OpenCLProgramBuildError
    try:
        api.BuildProgram(program, num_devices, devices_ptrs, options_cstr, <void*>NULL, <void*>NULL)
    except OpenCLRuntimeError as err:
        if err.status == CL_BUILD_PROGRAM_FAILURE:
            log = GetProgramBuildLog(program)
            err = OpenCLProgramBuildError(err, log)
        raise err

    return program

cdef GetProgramBuildLog(cl_program program):
    cdef size_t length;
    cdef cl_int status = api.clGetProgramBuildInfo(
            program,
            env.get_primary_device(),
            CL_PROGRAM_BUILD_LOG,
            0,
            NULL,
            &length)
    check_status(status)

    cdef array.array info = array.array('b')
    array.resize(info, length)
    status = api.clGetProgramBuildInfo(
            program,
            env.get_primary_device(),
            CL_PROGRAM_BUILD_LOG,
            length,
            info.data.as_voidptr,
            NULL)
    check_status(status)
    return info.tobytes().decode('utf8')


# Synchronize Running Kernel
cdef RunNDRangeKernel(
        cl_command_queue command_queue,
        cl_kernel kernel,
        size_t work_dim,
        size_t* global_work_offset,
        size_t* global_work_size,
        size_t* local_work_size,
        cl_uint num_events_in_wait_list,
        cl_event* event_wait_list):

    cdef cl_event[1] event
    api.EnqueueNDRangeKernel(
        command_queue=command_queue,
        kernel=kernel,
        work_dim=work_dim,
        global_work_offset=global_work_offset,
        global_work_size=global_work_size,
        local_work_size=local_work_size,
        num_events_in_wait_list=num_events_in_wait_list,
        event_wait_list=event_wait_list,
        event=&event[0]
    )
    api.WaitForEvents(1, &event[0])
