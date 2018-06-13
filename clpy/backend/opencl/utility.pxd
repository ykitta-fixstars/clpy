include "types.pxi"

################################################################################
# helpers
cdef cl_uint GetDeviceMemBaseAddrAlign(cl_device_id device)
cdef GetDeviceAddressBits(cl_device_id device)

################################################################################
# utility
cdef void SetKernelArgLocalMemory(cl_kernel kernel, arg_index, size_t size)
cdef is_valid_kernel_name(name)
cdef cl_program CreateProgram(sources, cl_context context, num_devices, cl_device_id* devices_ptrs, options=*) except *
cdef GetProgramBuildLog(cl_program program)
cdef RunNDRangeKernel(
    cl_command_queue command_queue,
    cl_kernel kernel,
    size_t work_dim,
    size_t* global_work_offset,
    size_t* global_work_size,
    size_t* local_work_size,
    cl_uint num_events_in_wait_list,
    cl_event* event_wait_list)
