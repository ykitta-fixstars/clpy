include "types.pxi"

cdef cl_context get_context()
cdef cl_command_queue get_command_queue()
cdef cl_device_id* get_devices_ptrs()
cdef cl_device_id get_primary_device()
