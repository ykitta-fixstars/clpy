# -*- coding: utf-8 -*-
import cython
cimport clpy.backend.opencl.utility
import clpy.backend.opencl.env
cimport clpy.backend.opencl.env

cdef __device_typeof_size():
    host_size_t_bits = cython.sizeof(Py_ssize_t)*8
    device_address_bits = clpy.backend.opencl.utility.GetDeviceAddressBits(
        clpy.backend.opencl.env.get_primary_device())
    if host_size_t_bits != device_address_bits:
        raise "Host's size_t is different from device's size_t."

    if device_address_bits == 32:
        return 'uint'
    elif device_address_bits == 64:
        return 'ulong'
    else:
        raise "There is no type of size_t."

device_typeof_size = __device_typeof_size()
