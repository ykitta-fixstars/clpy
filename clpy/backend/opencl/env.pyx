# -*- coding: utf-8 -*-
import atexit
import logging

from clpy.backend.opencl cimport api
##########################################
# Initialization
##########################################

logging.info("Get num_platforms...", end='')
cdef cl_uint num_platforms = api.GetPlatformIDs(0, <cl_platform_id*>NULL)
logging.info("SUCCESS")
logging.info("%d platform(s) found" % num_platforms)

logging.info("Get the first platform...", end='')
cdef cl_platform_id[1] __platforms_ptr
num_platforms = api.GetPlatformIDs(1, &__platforms_ptr[0])
cdef cl_platform_id primary_platform = __platforms_ptr[0]
logging.info("SUCCESS")

logging.info("Get num_devices...", end='')
cdef cl_uint __num_devices = api.GetDeviceIDs(
    primary_platform,
    CL_DEVICE_TYPE_DEFAULT,
    0,
    <cl_device_id*>NULL)
logging.info("SUCCESS")
logging.info("%d device(s) found" % __num_devices)

# clpy now supports only one device.
__num_devices = 1

logging.info("Get the first device...", end='')
cdef cl_device_id[1] __devices_ptr
api.GetDeviceIDs(
    primary_platform,
    1,
    __num_devices,
    &__devices_ptr[0])
num_devices = __num_devices     # provide as pure python interface
cdef cl_device_id __primary_device = __devices_ptr[0]
logging.info("SUCCESS")

logging.info("Create context...", end='')
cdef cl_context __context = api.CreateContext(
    properties=<cl_context_properties*>NULL,
    num_devices=__num_devices,
    devices=&__devices_ptr[0],
    pfn_notify=<void*>NULL,
    user_data=<void*>NULL)
logging.info("SUCCESS")

logging.info("Create command_queue...", end='')
cdef cl_command_queue __command_queue \
    = api.CreateCommandQueue(__context, __devices_ptr[0], 0)
logging.info("SUCCESS")

cdef cl_context get_context():
    return __context

cdef cl_command_queue get_command_queue():
    return __command_queue

cdef cl_device_id* get_devices_ptrs():
    return &__devices_ptr[0]

cdef cl_device_id get_primary_device():
    return __primary_device


def release():
    """Release command_queue and context automatically."""
    logging.info("Flush...", end='')
    api.Flush(__command_queue)
    logging.info("SUCCESS")

    logging.info("Finish...", end='')
    api.Finish(__command_queue)
    logging.info("SUCCESS")

    logging.info("Release command queue...", end='')
    api.ReleaseCommandQueue(__command_queue)
    logging.info("SUCCESS")

    logging.info("Release context...", end='')
    api.ReleaseContext(__context)
    logging.info("SUCCESS")

    # Release kernels, programs here if needed.

atexit.register(release)
