include "types.pxi"

cdef extern from "CL/cl.h":
    cl_int clGetPlatformIDs(cl_uint, cl_platform_id *, cl_uint *)
    cl_int clGetPlatformInfo(cl_platform_id, cl_platform_info, size_t,
                             void *, size_t *)
    cl_int clGetDeviceIDs(cl_platform_id, cl_device_type, cl_uint,
                          cl_device_id *, cl_uint *)
    cl_int clGetDeviceInfo(cl_device_id, cl_device_info, size_t, void *,
                           size_t *)
    cl_int clCreateSubDevices(
        cl_device_id, const cl_device_partition_property *, cl_uint,
        cl_device_id *, cl_uint *)
    cl_int clRetainDevice(cl_device_id)
    cl_int clReleaseDevice(cl_device_id)
    cl_context clCreateContext(
        const cl_context_properties *, cl_uint, const cl_device_id *,
        void(*)(const char *, const void *, size_t, void *), void *, cl_int *)
    cl_context clCreateContextFromType(
        const cl_context_properties *, cl_device_type,
        void(*)(const char *, const void *, size_t, void *), void *, cl_int *)
    cl_int clRetainContext(cl_context)
    cl_int clReleaseContext(cl_context)
    cl_int clGetContextInfo(cl_context, cl_context_info, size_t, void *,
                            size_t *)
    cl_command_queue clCreateCommandQueueWithProperties(
        cl_context, cl_device_id, const cl_queue_properties *, cl_int *)
    cl_int clRetainCommandQueue(cl_command_queue)
    cl_int clReleaseCommandQueue(cl_command_queue)
    cl_int clGetCommandQueueInfo(cl_command_queue, cl_command_queue_info,
                                 size_t, void *, size_t *)
    cl_mem clCreateBuffer(cl_context, cl_mem_flags, size_t, void *, cl_int *)
    cl_mem clCreateSubBuffer(cl_mem, cl_mem_flags, cl_buffer_create_type,
                             const void *, cl_int *)
    cl_mem clCreateImage(cl_context, cl_mem_flags, const cl_image_format *,
                         const cl_image_desc *, void *, cl_int *)
    cl_mem clCreatePipe(cl_context, cl_mem_flags, cl_uint, cl_uint,
                        const cl_pipe_properties *, cl_int *)
    cl_int clRetainMemObject(cl_mem)
    cl_int clReleaseMemObject(cl_mem)
    cl_int clGetSupportedImageFormats(cl_context, cl_mem_flags,
                                      cl_mem_object_type, cl_uint,
                                      cl_image_format *, cl_uint *)
    cl_int clGetMemObjectInfo(cl_mem, cl_mem_info, size_t, void *, size_t *)
    cl_int clGetImageInfo(cl_mem, cl_image_info, size_t, void *, size_t *)
    cl_int clGetPipeInfo(cl_mem, cl_pipe_info, size_t, void *, size_t *)
    cl_int clSetMemObjectDestructorCallback(cl_mem, void(*)(cl_mem, void*),
                                            void *)
    void * clSVMAlloc(cl_context, cl_svm_mem_flags, size_t, cl_uint)
    void clSVMFree(cl_context, void *)
    cl_sampler clCreateSamplerWithProperties(cl_context,
                                             const cl_sampler_properties *,
                                             cl_int *)
    cl_int clRetainSampler(cl_sampler)
    cl_int clReleaseSampler(cl_sampler)
    cl_int clGetSamplerInfo(cl_sampler, cl_sampler_info, size_t, void *,
                            size_t *)
    cl_program clCreateProgramWithSource(cl_context, cl_uint, const char **,
                                         const size_t *, cl_int *)
    cl_program clCreateProgramWithBinary(cl_context, cl_uint,
                                         const cl_device_id *, const size_t *,
                                         const unsigned char **, cl_int *,
                                         cl_int *)
    cl_program clCreateProgramWithBuiltInKernels(cl_context, cl_uint,
                                                 const cl_device_id *,
                                                 const char *, cl_int *)
    cl_int clRetainProgram(cl_program)
    cl_int clReleaseProgram(cl_program)
    cl_int clBuildProgram(cl_program, cl_uint, const cl_device_id *,
                          const char *, void(*)(cl_program, void *), void *)
    cl_int clCompileProgram(cl_program, cl_uint, const cl_device_id *,
                            const char *, cl_uint, const cl_program *,
                            const char **, void(*)(cl_program, void *),
                            void *)
    cl_program clLinkProgram(cl_context, cl_uint, const cl_device_id *,
                             const char *, cl_uint, const cl_program *,
                             void(*)(cl_program, void *), void *, cl_int *)
    cl_int clUnloadPlatformCompiler(cl_platform_id)
    cl_int clGetProgramInfo(cl_program, cl_program_info, size_t, void *,
                            size_t *)
    cl_int clGetProgramBuildInfo(cl_program, cl_device_id,
                                 cl_program_build_info, size_t, void *,
                                 size_t *)
    cl_kernel clCreateKernel(cl_program, const char *, cl_int *)
    cl_int clCreateKernelsInProgram(cl_program, cl_uint, cl_kernel *,
                                    cl_uint *)
    cl_int clRetainKernel(cl_kernel)
    cl_int clReleaseKernel(cl_kernel)
    cl_int clSetKernelArg(cl_kernel, cl_uint, size_t, const void *)
    cl_int clSetKernelArgSVMPointer(cl_kernel, cl_uint, const void *)
    cl_int clSetKernelExecInfo(cl_kernel, cl_kernel_exec_info, size_t,
                               const void *)
    cl_int clGetKernelInfo(cl_kernel, cl_kernel_info, size_t, void *, size_t *)
    cl_int clGetKernelArgInfo(cl_kernel, cl_uint, cl_kernel_arg_info, size_t,
                              void *, size_t *)
    cl_int clGetKernelWorkGroupInfo(cl_kernel, cl_device_id,
                                    cl_kernel_work_group_info, size_t, void *,
                                    size_t *)
    cl_int clWaitForEvents(cl_uint, const cl_event *)
    cl_int clGetEventInfo(cl_event, cl_event_info, size_t, void *, size_t *)
    cl_event clCreateUserEvent(cl_context, cl_int *)
    cl_int clRetainEvent(cl_event)
    cl_int clReleaseEvent(cl_event)
    cl_int clSetUserEventStatus(cl_event, cl_int)
    cl_int clSetEventCallback(cl_event, cl_int,
                              void(*)(cl_event, cl_int, void *), void *)
    cl_int clGetEventProfilingInfo(cl_event, cl_profiling_info, size_t, void *,
                                   size_t *)
    cl_int clFlush(cl_command_queue)
    cl_int clFinish(cl_command_queue)
    cl_int clEnqueueReadBuffer(cl_command_queue, cl_mem, cl_bool, size_t,
                               size_t, void *, cl_uint, const cl_event *,
                               cl_event *)
    cl_int clEnqueueReadBufferRect(cl_command_queue, cl_mem, cl_bool,
                                   const size_t *, const size_t *,
                                   const size_t *, size_t, size_t,
                                   size_t, size_t, void *, cl_uint,
                                   const cl_event *, cl_event *)
    cl_int clEnqueueWriteBuffer(cl_command_queue, cl_mem, cl_bool, size_t,
                                size_t, const void *, cl_uint,
                                const cl_event *, cl_event *)
    cl_int clEnqueueWriteBufferRect(cl_command_queue, cl_mem, cl_bool,
                                    const size_t *, const size_t *,
                                    const size_t *, size_t, size_t, size_t,
                                    size_t, const void *, cl_uint,
                                    const cl_event *, cl_event *)
    cl_int clEnqueueFillBuffer(cl_command_queue, cl_mem, const void *, size_t,
                               size_t, size_t, cl_uint, const cl_event *,
                               cl_event *)
    cl_int clEnqueueCopyBuffer(cl_command_queue, cl_mem, cl_mem, size_t,
                               size_t, size_t, cl_uint, const cl_event *,
                               cl_event *)
    cl_int clEnqueueCopyBufferRect(cl_command_queue, cl_mem, cl_mem,
                                   const size_t *, const size_t *,
                                   const size_t *, size_t, size_t, size_t,
                                   size_t, cl_uint, const cl_event *,
                                   cl_event *)
    cl_int clEnqueueReadImage(cl_command_queue, cl_mem, cl_bool,
                              const size_t *, const size_t *, size_t, size_t,
                              void *, cl_uint, const cl_event *, cl_event *)
    cl_int clEnqueueWriteImage(cl_command_queue, cl_mem, cl_bool,
                               const size_t *, const size_t *, size_t, size_t,
                               const void *, cl_uint, const cl_event *,
                               cl_event *)
    cl_int clEnqueueFillImage(cl_command_queue, cl_mem, const void *,
                              const size_t *, const size_t *, cl_uint,
                              const cl_event *, cl_event *)
    cl_int clEnqueueCopyImage(cl_command_queue, cl_mem, cl_mem, const size_t *,
                              const size_t *, const size_t *, cl_uint,
                              const cl_event *, cl_event *)
    cl_int clEnqueueCopyImageToBuffer(cl_command_queue, cl_mem, cl_mem,
                                      const size_t *, const size_t *, size_t,
                                      cl_uint, const cl_event *, cl_event *)
    cl_int clEnqueueCopyBufferToImage(cl_command_queue, cl_mem, cl_mem, size_t,
                                      const size_t *, const size_t *, cl_uint,
                                      const cl_event *, cl_event *)
    void * clEnqueueMapBuffer(cl_command_queue, cl_mem, cl_bool, cl_map_flags,
                              size_t, size_t, cl_uint, const cl_event *,
                              cl_event *, cl_int *)
    void * clEnqueueMapImage(cl_command_queue, cl_mem, cl_bool, cl_map_flags,
                             const size_t *, const size_t *, size_t *,
                             size_t *, cl_uint, const cl_event *, cl_event *,
                             cl_int *)
    cl_int clEnqueueUnmapMemObject(cl_command_queue, cl_mem, void *, cl_uint,
                                   const cl_event *, cl_event *)
    cl_int clEnqueueMigrateMemObjects(cl_command_queue, cl_uint,
                                      const cl_mem *, cl_mem_migration_flags,
                                      cl_uint, const cl_event *, cl_event *)
    cl_int clEnqueueNDRangeKernel(
        cl_command_queue, cl_kernel, cl_uint, const size_t *, const size_t *,
        const size_t *, cl_uint, const cl_event *, cl_event *)
    cl_int clEnqueueNativeKernel(
        cl_command_queue, void(*)(void *), void *, size_t, cl_uint,
        const cl_mem *, const void **, cl_uint, const cl_event *, cl_event *)
    cl_int clEnqueueMarkerWithWaitList(cl_command_queue, cl_uint,
                                       const cl_event *, cl_event *)
    cl_int clEnqueueBarrierWithWaitList(cl_command_queue, cl_uint,
                                        const cl_event *, cl_event *)
    cl_int clEnqueueSVMFree(
        cl_command_queue, cl_uint, void *[],
        void(*)(cl_command_queue, cl_uint, void *[], void *),
        void *, cl_uint, const cl_event *, cl_event *)
    cl_int clEnqueueSVMMemcpy(cl_command_queue, cl_bool, void *, const void *,
                              size_t, cl_uint, const cl_event *, cl_event *)
    cl_int clEnqueueSVMMemFill(cl_command_queue, void *, const void *, size_t,
                               size_t, cl_uint, const cl_event *, cl_event *)
    cl_int clEnqueueSVMMap(cl_command_queue, cl_bool, cl_map_flags, void *,
                           size_t, cl_uint, const cl_event *, cl_event *)
    cl_int clEnqueueSVMUnmap(cl_command_queue, void *, cl_uint,
                             const cl_event *, cl_event *)
    void * clGetExtensionFunctionAddressForPlatform(cl_platform_id,
                                                    const char *)
    cl_mem clCreateImage2D(cl_context, cl_mem_flags, const cl_image_format *,
                           size_t, size_t, size_t, void *, cl_int *)
    cl_mem clCreateImage3D(cl_context, cl_mem_flags,
                           const cl_image_format *,
                           size_t, size_t, size_t, size_t, size_t,
                           void *, cl_int *)
    cl_int clEnqueueMarker(cl_command_queue, cl_event *)
    cl_int clEnqueueWaitForEvents(cl_command_queue, cl_uint, const cl_event *)
    cl_int clEnqueueBarrier(cl_command_queue)
    cl_int clUnloadCompiler()
    void * clGetExtensionFunctionAddress(const char *)
    cl_command_queue clCreateCommandQueue(cl_context, cl_device_id,
                                          cl_command_queue_properties,
                                          cl_int *)
    cl_sampler clCreateSampler(cl_context, cl_bool, cl_addressing_mode,
                               cl_filter_mode, cl_int *)
    cl_int clEnqueueTask(cl_command_queue, cl_kernel, cl_uint,
                         const cl_event *, cl_event *)

###############################################################################
# thin wrappers
cdef cl_uint GetPlatformIDs(size_t num_entries,
                            cl_platform_id* platforms) except *
cdef cl_uint GetDeviceIDs(cl_platform_id platform,
                          size_t device_type,
                          size_t num_entries,
                          cl_device_id* devices) except *
cdef cl_context CreateContext(
    cl_context_properties* properties,
    size_t num_devices,
    cl_device_id* devices,
    void* pfn_notify,
    void* user_data)
cdef cl_command_queue CreateCommandQueue(
    cl_context context,
    cl_device_id device,
    cl_command_queue_properties properties)
cdef cl_mem CreateBuffer(
    cl_context context,
    size_t flags,
    size_t size,
    void* host_ptr)
cdef cl_program CreateProgramWithSource(
    cl_context context,
    cl_uint count,
    char** strings,
    size_t* lengths)
cdef void BuildProgram(
    cl_program program,
    cl_uint num_devices,
    cl_device_id* device_list,
    char* options,
    void* pfn_notify,
    void* user_data) except *
cdef cl_kernel CreateKernel(cl_program program, char* kernel_name)
cdef void SetKernelArg(cl_kernel kernel,
                       arg_index,
                       arg_size,
                       void* arg_value) except *
cdef void EnqueueTask(
    cl_command_queue command_queue,
    cl_kernel kernel,
    cl_uint num_events_in_wait_list,
    cl_event* event_wait_list,
    cl_event* event) except *
cdef void EnqueueNDRangeKernel(
    cl_command_queue command_queue,
    cl_kernel kernel,
    size_t work_dim,
    size_t* global_work_offset,
    size_t* global_work_size,
    size_t* local_work_size,
    cl_uint num_events_in_wait_list,
    cl_event* event_wait_list,
    cl_event* event) except *
cdef void EnqueueReadBuffer(
    cl_command_queue command_queue,
    cl_mem buffer,
    blocking_read,
    size_t offset,
    size_t cb,
    void* host_ptr,
    cl_uint num_events_in_wait_list,
    cl_event* event_wait_list,
    cl_event* event) except *
cdef void EnqueueWriteBuffer(
    cl_command_queue command_queue,
    cl_mem buffer,
    blocking_write,
    size_t offset,
    size_t cb,
    void* host_ptr,
    cl_uint num_events_in_wait_list,
    cl_event* event_wait_list,
    cl_event* event) except *
cdef void EnqueueCopyBuffer(
    cl_command_queue command_queue,
    cl_mem src_buffer,
    cl_mem dst_buffer,
    size_t src_offset,
    size_t dst_offset,
    size_t cb,
    cl_uint num_events_in_wait_list,
    cl_event* event_wait_list,
    cl_event* event) except *
cdef void Flush(cl_command_queue command_queue) except *
cdef void Finish(cl_command_queue command_queue) except *
cdef void ReleaseKernel(cl_kernel kernel) except *
cdef void ReleaseProgram(cl_program program) except *
cdef void ReleaseMemObject(cl_mem memobj) except *
cdef void ReleaseCommandQueue(cl_command_queue command_queue) except *
cdef void ReleaseContext(cl_context context) except *
cdef void WaitForEvents(size_t num_events, cl_event* event_list) except *
