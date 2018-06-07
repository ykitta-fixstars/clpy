cdef extern from "CL/cl.h":
    cdef enum:
        CL_INVALID_ARG_VALUE
        CL_ADDRESS_CLAMP
        CL_DEVICE_PARENT_DEVICE
        CL_EVENT_REFERENCE_COUNT
        __OPENCL_CL_H
        CL_PROFILING_COMMAND_SUBMIT
        CL_FALSE
        CL_ARGB
        CL_QUEUE_DEVICE
        CL_DEVICE_EXTENSIONS
        CL_PROGRAM_NUM_DEVICES
        CL_DEVICE_AFFINITY_DOMAIN_NEXT_PARTITIONABLE
        CL_PROGRAM_BUILD_OPTIONS
        CL_MAP_WRITE_INVALIDATE_REGION
        CL_MEM_OBJECT_IMAGE1D_BUFFER
        CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG
        CL_COMMAND_COPY_IMAGE
        CL_RGx
        CL_IMAGE_NUM_SAMPLES
        CL_BGRA
        CL_A
        CL_KERNEL_ARG_ACCESS_NONE
        CL_IMAGE_DEPTH
        CL_IMAGE_FORMAT
        CL_UNSIGNED_INT32
        CL_SAMPLER_ADDRESSING_MODE
        CL_DEVICE_AFFINITY_DOMAIN_NUMA
        CL_DEVICE_SVM_COARSE_GRAIN_BUFFER
        CL_BUILD_NONE
        CL_COMMAND_SVM_MAP
        CL_QUEUED
        CL_INVALID_IMAGE_FORMAT_DESCRIPTOR
        CL_MEM_OBJECT_BUFFER
        CL_COMMAND_WRITE_BUFFER_RECT
        CL_INVALID_WORK_ITEM_SIZE
        CL_IMAGE_SLICE_PITCH
        CL_PROGRAM_DEVICES
        CL_DEVICE_MAX_PARAMETER_SIZE
        CL_MEM_SIZE
        CL_COMMAND_USER
        CL_UNORM_INT16
        CL_MEM_OBJECT_IMAGE2D_ARRAY
        CL_DEVICE_IMAGE_MAX_ARRAY_SIZE
        CL_DEVICE_IMAGE_SUPPORT
        CL_INVALID_BUILD_OPTIONS
        CL_COMMAND_FILL_IMAGE
        CL_KERNEL_ARG_ADDRESS_CONSTANT
        CL_HALF_FLOAT
        CL_IMAGE_WIDTH
        CL_COMPILE_PROGRAM_FAILURE
        CL_COMMAND_MARKER
        CL_RUNNING
        CL_sRGB
        CL_KERNEL_LOCAL_MEM_SIZE
        CL_INVALID_PLATFORM
        CL_INVALID_PROGRAM
        CL_FP_ROUND_TO_INF
        CL_DEVICE_MAX_CLOCK_FREQUENCY
        CL_SAMPLER_LOD_MAX
        CL_KERNEL_PRIVATE_MEM_SIZE
        CL_CONTEXT_PROPERTIES
        CL_MEM_HOST_PTR
        CL_PROGRAM_CONTEXT
        CL_DEVICE_PIPE_MAX_ACTIVE_RESERVATIONS
        CL_BUFFER_CREATE_TYPE_REGION
        CL_KERNEL_PROGRAM
        CL_sRGBx
        CL_EVENT_COMMAND_QUEUE
        CL_ADDRESS_REPEAT
        CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT
        CL_CONTEXT_NUM_DEVICES
        CL_INVALID_BUFFER_SIZE
        CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE
        CL_DEVICE_MAX_PIPE_ARGS
        CL_PLATFORM_VENDOR
        CL_DEVICE_NOT_FOUND
        CL_DEVICE_SVM_FINE_GRAIN_BUFFER
        CL_ADDRESS_MIRRORED_REPEAT
        CL_DEVICE_TYPE_DEFAULT
        CL_UNSIGNED_INT16
        CL_SAMPLER_MIP_FILTER_MODE
        CL_TRUE
        CL_DEVICE_IMAGE3D_MAX_WIDTH
        CL_DEVICE_BUILT_IN_KERNELS
        CL_DEVICE_REFERENCE_COUNT
        CL_MEM_HOST_NO_ACCESS
        CL_DEVICE_IMAGE2D_MAX_WIDTH
        CL_MEM_MAP_COUNT
        CL_PROGRAM_BUILD_LOG
        CL_EVENT_CONTEXT
        CL_MEM_COPY_OVERLAP
        CL_MEM_HOST_READ_ONLY
        CL_ADDRESS_CLAMP_TO_EDGE
        CL_DEVICE_SVM_ATOMICS
        CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT
        CL_VERSION_2_0
        CL_DEVICE_AFFINITY_DOMAIN_L4_CACHE
        CL_RG
        CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE
        CL_EXEC_NATIVE_KERNEL
        CL_BUILD_SUCCESS
        CL_PROGRAM_REFERENCE_COUNT
        CL_OUT_OF_HOST_MEMORY
        CL_INVALID_COMMAND_QUEUE
        CL_DEVICE_PROFILE
        CL_KERNEL_ARG_TYPE_QUALIFIER
        CL_BUILD_ERROR
        CL_LINK_PROGRAM_FAILURE
        CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE
        CL_DEVICE_AFFINITY_DOMAIN_L2_CACHE
        CL_FP_ROUND_TO_NEAREST
        CL_SIGNED_INT8
        CL_MEM_SVM_FINE_GRAIN_BUFFER
        CL_KERNEL_EXEC_INFO_SVM_FINE_GRAIN_SYSTEM
        CL_DEVICE_PREFERRED_PLATFORM_ATOMIC_ALIGNMENT
        CL_DEVICE_TYPE_CPU
        CL_IMAGE_BUFFER
        CL_COMMAND_MAP_BUFFER
        CL_DEVICE_GLOBAL_MEM_CACHE_SIZE
        CL_DEVICE_AFFINITY_DOMAIN_L1_CACHE
        CL_COMMAND_NATIVE_KERNEL
        CL_PROGRAM_BINARY_TYPE
        CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE
        CL_EVENT_COMMAND_EXECUTION_STATUS
        CL_DEVICE_MAX_CONSTANT_ARGS
        CL_DEVICE_PIPE_MAX_PACKET_SIZE
        CL_PROFILING_COMMAND_QUEUED
        CL_INVALID_OPERATION
        CL_PROFILING_COMMAND_START
        CL_OUT_OF_RESOURCES
        CL_PROGRAM_BINARY_TYPE_EXECUTABLE
        CL_COMMAND_SVM_FREE
        CL_INVALID_SAMPLER
        CL_INVALID_WORK_GROUP_SIZE
        CL_MEM_KERNEL_READ_AND_WRITE
        CL_SUCCESS
        CL_FILTER_LINEAR
        CL_KERNEL_EXEC_INFO_SVM_PTRS
        CL_DEVICE_LOCAL_MEM_TYPE
        CL_RGB
        CL_Rx
        CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS
        CL_MEM_OBJECT_IMAGE2D
        CL_QUEUE_REFERENCE_COUNT
        CL_DEVICE_SVM_FINE_GRAIN_SYSTEM
        CL_FP_SOFT_FLOAT
        CL_DEVICE_VENDOR
        CL_PROGRAM_BINARY_TYPE_NONE
        CL_COMMAND_WRITE_BUFFER
        CL_COMMAND_READ_IMAGE
        CL_DEVICE_TYPE
        CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT
        CL_INVALID_QUEUE_PROPERTIES
        CL_DRIVER_VERSION
        CL_INVALID_KERNEL
        CL_INVALID_HOST_PTR
        CL_DEVICE_AFFINITY_DOMAIN_L3_CACHE
        CL_MEM_REFERENCE_COUNT
        CL_PROGRAM_BINARY_TYPE_LIBRARY
        CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE
        CL_MEM_OFFSET
        CL_IMAGE_NUM_MIP_LEVELS
        CL_MAP_READ
        CL_IMAGE_ARRAY_SIZE
        CL_DEVICE_ADDRESS_BITS
        CL_INVALID_KERNEL_DEFINITION
        CL_COMMAND_ACQUIRE_GL_OBJECTS
        CL_KERNEL_ARG_ACCESS_READ_ONLY
        CL_DEVICE_MAX_SAMPLERS
        CL_INVALID_COMPILER_OPTIONS
        CL_DEVICE_NAME
        CL_KERNEL_WORK_GROUP_SIZE
        CL_PROGRAM_BINARY_SIZES
        CL_INVALID_WORK_DIMENSION
        CL_MIGRATE_MEM_OBJECT_HOST
        CL_SAMPLER_CONTEXT
        CL_INVALID_MEM_OBJECT
        CL_IMAGE_ROW_PITCH
        CL_ADDRESS_NONE
        CL_INVALID_MIP_LEVEL
        CL_DEVICE_OPENCL_C_VERSION
        CL_PIPE_MAX_PACKETS
        CL_UNORM_INT24
        CL_READ_ONLY_CACHE
        CL_KERNEL_ARG_ADDRESS_PRIVATE
        CL_DEVICE_PLATFORM
        CL_COMMAND_UNMAP_MEM_OBJECT
        CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG
        CL_UNSIGNED_INT8
        CL_DEVICE_COMPILER_AVAILABLE
        CL_MEM_READ_ONLY
        CL_COMMAND_READ_BUFFER
        CL_COMMAND_FILL_BUFFER
        CL_KERNEL_ARG_TYPE_PIPE
        CL_SIGNED_INT32
        CL_CONTEXT_PLATFORM
        CL_KERNEL_ARG_TYPE_RESTRICT
        CL_IMAGE_HEIGHT
        CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE
        CL_QUEUE_ON_DEVICE
        CL_CONTEXT_DEVICES
        CL_DEPTH_STENCIL
        CL_QUEUE_CONTEXT
        CL_DEVICE_MAX_MEM_ALLOC_SIZE
        CL_UNORM_INT_101010
        CL_MEM_OBJECT_IMAGE3D
        CL_MEM_CONTEXT
        CL_INVALID_KERNEL_ARGS
        CL_KERNEL_REFERENCE_COUNT
        CL_KERNEL_NUM_ARGS
        CL_INVALID_DEVICE_TYPE
        CL_DEVICE_SVM_CAPABILITIES
        CL_COMMAND_COPY_BUFFER_RECT
        CL_DEVICE_VENDOR_ID
        CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR
        CL_PROGRAM_BINARIES
        CL_INVALID_EVENT
        CL_INVALID_DEVICE
        CL_COMMAND_SVM_MEMCPY
        CL_MEM_SVM_ATOMICS
        CL_NON_BLOCKING
        CL_IMAGE_ELEMENT_SIZE
        CL_MEM_FLAGS
        CL_DEVICE_IMAGE3D_MAX_HEIGHT
        CL_KERNEL_ARG_ADDRESS_QUALIFIER
        CL_RGBA
        CL_R
        CL_UNORM_INT8
        CL_COMMAND_BARRIER
        CL_DEVICE_QUEUE_ON_DEVICE_PREFERRED_SIZE
        CL_PROFILING_COMMAND_COMPLETE
        CL_SAMPLER_LOD_MIN
        CL_DEVICE_GLOBAL_MEM_SIZE
        CL_READ_WRITE_CACHE
        CL_PROGRAM_NUM_KERNELS
        CL_VERSION_1_0
        CL_INVALID_ARG_SIZE
        CL_DEVICE_HOST_UNIFIED_MEMORY
        CL_DEVICE_TYPE_ALL
        CL_DEVICE_IMAGE2D_MAX_HEIGHT
        CL_NONE
        CL_COMPLETE
        CL_FP_INF_NAN
        CL_COMMAND_WRITE_IMAGE
        CL_DEVICE_PROFILING_TIMER_RESOLUTION
        CL_VERSION_1_2
        CL_DEVICE_MAX_ON_DEVICE_EVENTS
        CL_KERNEL_ARG_ACCESS_QUALIFIER
        CL_DEVICE_IMAGE3D_MAX_DEPTH
        CL_FP_DENORM
        CL_KERNEL_ARG_ACCESS_WRITE_ONLY
        CL_PLATFORM_VERSION
        CL_SAMPLER_NORMALIZED_COORDS
        CL_DEVICE_DOUBLE_FP_CONFIG
        CL_MEM_USE_HOST_PTR
        CL_INVALID_DEVICE_PARTITION_COUNT
        CL_DEVICE_SINGLE_FP_CONFIG
        CL_PROGRAM_BUILD_STATUS
        CL_INVALID_PIPE_SIZE
        CL_DEVICE_MAX_WRITE_IMAGE_ARGS
        CL_DEVICE_EXECUTION_CAPABILITIES
        CL_INTENSITY
        CL_INVALID_GLOBAL_WORK_SIZE
        CL_KERNEL_ARG_NAME
        CL_MEM_OBJECT_IMAGE1D
        CL_DEVICE_PRINTF_BUFFER_SIZE
        CL_GLOBAL
        CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE
        CL_PROFILING_INFO_NOT_AVAILABLE
        CL_sBGRA
        CL_DEVICE_PARTITION_MAX_SUB_DEVICES
        CL_MEM_READ_WRITE
        CL_PROGRAM_KERNEL_NAMES
        CL_COMMAND_COPY_BUFFER
        CL_DEVICE_MAX_READ_WRITE_IMAGE_ARGS
        CL_CONTEXT_REFERENCE_COUNT
        CL_COMMAND_READ_BUFFER_RECT
        CL_PROGRAM_SOURCE
        CL_DEVICE_VERSION
        CL_DEVICE_NATIVE_VECTOR_WIDTH_INT
        CL_KERNEL_ARG_TYPE_NONE
        CL_INVALID_KERNEL_NAME
        CL_EXEC_KERNEL
        CL_COMMAND_MAP_IMAGE
        CL_MEM_TYPE
        CL_QUEUE_PROFILING_ENABLE
        CL_MAP_WRITE
        CL_FILTER_NEAREST
        CL_KERNEL_ATTRIBUTES
        CL_PLATFORM_NAME
        CL_DEVICE_PARTITION_TYPE
        CL_COMMAND_NDRANGE_KERNEL
        CL_MEM_ASSOCIATED_MEMOBJECT
        CL_LOCAL
        CL_INVALID_BINARY
        CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF
        CL_UNORM_SHORT_565
        CL_MEM_HOST_WRITE_ONLY
        CL_DEVICE_QUEUE_ON_DEVICE_PROPERTIES
        CL_sRGBA
        CL_KERNEL_CONTEXT
        CL_DEVICE_MEM_BASE_ADDR_ALIGN
        CL_INVALID_PROPERTY
        CL_MEM_USES_SVM_POINTER
        CL_EVENT_COMMAND_TYPE
        CL_COMMAND_COPY_BUFFER_TO_IMAGE
        CL_CONTEXT_INTEROP_USER_SYNC
        CL_DEVICE_NOT_AVAILABLE
        CL_DEVICE_PARTITION_PROPERTIES
        CL_SAMPLER_FILTER_MODE
        CL_DEVICE_PARTITION_EQUALLY
        CL_RA
        CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT
        CL_KERNEL_ARG_ACCESS_READ_WRITE
        CL_PLATFORM_PROFILE
        CL_KERNEL_ARG_TYPE_NAME
        CL_DEVICE_QUEUE_ON_DEVICE_MAX_SIZE
        CL_COMMAND_TASK
        CL_IMAGE_FORMAT_MISMATCH
        CL_MEM_OBJECT_ALLOCATION_FAILURE
        CL_DEVICE_MAX_GLOBAL_VARIABLE_SIZE
        CL_INVALID_PROGRAM_EXECUTABLE
        CL_DEVICE_PARTITION_BY_AFFINITY_DOMAIN
        CL_LINKER_NOT_AVAILABLE
        CL_SUBMITTED
        CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT
        CL_COMMAND_RELEASE_GL_OBJECTS
        CL_MAP_FAILURE
        CL_DEVICE_MAX_WORK_ITEM_SIZES
        CL_DEVICE_PARTITION_FAILED
        CL_DEVICE_MAX_READ_IMAGE_ARGS
        CL_INVALID_GL_OBJECT
        CL_SIGNED_INT16
        CL_DEVICE_PREFERRED_LOCAL_ATOMIC_ALIGNMENT
        CL_COMMAND_COPY_IMAGE_TO_BUFFER
        CL_QUEUE_PROPERTIES
        CL_INVALID_GLOBAL_OFFSET
        CL_QUEUE_SIZE
        CL_SAMPLER_REFERENCE_COUNT
        CL_DEVICE_PARTITION_BY_COUNTS
        CL_PIPE_PACKET_SIZE
        CL_COMPILER_NOT_AVAILABLE
        CL_DEVICE_PARTITION_BY_COUNTS_LIST_END
        CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT
        CL_INVALID_IMAGE_DESCRIPTOR
        CL_DEVICE_QUEUE_PROPERTIES
        CL_KERNEL_FUNCTION_NAME
        CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST
        CL_UNORM_SHORT_555
        CL_MISALIGNED_SUB_BUFFER_OFFSET
        CL_COMMAND_SVM_UNMAP
        CL_BUILD_PROGRAM_FAILURE
        CL_PLATFORM_EXTENSIONS
        CL_PROFILING_COMMAND_END
        CL_INVALID_CONTEXT
        CL_DEVICE_LINKER_AVAILABLE
        CL_FP_ROUND_TO_ZERO
        CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR
        CL_ABGR
        CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF
        CL_DEVICE_AVAILABLE
        CL_IMAGE_FORMAT_NOT_SUPPORTED
        CL_COMMAND_MIGRATE_MEM_OBJECTS
        CL_QUEUE_ON_DEVICE_DEFAULT
        CL_DEVICE_LOCAL_MEM_SIZE
        CL_DEVICE_QUEUE_ON_HOST_PROPERTIES
        CL_KERNEL_ARG_TYPE_VOLATILE
        CL_DEVICE_ENDIAN_LITTLE
        CL_SNORM_INT16
        CL_KERNEL_GLOBAL_WORK_SIZE
        CL_DEVICE_IMAGE_PITCH_ALIGNMENT
        CL_DEVICE_PREFERRED_INTEROP_USER_SYNC
        CL_INVALID_ARG_INDEX
        CL_DEPTH
        CL_DEVICE_IMAGE_BASE_ADDRESS_ALIGNMENT
        CL_DEVICE_TYPE_ACCELERATOR
        CL_INVALID_IMAGE_SIZE
        CL_FP_FMA
        CL_MEM_OBJECT_PIPE
        CL_MEM_COPY_HOST_PTR
        CL_INVALID_EVENT_WAIT_LIST
        CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED
        CL_DEVICE_PARTITION_AFFINITY_DOMAIN
        CL_PROGRAM_BINARY_TYPE_COMPILED_OBJECT
        CL_RGBx
        CL_FLOAT
        CL_BLOCKING
        CL_DEVICE_IMAGE_MAX_BUFFER_SIZE
        CL_INVALID_LINKER_OPTIONS
        CL_DEVICE_MAX_COMPUTE_UNITS
        CL_LUMINANCE
        CL_MEM_OBJECT_IMAGE1D_ARRAY
        CL_KERNEL_COMPILE_WORK_GROUP_SIZE
        CL_KERNEL_ARG_ADDRESS_LOCAL
        CL_PROGRAM_BUILD_GLOBAL_VARIABLE_TOTAL_SIZE
        CL_DEVICE_TYPE_CUSTOM
        CL_DEVICE_ERROR_CORRECTION_SUPPORT
        CL_KERNEL_ARG_TYPE_CONST
        CL_DEVICE_MAX_WORK_GROUP_SIZE
        CL_DEVICE_TYPE_GPU
        CL_DEVICE_PREFERRED_GLOBAL_ATOMIC_ALIGNMENT
        CL_KERNEL_ARG_INFO_NOT_AVAILABLE
        CL_DEVICE_MAX_ON_DEVICE_QUEUES
        CL_VERSION_1_1
        CL_KERNEL_ARG_ADDRESS_GLOBAL
        CL_INVALID_VALUE
        CL_INVALID_DEVICE_QUEUE
        CL_COMMAND_SVM_MEMFILL
        CL_DEVICE_GLOBAL_MEM_CACHE_TYPE
        CL_BUILD_IN_PROGRESS
        CL_MEM_ALLOC_HOST_PTR
        CL_DEVICE_GLOBAL_VARIABLE_PREFERRED_TOTAL_SIZE
        CL_SNORM_INT8
        CL_MEM_WRITE_ONLY


cdef extern from "CL/cl.h":
    ctypedef signed char int8_t
    ctypedef short int int16_t
    ctypedef int int32_t
    ctypedef long int int64_t
    ctypedef unsigned char uint8_t
    ctypedef unsigned short int uint16_t
    ctypedef unsigned int uint32_t
    ctypedef unsigned long int uint64_t
    ctypedef signed char int_least8_t
    ctypedef short int int_least16_t
    ctypedef int int_least32_t
    ctypedef long int int_least64_t
    ctypedef unsigned char uint_least8_t
    ctypedef unsigned short int uint_least16_t
    ctypedef unsigned int uint_least32_t
    ctypedef unsigned long int uint_least64_t
    ctypedef signed char int_fast8_t
    ctypedef long int int_fast16_t
    ctypedef long int int_fast32_t
    ctypedef long int int_fast64_t
    ctypedef unsigned char uint_fast8_t
    ctypedef unsigned long int uint_fast16_t
    ctypedef unsigned long int uint_fast32_t
    ctypedef unsigned long int uint_fast64_t
    ctypedef long int intptr_t
    ctypedef unsigned long int uintptr_t
    ctypedef long int intmax_t
    ctypedef unsigned long int uintmax_t
    ctypedef int8_t cl_char
    ctypedef uint8_t cl_uchar
    ctypedef int16_t cl_short
    ctypedef uint16_t cl_ushort
    ctypedef int32_t cl_int
    ctypedef uint32_t cl_uint
    ctypedef int64_t cl_long
    ctypedef uint64_t cl_ulong
    ctypedef uint16_t cl_half
    ctypedef float cl_float
    ctypedef double cl_double
    ctypedef long int ptrdiff_t
    ctypedef long unsigned int size_t
    ctypedef int wchar_t
    ctypedef struct max_align_t:
        long long __max_align_ll
        long double __max_align_ld
    ctypedef unsigned int cl_GLuint
    ctypedef int cl_GLint
    ctypedef unsigned int cl_GLenum
    ctypedef int __m64
    ctypedef int __v2si
    ctypedef short __v4hi
    ctypedef char __v8qi
    ctypedef long long __v1di
    ctypedef float __v2sf
    ctypedef unsigned char __u_char
    ctypedef unsigned short int __u_short
    ctypedef unsigned int __u_int
    ctypedef unsigned long int __u_long
    ctypedef signed char __int8_t
    ctypedef unsigned char __uint8_t
    ctypedef signed short int __int16_t
    ctypedef unsigned short int __uint16_t
    ctypedef signed int __int32_t
    ctypedef unsigned int __uint32_t
    ctypedef signed long int __int64_t
    ctypedef unsigned long int __uint64_t
    ctypedef long int __quad_t
    ctypedef unsigned long int __u_quad_t
    ctypedef unsigned long int __dev_t
    ctypedef unsigned int __uid_t
    ctypedef unsigned int __gid_t
    ctypedef unsigned long int __ino_t
    ctypedef unsigned long int __ino64_t
    ctypedef unsigned int __mode_t
    ctypedef unsigned long int __nlink_t
    ctypedef long int __off_t
    ctypedef long int __off64_t
    ctypedef int __pid_t
    ctypedef struct __fsid_t:
        int __val[2]
    ctypedef long int __clock_t
    ctypedef unsigned long int __rlim_t
    ctypedef unsigned long int __rlim64_t
    ctypedef unsigned int __id_t
    ctypedef long int __time_t
    ctypedef unsigned int __useconds_t
    ctypedef long int __suseconds_t
    ctypedef int __daddr_t
    ctypedef int __key_t
    ctypedef int __clockid_t
    ctypedef long int __blksize_t
    ctypedef long int __blkcnt_t
    ctypedef long int __blkcnt64_t
    ctypedef unsigned long int __fsblkcnt_t
    ctypedef unsigned long int __fsblkcnt64_t
    ctypedef unsigned long int __fsfilcnt_t
    ctypedef unsigned long int __fsfilcnt64_t
    ctypedef long int __fsword_t
    ctypedef long int __ssize_t
    ctypedef long int __syscall_slong_t
    ctypedef unsigned long int __syscall_ulong_t
    ctypedef __off64_t __loff_t
    ctypedef long int __intptr_t
    ctypedef unsigned int __socklen_t
    ctypedef struct div_t:
        int quot
        int rem
    ctypedef struct ldiv_t:
        long int quot
        long int rem
    ctypedef struct lldiv_t:
        long long int quot
        long long int rem
    ctypedef __u_char u_char
    ctypedef __u_short u_short
    ctypedef __u_int u_int
    ctypedef __u_long u_long
    ctypedef __quad_t quad_t
    ctypedef __u_quad_t u_quad_t
    ctypedef __fsid_t fsid_t
    ctypedef __loff_t loff_t
    ctypedef __ino_t ino_t
    ctypedef __dev_t dev_t
    ctypedef __gid_t gid_t
    ctypedef __mode_t mode_t
    ctypedef __nlink_t nlink_t
    ctypedef __uid_t uid_t
    ctypedef __off_t off_t
    ctypedef __pid_t pid_t
    ctypedef __id_t id_t
    ctypedef __ssize_t ssize_t
    ctypedef __daddr_t daddr_t
    ctypedef __key_t key_t
    ctypedef __clock_t clock_t
    ctypedef __time_t time_t
    ctypedef __clockid_t clockid_t
    ctypedef unsigned long int ulong
    ctypedef unsigned short int ushort
    ctypedef unsigned int uint
    ctypedef unsigned int u_int8_t
    ctypedef unsigned int u_int16_t
    ctypedef unsigned int u_int32_t
    ctypedef unsigned int u_int64_t
    ctypedef int register_t
    ctypedef int __sig_atomic_t
    ctypedef __suseconds_t suseconds_t
    ctypedef long int __fd_mask
    ctypedef __fd_mask fd_mask
    ctypedef __blksize_t blksize_t
    ctypedef __blkcnt_t blkcnt_t
    ctypedef __fsblkcnt_t fsblkcnt_t
    ctypedef __fsfilcnt_t fsfilcnt_t
    ctypedef unsigned long int pthread_t
    ctypedef unsigned int pthread_key_t
    ctypedef int pthread_once_t
    ctypedef int pthread_spinlock_t
    ctypedef float __m128
    ctypedef float __v4sf
    ctypedef double __v2df
    ctypedef long long __v2di
    ctypedef unsigned long long __v2du
    ctypedef int __v4si
    ctypedef unsigned int __v4su
    ctypedef short __v8hi
    ctypedef unsigned short __v8hu
    ctypedef char __v16qi
    ctypedef unsigned char __v16qu
    ctypedef long long __m128i
    ctypedef double __m128d
    ctypedef float __cl_float4
    ctypedef cl_uchar __cl_uchar16
    ctypedef cl_char __cl_char16
    ctypedef cl_ushort __cl_ushort8
    ctypedef cl_short __cl_short8
    ctypedef cl_uint __cl_uint4
    ctypedef cl_int __cl_int4
    ctypedef cl_ulong __cl_ulong2
    ctypedef cl_long __cl_long2
    ctypedef cl_double __cl_double2
    ctypedef cl_uchar __cl_uchar8
    ctypedef cl_char __cl_char8
    ctypedef cl_ushort __cl_ushort4
    ctypedef cl_short __cl_short4
    ctypedef cl_uint __cl_uint2
    ctypedef cl_int __cl_int2
    ctypedef cl_ulong __cl_ulong1
    ctypedef cl_long __cl_long1
    ctypedef cl_float __cl_float2
    ctypedef cl_uint cl_bool
    ctypedef cl_ulong cl_bitfield
    ctypedef cl_bitfield cl_device_type
    ctypedef cl_uint cl_platform_info
    ctypedef cl_uint cl_device_info
    ctypedef cl_bitfield cl_device_fp_config
    ctypedef cl_uint cl_device_mem_cache_type
    ctypedef cl_uint cl_device_local_mem_type
    ctypedef cl_bitfield cl_device_exec_capabilities
    ctypedef cl_bitfield cl_device_svm_capabilities
    ctypedef cl_bitfield cl_command_queue_properties
    ctypedef intptr_t cl_device_partition_property
    ctypedef cl_bitfield cl_device_affinity_domain
    ctypedef intptr_t cl_context_properties
    ctypedef cl_uint cl_context_info
    ctypedef cl_bitfield cl_queue_properties
    ctypedef cl_uint cl_command_queue_info
    ctypedef cl_uint cl_channel_order
    ctypedef cl_uint cl_channel_type
    ctypedef cl_bitfield cl_mem_flags
    ctypedef cl_bitfield cl_svm_mem_flags
    ctypedef cl_uint cl_mem_object_type
    ctypedef cl_uint cl_mem_info
    ctypedef cl_bitfield cl_mem_migration_flags
    ctypedef cl_uint cl_image_info
    ctypedef cl_uint cl_buffer_create_type
    ctypedef cl_uint cl_addressing_mode
    ctypedef cl_uint cl_filter_mode
    ctypedef cl_uint cl_sampler_info
    ctypedef cl_bitfield cl_map_flags
    ctypedef intptr_t cl_pipe_properties
    ctypedef cl_uint cl_pipe_info
    ctypedef cl_uint cl_program_info
    ctypedef cl_uint cl_program_build_info
    ctypedef cl_uint cl_program_binary_type
    ctypedef cl_int cl_build_status
    ctypedef cl_uint cl_kernel_info
    ctypedef cl_uint cl_kernel_arg_info
    ctypedef cl_uint cl_kernel_arg_address_qualifier
    ctypedef cl_uint cl_kernel_arg_access_qualifier
    ctypedef cl_bitfield cl_kernel_arg_type_qualifier
    ctypedef cl_uint cl_kernel_work_group_info
    ctypedef cl_uint cl_event_info
    ctypedef cl_uint cl_command_type
    ctypedef cl_uint cl_profiling_info
    ctypedef cl_bitfield cl_sampler_properties
    ctypedef cl_uint cl_kernel_exec_info
    ctypedef struct cl_image_format:
        cl_channel_order image_channel_order
        cl_channel_type image_channel_data_type
    ctypedef struct cl_image_desc:
        cl_mem_object_type image_type
        size_t image_width
        size_t image_height
        size_t image_depth
        size_t image_array_size
        size_t image_row_pitch
        size_t image_slice_pitch
        cl_uint num_mip_levels
        cl_uint num_samples
    ctypedef struct cl_buffer_region:
        size_t origin
        size_t size
    cdef struct _cl_platform_id:
        pass
    cdef struct _cl_device_id:
        pass
    cdef struct _cl_context:
        pass
    cdef struct _cl_command_queue:
        pass
    cdef struct _cl_mem:
        pass
    cdef struct _cl_sampler:
        pass
    cdef struct _cl_program:
        pass
    cdef struct _cl_kernel:
        pass
    cdef struct _cl_event:
        pass
    ctypedef _cl_platform_id *cl_platform_id
    ctypedef _cl_device_id *cl_device_id
    ctypedef _cl_context *cl_context
    ctypedef _cl_command_queue *cl_command_queue
    ctypedef _cl_mem *cl_mem
    ctypedef _cl_sampler *cl_sampler
    ctypedef _cl_program *cl_program
    ctypedef _cl_kernel *cl_kernel
    ctypedef _cl_event *cl_event
