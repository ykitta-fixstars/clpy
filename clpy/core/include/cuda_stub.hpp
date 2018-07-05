#pragma once

#define __global__ __attribute((annotate("cu_global")))
#define __device__ __attribute((annotate("cu_device")))
#define __shared__ __attribute((annotate("cu_shared")))
