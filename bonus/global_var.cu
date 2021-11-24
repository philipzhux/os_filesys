#include <inttypes.h>

typedef uint32_t u32;
__device__ __managed__ u32 gtime = 0;
__device__ __managed__ u32 curr_dir_fd = 0;