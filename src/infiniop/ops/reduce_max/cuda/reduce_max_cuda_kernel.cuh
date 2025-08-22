#ifndef __REDUCE_MAX_CUDA_KERNEL_CUH__
#define __REDUCE_MAX_CUDA_KERNEL_CUH__

#include "reduce_max_cuda.h"

namespace op::reduce_max::cuda {

// 通用 Reduce-Max Kernel (单次 block 规约)
template <typename T, typename Op>
__global__ void reduce_max_kernel(const T* __restrict__ input,
                                  T* __restrict__ output,
                                  size_t N,
                                  Op op) {
    extern __shared__ T sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // 读数据到共享内存
    T val = (idx < N) ? input[idx] : std::numeric_limits<T>::lowest();
    sdata[tid] = val;
    __syncthreads();

    // 规约：共享内存内逐步折半
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] = op(sdata[tid], sdata[tid + stride]);
        }
        __syncthreads();
    }

    // 每个 block 的结果写到 output
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// 启动函数：递归调用直到只剩下 1 个 block
template <typename T>
T launch_reduce_max(const T* d_input, size_t N, cudaStream_t stream = 0) {
    constexpr int threads = 256;
    int blocks = (N + threads - 1) / threads;

    T* d_intermediate;
    cudaMalloc(&d_intermediate, blocks * sizeof(T));

    ReduceMaxOp op{};
    
    reduce_max_kernel<<<blocks, threads, threads * sizeof(T), stream>>>(d_input, d_intermediate, N, op);

   
    int s = blocks;
    while (s > 1) {
        int threads2 = (s > 256) ? 256 : s;
        int blocks2 = (s + threads2 - 1) / threads2;
        reduce_max_kernel<<<blocks2, threads2, threads2 * sizeof(T), stream>>>(d_intermediate, d_intermediate, s, op);
        s = blocks2;
    }

    // 拷贝结果
    T h_out;
    cudaMemcpy(&h_out, d_intermediate, sizeof(T), cudaMemcpyDeviceToHost);
    cudaFree(d_intermediate);
    return h_out;
}

} // namespace op::reduce_max::cuda

#endif // __REDUCE_MAX_CUDA_KERNEL_CUH__
