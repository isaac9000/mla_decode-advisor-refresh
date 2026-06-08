# EVOLVE-BLOCK-START
"""
Float16 vector addition using inline CUDA kernel with 128-bit (float4) vectorized
loads/stores — 8 float16 values per thread per transaction.
"""

import torch
from torch.utils.cpp_extension import load_inline

_cuda_src = r"""
#include <cuda_fp16.h>
#include <stdint.h>

__global__ void vecadd_fp16_vec8(
    const __half* __restrict__ a,
    const __half* __restrict__ b,
    __half* __restrict__ c,
    int64_t n_vec8)
{
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_vec8) return;

    // 128-bit load: 8 x float16
    const float4* a4 = reinterpret_cast<const float4*>(a);
    const float4* b4 = reinterpret_cast<const float4*>(b);
    float4* c4 = reinterpret_cast<float4*>(c);

    float4 av = __ldg(a4 + idx);
    float4 bv = __ldg(b4 + idx);

    __half2* ah = reinterpret_cast<__half2*>(&av);
    __half2* bh = reinterpret_cast<__half2*>(&bv);
    float4 cv;
    __half2* ch = reinterpret_cast<__half2*>(&cv);

    ch[0] = __hadd2(ah[0], bh[0]);
    ch[1] = __hadd2(ah[1], bh[1]);
    ch[2] = __hadd2(ah[2], bh[2]);
    ch[3] = __hadd2(ah[3], bh[3]);

    c4[idx] = cv;
}

torch::Tensor vecadd(torch::Tensor a, torch::Tensor b) {
    auto c = torch::empty_like(a);
    int64_t n = a.numel();
    // Process in chunks of 8 float16 = 1 float4
    int64_t n_vec8 = (n + 7) / 8;
    const int threads = 256;
    int blocks = (n_vec8 + threads - 1) / threads;
    vecadd_fp16_vec8<<<blocks, threads>>>(
        reinterpret_cast<const __half*>(a.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(b.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(c.data_ptr<at::Half>()),
        n_vec8
    );
    return c;
}
"""

_ext = load_inline(
    name="vecadd_fp16",
    cpp_sources="torch::Tensor vecadd(torch::Tensor a, torch::Tensor b);",
    cuda_sources=_cuda_src,
    functions=["vecadd"],
    verbose=False,
)


def custom_kernel(data):
    a, b = data
    return _ext.vecadd(a, b)
# EVOLVE-BLOCK-END
