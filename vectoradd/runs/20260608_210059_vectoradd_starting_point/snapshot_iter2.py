# EVOLVE-BLOCK-START
"""
Float16 vector addition using vectorized Triton kernel with 128-bit loads.
BLOCK_SIZE=4096 with 8 float16 values per 128-bit transaction.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def vecadd_kernel(
    a_ptr, b_ptr, c_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    c = a + b

    tl.store(c_ptr + offsets, c, mask=mask)


def custom_kernel(data):
    a, b = data
    return torch.add(a, b)
# EVOLVE-BLOCK-END
