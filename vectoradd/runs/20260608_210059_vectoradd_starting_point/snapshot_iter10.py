# EVOLVE-BLOCK-START
"""
Float16 vector addition using torch.add with a pre-allocated output buffer (out=)
keyed by tensor shape to skip allocation on the hot path without mutating inputs.
"""

import torch

_out_buffers = {}


def custom_kernel(data):
    a, b = data
    shape = a.shape
    if shape not in _out_buffers:
        _out_buffers[shape] = torch.empty(shape, dtype=torch.float16, device=a.device)
    return torch.add(a, b, out=_out_buffers[shape])
# EVOLVE-BLOCK-END
