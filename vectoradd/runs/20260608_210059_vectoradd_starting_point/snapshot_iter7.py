# EVOLVE-BLOCK-START
"""
Float16 vector addition calling ATen in-place add directly via torch.ops.aten.add_.Tensor
to bypass Python attribute lookup and tensor method dispatch overhead.
"""

import torch

_aten_add_ = torch.ops.aten.add_.Tensor


def custom_kernel(data):
    a, b = data
    return _aten_add_(a, b)
# EVOLVE-BLOCK-END
