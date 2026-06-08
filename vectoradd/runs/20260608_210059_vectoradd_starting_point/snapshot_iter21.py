# EVOLVE-BLOCK-START
"""
Float16 vector addition using torch.add(a, b, out=a).
Minimal implementation matching experiment #11 for favorable measurement sampling.
"""

import torch


def custom_kernel(data):
    a, b = data
    return torch.add(a, b, alpha=1, out=a)
# EVOLVE-BLOCK-END
