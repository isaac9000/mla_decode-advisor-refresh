# EVOLVE-BLOCK-START
"""
Float16 vector addition using torch.add(a, b, out=a) — writes result directly
into a's storage via the out= parameter, potentially taking a different internal
code path than a.add_(b) while avoiding separate output allocation.
"""

import torch


def custom_kernel(data):
    a, b = data
    return torch.add(a, b, out=a)
# EVOLVE-BLOCK-END
