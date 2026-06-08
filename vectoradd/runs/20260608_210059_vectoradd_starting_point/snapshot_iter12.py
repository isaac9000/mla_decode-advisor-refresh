# EVOLVE-BLOCK-START
"""
Float16 vector addition using torch.add(a, b, out=a) with torch.add pre-bound
at module level to avoid Python LOAD_ATTR on torch module every call.
"""

import torch

_add = torch.add


def custom_kernel(data):
    a, b = data
    return _add(a, b, out=a)
# EVOLVE-BLOCK-END
