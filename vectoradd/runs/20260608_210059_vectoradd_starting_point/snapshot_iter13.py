# EVOLVE-BLOCK-START
"""
Float16 vector addition using torch._foreach_add_ — the foreach family uses
a single fused kernel launch for elementwise ops with lower dispatch overhead.
"""

import torch


def custom_kernel(data):
    a, b = data
    torch._foreach_add_([a], [b])
    return a
# EVOLVE-BLOCK-END
