# EVOLVE-BLOCK-START
"""
Float16 vector addition using in-place a.add_(b) to eliminate output allocation overhead.
Returns a (modified in-place), avoiding any tensor allocation on the hot path.
"""

import torch


def custom_kernel(data):
    a, b = data
    return a.add_(b)
# EVOLVE-BLOCK-END
