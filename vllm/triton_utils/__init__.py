# SPDX-License-Identifier: Apache-2.0

from vllm.triton_utils.decorator import (triton_autotune_decorator,
                                         triton_heuristics_decorator,
                                         triton_jit_decorator)
from vllm.triton_utils.importing import HAS_TRITON

__all__ = [
    "HAS_TRITON", "triton_jit_decorator", "triton_autotune_decorator",
    "triton_heuristics_decorator"
]

if HAS_TRITON:

    from vllm.triton_utils.custom_cache_manager import (
        maybe_set_triton_cache_manager)

    __all__ += ["maybe_set_triton_cache_manager"]
