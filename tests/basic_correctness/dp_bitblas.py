"""Compare the outputs of a GPTQ model to a Marlin model.

Note: GPTQ and Marlin do not have bitwise correctness.
As a result, in this test, we just confirm that the top selected tokens of the
Marlin/GPTQ models are in the top 3 selections of each other.

Note: Marlin internally uses locks to synchronize the threads. This can
result in very slight nondeterminism for Marlin. As a result, we re-run the test
up to 3 times to see if we pass.

Run `pytest tests/models/test_marlin.py`.
"""

from ..conftest import VllmRunner
import os
from vllm.platforms import current_platform
print(current_platform)
# ckpt_path = "hxbgsyxh/llama-13b-4bit-g-1-bitblas"
ckpt_path = "hxbgsyxh/llama-13b-4bit-g-1"

with VllmRunner(
        ckpt_path,
        dtype="half",
        gpu_memory_utilization=0.5,
        # set enforce_eager = True to disable cuda graph
        enforce_eager=True,
        quantization="bitblas",
) as bitnet_model:
    bitbnet_outputs = bitnet_model.generate_greedy(["Hi, tell me about microsoft?"], max_tokens=128)
    print("bitnet inference output:")
    print(bitbnet_outputs[0][0])
    print(bitbnet_outputs[0][1])