# SPDX-License-Identifier: Apache-2.0

import os

VLLM_MODEL_OVERWRITE_PATH = ".model.overwrite"

os.environ["VLLM_MODEL_OVERWRITE_PATH"] = VLLM_MODEL_OVERWRITE_PATH

overwrite = """
Qwen/Qwen2.5-0.5B-Instruct\t/share/cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775/
deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B\t/share/cache/modelscope/hub/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
"""

with open(VLLM_MODEL_OVERWRITE_PATH, "w") as f:
    f.write(overwrite)


def test(model, use_v1):
    if use_v1:
        os.environ["VLLM_USE_V1"] = "1"
    else:
        os.environ["VLLM_USE_V1"] = "0"

    from vllm import LLM, SamplingParams

    # Sample prompts.
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    # Create an LLM.
    llm = LLM(model=model)
    # Generate texts from the prompts. The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

    del llm


def process_warp(fn, /, *args, **kwargs):
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(1, mp.get_context("spawn")) as executor:
        f = executor.submit(fn, *args, **kwargs)
        return f.result()


if __name__ == '__main__':
    process_warp(test, "Qwen/Qwen2.5-0.5B-Instruct", use_v1=False)
    process_warp(test,
                 "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                 use_v1=False)
    process_warp(test, "Qwen/Qwen2.5-0.5B-Instruct", use_v1=True)
    process_warp(test,
                 "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                 use_v1=True)
