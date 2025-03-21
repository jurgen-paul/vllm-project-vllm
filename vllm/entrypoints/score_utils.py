# SPDX-License-Identifier: Apache-2.0
from typing import Union

from torch.nn import CosineSimilarity

from vllm.outputs import PoolingRequestOutput
from vllm.transformers_utils.tokenizer import (PreTrainedTokenizer,
                                               PreTrainedTokenizerFast)


def _cosine_similarity(
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    embed_1: list[PoolingRequestOutput],
    embed_2: list[PoolingRequestOutput],
) -> list[PoolingRequestOutput]:

    scorer = CosineSimilarity(0)
    scores: Union[list[PoolingRequestOutput]] = []

    for emb_1, emb_2 in zip(embed_1, embed_2):
        pair_score = scorer(emb_1.outputs.data, emb_2.outputs.data)

        padding = []
        if (pad_token_id := getattr(tokenizer, "pad_token_id",
                                    None)) is not None:
            padding = [pad_token_id]

        tokens = emb_1.prompt_token_ids + padding + emb_2.prompt_token_ids

        scores.append(
            PoolingRequestOutput(
                request_id=f"{emb_1.request_id}_{emb_2.request_id}",
                outputs=pair_score,
                prompt_token_ids=tokens,
                finished=True))

    return scores


def _validate_score_input_lens(
    texts_1: Union[list[str], list[dict]],
    texts_2: Union[list[str], list[dict]],
):
    if len(texts_1) > 1 and len(texts_1) != len(texts_2):
        raise ValueError("Input lengths must be either 1:1, 1:N or N:N")
    if len(texts_1) == 0:
        raise ValueError("At least one text element must be given")
    if len(texts_2) == 0:
        raise ValueError("At least one text_pair element must be given")


def _validate_truncation_size(max_model_len: int,
                              truncate_prompt_tokens: int) -> int:
    if truncate_prompt_tokens is not None and truncate_prompt_tokens == -1:
        truncate_prompt_tokens = max_model_len
        return truncate_prompt_tokens

    if truncate_prompt_tokens is not None \
                            and truncate_prompt_tokens > max_model_len:
        raise ValueError(
            f"truncate_prompt_tokens value ({truncate_prompt_tokens}) "
            f"is greater than max_model_len ({max_model_len})."
            f" Please, select a smaller truncation size.")

    return truncate_prompt_tokens
