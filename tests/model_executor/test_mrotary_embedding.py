from unittest.mock import Mock

import torch
import pytest

from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding

IMAGE = 101
VIDEO = 102
VISION_START = 103
VISION_END = 104

SPATIAL_MERGE_SIZE_2 = 2

def create_image(t: int, h: int, w: int, merge_size: int):
    return [VISION_START] + [IMAGE] * t * (h // merge_size) * (w // merge_size) + [VISION_END]

def create_video(t: int, h: int, w: int, merge_size: int):
    return [VISION_START] + [VIDEO] * t * (h // merge_size) * (w // merge_size) + [VISION_END]

vl_test_cases = [
    # text and image and video
    {
        "input": [
            {
                "type": "tokens",
                "tokens": [0, 1, 2, 3],
            },
            {
                "type": "image",
                "t": 1,
                "h": 4 * SPATIAL_MERGE_SIZE_2,
                "w": 6 * SPATIAL_MERGE_SIZE_2,
            },
            {
                "type": "video",
                "t": 4,
                "h": 8 * SPATIAL_MERGE_SIZE_2,
                "w": 12 * SPATIAL_MERGE_SIZE_2,
                "second_per_grid_t": 1.0,
            },
            {
                "type": "tokens",
                "tokens": [4, 5, 6, 7, 8],
            },
        ],
        "spatial_merge_size": SPATIAL_MERGE_SIZE_2,
        "tokens_per_second": 1.0,
    },
    # text and image
    {
        "input": [
            {
                "type": "tokens",
                "tokens": [0, 1, 2, 3],
            },
            {
                "type": "image",
                "t": 1,
                "h": 2 * SPATIAL_MERGE_SIZE_2,
                "w": 2 * SPATIAL_MERGE_SIZE_2,
            },
            {
                "type": "tokens",
                "tokens": [4, 5, 6, 7, 8],
            },
        ],
        "spatial_merge_size": SPATIAL_MERGE_SIZE_2,
        "tokens_per_second": 1.0,
    },
    # text and video
    {
        "input": [
            {
                "type": "tokens",
                "tokens": [0, 1, 2, 3],
            },
            {
                "type": "video",
                "t": 4,
                "h": 6 * SPATIAL_MERGE_SIZE_2,
                "w": 8 * SPATIAL_MERGE_SIZE_2,
                "second_per_grid_t": 1.0,
            },
            {
                "type": "tokens",
                "tokens": [4, 5, 6, 7, 8],
            },
        ],
        "spatial_merge_size": SPATIAL_MERGE_SIZE_2,
    },
    # text only
    {
        "input": [
            {
                "type": "tokens",
                "tokens": [0, 1, 2, 3],
            },
        ],
        "spatial_merge_size": SPATIAL_MERGE_SIZE_2,
    },
]

@pytest.mark.parametrize("test_case", vl_test_cases)
def test_vl_get_input_positions_and_delta_correctness(
    test_case,
):
    input = test_case["input"]
    spatial_merge_size = test_case["spatial_merge_size"]
    tokens_per_second = test_case.get("tokens_per_second", 1.0)

    hf_config = Mock()
    hf_config.image_token_id = IMAGE
    hf_config.video_token_id = VIDEO
    hf_config.vision_start_token_id = VISION_START
    hf_config.vision_end_token_id = VISION_END

    hf_config.vision_config = Mock()
    hf_config.vision_config.spatial_merge_size = spatial_merge_size
    hf_config.vision_config.tokens_per_second = tokens_per_second

    input_tokens = []
    image_grid_thw = []
    video_grid_thw = []
    second_per_grid_ts = []

    for item in input:
        if item["type"] == "tokens":
            input_tokens.extend(item["tokens"])
        elif item["type"] == "image":
            input_tokens.extend(create_image(item["t"], item["h"], item["w"], spatial_merge_size))
            image_grid_thw.append([item["t"], item["h"], item["w"]])
        elif item["type"] == "video":
            input_tokens.extend(create_video(item["t"], item["h"], item["w"], spatial_merge_size))
            video_grid_thw.append([item["t"], item["h"], item["w"]])
            second_per_grid_ts.append(item["second_per_grid_t"])
    
    input_positions_torch, mrope_position_delta_torch = MRotaryEmbedding.get_input_positions_and_delta(
        input_tokens=input_tokens,
        hf_config=hf_config,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        second_per_grid_ts=second_per_grid_ts,
        use_numba=False,
    )

    input_positions_numba, mrope_position_delta_numba = MRotaryEmbedding.get_input_positions_and_delta(
        input_tokens=input_tokens,
        hf_config=hf_config,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        second_per_grid_ts=second_per_grid_ts,
        use_numba=True,
    )
    assert input_positions_torch.dtype == input_positions_numba.dtype
    assert input_positions_torch.shape == input_positions_numba.shape

    assert torch.equal(input_positions_torch, input_positions_numba)
    assert mrope_position_delta_torch == mrope_position_delta_numba
