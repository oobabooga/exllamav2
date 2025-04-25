from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from exllamav2 import ExLlamaV2, ExLlamaV2Tokenizer
from exllamav2.config import ExLlamaV2Config
from exllamav2.vlm.util import (
    convert_to_rgb,
    size_to_longest_edge_and_patch_size,
    normalize_image
)

def preprocess(
    config: ExLlamaV2Config,
    image: Image
) -> (torch.Tensor, tuple):

    patch_size = tuple(config.vision_patch_size[d] for d in ["height", "width"])
    size = tuple(config.vision_size[d] for d in ["height", "width"])

    resample = Image.Resampling(config.vision_resample)
    image_mean = tuple(config.vision_image_mean)
    image_std = tuple(config.vision_image_std)
    rescale_factor = config.vision_rescale_factor

    # Convert to RGB and resize as necessary

    image = convert_to_rgb(image)
    old_size = image.size
    new_size = size
    if old_size != new_size:
        image = image.resize(new_size, resample = resample)

    # Convert to numpy array and normalize

    image = np.array(image).astype(np.float32)
    image = image * rescale_factor
    image = normalize_image(image, image_mean, image_std)

    # Convert to tensor, shape (3, resized_height, resized_width)

    image = image.transpose(2, 0, 1)
    image = torch.from_numpy(image).half()
    return image, new_size, None, None

def postprocess(
    model: ExLlamaV2,
    tokenizer: ExLlamaV2Tokenizer,
    embeddings: torch.Tensor,
    features_y: int,
    features_x: int,
):
    """
    Insert <start_of_image> and <end_of_image> tokens in image feature embeddings
    """

    id_start = tokenizer.single_id("<start_of_image>")
    id_end = tokenizer.single_id("<end_of_image>")
    img_start = model.modules[0].forward(torch.tensor([id_start], dtype=torch.long)).to(embeddings.device).half()
    img_end = model.modules[0].forward(torch.tensor([id_end], dtype=torch.long)).to(embeddings.device).half()

    embeddings = torch.cat((img_start, embeddings, img_end), dim = 0)
    return embeddings, 1, 1


def position_embeddings(
    config: ExLlamaV2Config,
    height: int,
    width: int,
    max_width: int,
    rope_sin: torch.Tensor,
    rope_cos: torch.Tensor,
    thw_grid: tuple | None = None,
):
    assert thw_grid is None, \
        "Video not supported for Siglip"

    # Siglip uses learned embeddings
    return None, None


def pre_project(
    config: ExLlamaV2Config,
    vision_outputs: torch.Tensor
):
    bsz, _, seq_length = vision_outputs.shape
    patches_per_image = int(config.vision_size["width"] // config.vision_patch_size["width"])

    reshaped_vision_outputs = (
        vision_outputs.transpose(1, 2)
        .reshape(bsz, seq_length, patches_per_image, patches_per_image)
        .contiguous()
    )

    tokens_per_side = int(config.vision_mm_tokens_per_image ** 0.5)
    kernel_size = patches_per_image // tokens_per_side
    pooled_vision_outputs = (
        F.avg_pool2d(reshaped_vision_outputs, kernel_size = kernel_size, stride = kernel_size)
        .flatten(2)
        .transpose(1, 2)
        .contiguous()
    )

    return pooled_vision_outputs