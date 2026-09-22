import logging
import math
import numbers
from contextlib import nullcontext

import torch

from comfy import model_management
from comfy.text_encoders.minimax import process_video_block

from .encoder_helpers import (
    fuse_deepstack_layers,
    fuse_visual_token_sources,
    prepare_vlm_image,
)


QWEN_IMAGE_PAD_ID = 151655
QWEN_VIDEO_PAD_ID = 151656
QWEN_VIDEO_TOKEN = "<|vision_start|><|video_pad|><|vision_end|>"


def _positive_fps(value, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, numbers.Real)
        or not math.isfinite(float(value))
        or value <= 0
    ):
        raise ValueError(f"Qwen3-VL {name} must be a finite value greater than zero.")
    return float(value)


def prepare_qwen3vl_video_frames(
    video,
    resolution,
    video_fps,
    sample_fps,
) -> tuple[torch.Tensor, list[int]]:
    sources = list(video) if isinstance(video, (list, tuple)) else [video]
    source_frames = []
    for source in sources:
        if not torch.is_tensor(source) or source.ndim != 4 or source.shape[-1] != 3:
            raise ValueError("Qwen3-VL video requires BHWC RGB IMAGE tensors.")
        source_frames.extend(source[index:index + 1] for index in range(source.shape[0]))
    frame_count = len(source_frames)
    if frame_count < 1:
        raise ValueError("Qwen3-VL video requires at least one source frame.")
    source_fps = _positive_fps(video_fps, "video_fps")
    target_fps = min(_positive_fps(sample_fps, "sample_fps"), source_fps)
    indices = []
    sample_index = 0
    while True:
        source_index = int(math.floor(sample_index * source_fps / target_fps + 0.5))
        if source_index >= frame_count:
            break
        if not indices or source_index != indices[-1]:
            indices.append(source_index)
        sample_index += 1
    prepared = [prepare_vlm_image(source_frames[index], resolution) for index in indices]
    if len({tuple(frame.shape[1:3]) for frame in prepared}) != 1:
        raise ValueError("Qwen3-VL video frames must resolve to one common spatial size.")
    frames = torch.cat(prepared, dim=0)
    motion_delta = (
        float((frames[1:] - frames[:-1]).abs().mean()) if frames.shape[0] > 1 else 0.0
    )
    logging.info(
        "UC_TextGenerate Qwen3-VL video: source_frames=%d sampled_frames=%d temporal_blocks=%d source_fps=%.4f sample_fps=%.4f first_index=%d last_index=%d motion_delta=%.8f",
        frame_count,
        len(indices),
        (len(indices) + 1) // 2,
        source_fps,
        target_fps,
        indices[0],
        indices[-1],
        motion_delta,
    )
    if len(indices) % 2 == 1:
        frames = torch.cat([frames, frames[-1:]], dim=0)
        indices.append(indices[-1])
    return frames, indices


def qwen3vl_video_prompt(indices: list[int], video_fps) -> str:
    source_fps = _positive_fps(video_fps, "video_fps")
    if not indices or len(indices) % 2 != 0:
        raise ValueError("Qwen3-VL video sampling must produce paired frame indices.")
    return "Video 1: " + "".join(
        f"<{((indices[index] + indices[index + 1]) / 2) / source_fps:.1f} seconds>{QWEN_VIDEO_TOKEN}"
        for index in range(0, len(indices), 2)
    )


def _qwen_clip_model(clip):
    stage = getattr(clip, "cond_stage_model", None)
    if stage is None:
        return None
    for name in (getattr(stage, "clip", None), getattr(stage, "clip_name", None)):
        candidate = getattr(stage, name, None) if isinstance(name, str) else None
        if candidate is not None and hasattr(candidate, "process_tokens") and hasattr(
            candidate, "transformer"
        ):
            return candidate
    if hasattr(stage, "process_tokens") and hasattr(stage, "transformer"):
        return stage
    return None


def _token_rows(tokens):
    if isinstance(tokens, dict):
        tokens = next(iter(tokens.values()))
    return [[item[0] for item in batch] for batch in tokens]


def _visual_grid_shape(block):
    grid = block["extra"].get("grid")
    if grid is None:
        raise ValueError("Qwen3-VL visual source is missing its exact grid metadata.")
    values = grid.reshape(-1).tolist() if torch.is_tensor(grid) else list(grid)
    if len(values) < 3:
        raise ValueError(f"Qwen3-VL visual source received a malformed grid: {grid!r}.")
    shape = (int(values[-2]) // 2, int(values[-1]) // 2)
    size = block["embedding"].reshape(-1, block["embedding"].shape[-1]).shape[0]
    if shape[0] * shape[1] != size:
        raise ValueError(f"Qwen3-VL visual grid {shape} does not match its {size} embeddings.")
    return shape


def _load_qwen_generation_model(clip):
    model = _qwen_clip_model(clip)
    if model is None:
        raise ValueError("Qwen multimodal generation requires a supported Core Qwen model wrapper.")
    if hasattr(clip, "load_model"):
        clip.load_model()
    device = getattr(getattr(clip, "patcher", None), "load_device", None) or getattr(
        model, "execution_device", None
    )
    if device is None:
        device = model.transformer.get_input_embeddings().weight.device
    if hasattr(model, "reset_clip_options"):
        model.reset_clip_options()
        model.set_clip_options({"layer": None, "execution_device": device})
    return model, device


def _preprocess_qwen3vl_image(model, device, image):
    embedding, extra = model.transformer.preprocess_embed(
        {"type": "image", "data": image, "original_type": "image"}, device=device
    )
    if (
        embedding is None
        or not isinstance(extra, dict)
        or extra.get("grid") is None
        or not extra.get("deepstack")
    ):
        raise ValueError("Qwen3-VL image preprocessing did not return grid and DeepStack metadata.")
    return {"kind": "image", "embedding": embedding, "extra": extra}


def _preprocess_qwen3vl_video_pair(model, device, frames):
    flatten, grid = process_video_block(frames)
    embedding, deepstack = model.transformer.visual(
        flatten.to(device, dtype=torch.float32), grid
    )
    if embedding is None or not deepstack:
        raise ValueError(
            "Qwen3-VL video preprocessing did not return temporal and DeepStack embeddings."
        )
    return {
        "kind": "video",
        "embedding": embedding,
        "extra": {"grid": grid, "deepstack": deepstack},
    }


def _fuse_preprocessed_qwen3vl_images(blocks, config, device):
    if not blocks:
        raise ValueError("Active visual fusion with video requires at least one image source.")
    shapes = [_visual_grid_shape(block) for block in blocks]
    size = blocks[0]["embedding"].reshape(-1, blocks[0]["embedding"].shape[-1]).shape[0]
    primary = [
        block["embedding"].reshape(-1, block["embedding"].shape[-1]).to(device)
        for block in blocks
    ]
    deepstacks = {
        index: block["extra"]["deepstack"] for index, block in enumerate(blocks)
    }
    mask_cache = {}
    fused = fuse_visual_token_sources(primary, config, device, mask_cache, size, shapes)
    fused_deepstack = fuse_deepstack_layers(
        deepstacks, config, device, mask_cache, size, shapes
    )
    extra = dict(blocks[0]["extra"])
    extra["deepstack"] = fused_deepstack
    return {"kind": "image", "embedding": fused, "extra": extra}


def _attach_preprocessed_visuals(rows, image_blocks, video_blocks):
    if len(rows) != 1:
        raise ValueError("Qwen3-VL mixed image/video generation requires one tokenizer batch.")
    image_queue = list(image_blocks)
    video_queue = list(video_blocks)
    tagged = []
    for index, value in enumerate(rows[0]):
        block = None
        if isinstance(value, numbers.Integral) and int(value) == QWEN_IMAGE_PAD_ID and image_queue:
            block = image_queue.pop(0)
        elif isinstance(value, numbers.Integral) and int(value) == QWEN_VIDEO_PAD_ID and video_queue:
            block = video_queue.pop(0)
        if block is not None:
            marker = {
                "type": "embedding",
                "data": block["embedding"],
                "_textgen_visual": block,
            }
            rows[0][index] = marker
            tagged.append(marker)
    if image_queue or video_queue:
        raise ValueError(
            "Qwen3-VL prompt did not contain every required image and video placeholder."
        )
    return tagged


def generate_qwen3vl_video(
    clip,
    full_prompt,
    images,
    video_frames,
    fusion_config,
    generation_args,
    thinking=False,
):
    model, device = _load_qwen_generation_model(clip)
    if not hasattr(model.transformer, "build_image_inputs"):
        raise ValueError("Video input requires a Core Qwen3-VL 4B, 8B, or 32B model wrapper.")

    context = (
        model_management.cuda_device_context(device)
        if hasattr(model_management, "cuda_device_context")
        else nullcontext()
    )
    with context:
        image_blocks = [
            _preprocess_qwen3vl_image(model, device, image) for image in images
        ]
        if fusion_config is not None:
            if fusion_config.get("save_blended_embeds", False):
                logging.warning(
                    "UC_TextGenerate ignores save_blended_embeds in mixed image/video generation."
                )
            image_blocks = [
                _fuse_preprocessed_qwen3vl_images(image_blocks, fusion_config, device)
            ]
        video_blocks = [
            _preprocess_qwen3vl_video_pair(
                model, device, video_frames[index:index + 2]
            )
            for index in range(0, video_frames.shape[0], 2)
        ]

        tokens = clip.tokenize(
            full_prompt, skip_template=True, min_length=1, thinking=thinking
        )
        rows = _token_rows(tokens)
        tagged = _attach_preprocessed_visuals(rows, image_blocks, video_blocks)
        other_values = [
            value
            for row in rows
            for value in row
            if not isinstance(value, numbers.Integral)
        ]
        embeds, _, _, embeds_info = model.process_tokens(rows, device)
        if len(embeds_info) != len(other_values):
            raise ValueError(
                "Qwen3-VL could not preserve the prepared multimodal token layout."
            )

        tagged_ids = {id(marker) for marker in tagged}
        mapped = 0
        for source, info in zip(other_values, embeds_info, strict=True):
            if id(source) not in tagged_ids:
                continue
            block = source["_textgen_visual"]
            info["type"] = "image"
            info["extra"] = block["extra"]
            info["textgen_source_kind"] = block["kind"]
            mapped += 1
        if mapped != len(tagged):
            raise ValueError(
                "Qwen3-VL could not map every prepared visual embedding into the prompt."
            )

        logging.info(
            "UC_TextGenerate Qwen3-VL media blocks: %s",
            [
                (
                    info.get("textgen_source_kind"),
                    int(info["index"]),
                    int(info["size"]),
                )
                for info in embeds_info
                if info.get("textgen_source_kind") is not None
            ],
        )

        position_ids, visual_mask, deepstack = model.transformer.build_image_inputs(
            embeds, embeds_info
        )
        return model.transformer.generate(
            embeds,
            **generation_args,
            position_ids=position_ids,
            visual_pos_masks=visual_mask,
            deepstack_embeds=deepstack,
        )
