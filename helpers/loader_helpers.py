import re
from typing import Any
import torch

import comfy.lora
import comfy.lora_convert
import comfy.sd
import comfy.utils
import folder_paths


_TEXT_PROJECTION_KEY = (
    "detector.backbone.language_backbone.encoder.text_projection"
)

TRANSFORMER_BLOCK_REGEX = re.compile(
    r"\b(transformer_blocks|single_transformer_blocks|joint_transformer_blocks|double_blocks|single_blocks|joint_blocks|blocks|layers)[._](\d+)(?=[._]|$)"
)


def extract_transformer_block_index(key: str) -> int | None:
    match = TRANSFORMER_BLOCK_REGEX.search(key)
    if match:
        return int(match.group(2))
    return None


def parse_block_ranges(spec: str) -> set[int]:
    if not spec or not spec.strip():
        return set()
    result = set()
    tokens = [t.strip() for t in spec.split(",") if t.strip()]
    for token in tokens:
        if "-" in token:
            parts = token.split("-")
            if len(parts) == 2 and parts[0].strip() and parts[1].strip():
                try:
                    start = int(parts[0].strip())
                    end = int(parts[1].strip())
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid block range '{token}' in '{spec}': expected integers"
                    ) from exc
                if start <= end:
                    result.update(range(start, end + 1))
                else:
                    result.update(range(end, start + 1))
            else:
                raise ValueError(f"Invalid block range syntax '{token}' in '{spec}'")
        else:
            try:
                result.add(int(token))
            except ValueError as exc:
                raise ValueError(
                    f"Invalid block number '{token}' in '{spec}': expected integer"
                ) from exc
    return result


def parse_layer_patterns(spec: str) -> list[re.Pattern]:
    if not spec or not spec.strip():
        return []
    patterns = []
    for line in spec.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            patterns.append(re.compile(line))
        except re.error as exc:
            raise ValueError(f"Invalid layer regex pattern '{line}': {exc}") from exc
    return patterns


def filter_lora_patches(
    loaded: dict[str, Any],
    layer_filter: str,
    block_filter: str,
    whitelist: bool = True,
) -> dict[str, Any]:
    patterns = parse_layer_patterns(layer_filter)
    block_set = parse_block_ranges(block_filter)

    has_layer_filter = len(patterns) > 0
    has_block_filter = len(block_set) > 0

    if not has_layer_filter and not has_block_filter:
        return dict(loaded)

    filtered = {}
    for key, patch in loaded.items():
        block_idx = extract_transformer_block_index(key)
        is_block = block_idx is not None

        if has_layer_filter:
            stripped = (
                key[len("diffusion_model."):]
                if key.startswith("diffusion_model.")
                else key
            )
            no_weight = (
                stripped[:-len(".weight")]
                if stripped.endswith(".weight")
                else stripped
            )
            layer_match = any(
                p.search(key) or p.search(stripped) or p.search(no_weight)
                for p in patterns
            )
        else:
            layer_match = True

        if is_block:
            block_match = (block_idx in block_set) if has_block_filter else True
            matched = block_match and layer_match
        else:
            if has_layer_filter:
                matched = layer_match
            else:
                matched = False

        if whitelist:
            if matched:
                filtered[key] = patch
        else:
            if not matched:
                filtered[key] = patch

    return filtered


def load_filtered_lora_for_model(
    model,
    lora_name: str,
    strength_model: float,
    layer_filter: str = "",
    block_filter: str = "",
    whitelist: bool = True,
    cache: tuple | None = None,
):
    if strength_model == 0:
        return model, cache

    lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
    lora = None
    lora_metadata = None

    if cache is not None and cache[0] == lora_path:
        lora = cache[1]
        lora_metadata = cache[2] if len(cache) > 2 else None

    if lora is None:
        lora, lora_metadata = comfy.utils.load_torch_file(
            lora_path, safe_load=True, return_metadata=True
        )
        cache = (lora_path, lora, lora_metadata)

    key_map = {}
    if model is not None:
        key_map = comfy.lora.model_lora_keys_unet(model.model, key_map)

    lora_converted = comfy.lora_convert.convert_lora(lora)
    loaded = comfy.lora.load_lora(lora_converted, key_map)

    filtered_loaded = filter_lora_patches(
        loaded, layer_filter, block_filter, whitelist=whitelist
    )

    new_modelpatcher = model.clone()
    new_modelpatcher.add_patches(filtered_loaded, strength_model)
    if lora_metadata:
        new_modelpatcher.set_attachments("lora_metadata", lora_metadata)

    return new_modelpatcher, cache


def load_sam31_checkpoint(
    checkpoint_path,
    precision="fp32",
    output_vae=True,
    output_clip=True,
    output_model=True,
    embedding_directory=None,
    disable_dynamic=False,
):
    state_dict, metadata = comfy.utils.load_torch_file(
        checkpoint_path, return_metadata=True
    )
    state_dict.pop(_TEXT_PROJECTION_KEY, None)
    dtype_options = {"dtype": torch.float32} if precision == "fp32" else {}
    output = comfy.sd.load_state_dict_guess_config(
        state_dict,
        output_vae=output_vae,
        output_clip=output_clip,
        output_clipvision=False,
        embedding_directory=embedding_directory,
        output_model=output_model,
        model_options=dtype_options,
        te_model_options=dtype_options,
        metadata=metadata,
        disable_dynamic=disable_dynamic,
    )
    if output is None:
        raise RuntimeError(f"Could not detect a SAM 3.1 model in {checkpoint_path}.")
    if output[0] is not None:
        output[0].cached_patcher_init = (
            load_sam31_checkpoint,
            (
                checkpoint_path,
                precision,
                False,
                False,
                True,
                embedding_directory,
            ),
            0,
        )
    if output[1] is not None and getattr(output[1], "patcher", None) is not None:
        output[1].patcher.cached_patcher_init = (
            load_sam31_clip_patcher,
            (checkpoint_path, precision, embedding_directory),
        )
    return output


def load_sam31_clip_patcher(
    checkpoint_path,
    precision="fp32",
    embedding_directory=None,
    disable_dynamic=False,
):
    _, clip, _, _ = load_sam31_checkpoint(
        checkpoint_path,
        precision,
        output_vae=False,
        output_clip=True,
        output_model=False,
        embedding_directory=embedding_directory,
        disable_dynamic=disable_dynamic,
    )
    return clip.patcher
