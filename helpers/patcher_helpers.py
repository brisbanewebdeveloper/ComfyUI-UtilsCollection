"""Model-scoped patch helpers for diffusion-model cache nodes."""

from __future__ import annotations

import bisect
import contextvars
import hashlib
import logging
import math
import os
import re
import time
import types
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

import comfy.lora
import comfy.ldm.common_dit
from comfy.ldm.ideogram4 import model as ideogram4_model
import comfy.ldm.modules.attention
import comfy.model_management
import comfy.model_patcher
import comfy.model_prefetch
import comfy.ops
import comfy.patcher_extension
import comfy.utils
import folder_paths
from comfy.text_encoders.minimax import process_video_block, token_tags_from_embeds_info
from comfy.ldm.minimax import model as minimax_model
from safetensors import safe_open


MINIMAX_H3_CACHE_OWNER_KEY = "utilscollection_minimax_h3_cache"
MINIMAX_H3_PDD_OWNER_KEY = "utilscollection_minimax_h3_pdd_acc"
UNIFIED_ATTENTION_OWNER_KEY = "utilscollection_unified_attention"
MINIMAX_H3_RADIAL_WRAPPER_KEY = "utilscollection_minimax_h3_radial"
MINIMAX_H3_RADIAL_STATE_KEY = "utilscollection_minimax_h3_radial_state"
MINIMAX_H3_SLA_WRAPPER_KEY = "utilscollection_minimax_h3_sla"
MINIMAX_H3_SLA_STATE_KEY = "utilscollection_minimax_h3_sla_state"
MINIMAX_H3_SLA_SAMPLING_SCOPE_KEY = "utilscollection_minimax_h3_sla_sampling_scope"
MINIMAX_H3_PROJECTION_FOLDER = "clip_projections"
MINIMAX_H3_PROJECTED_KEY = "qwen3vl_32b"
MINIMAX_H3_PROJECTION_PATCH_KEY = "utilscollection_minimax_h3_projection"
MINIMAX_H3_SOURCE_KEYS = {"qwen3vl_4b", "qwen3vl_8b"}
MINIMAX_H3_PAD_TOKEN = 151643
MINIMAX_H3_VISION_START = 151652
MINIMAX_H3_VISION_END = 151653
IDEOGRAM4_DEBANNER_WRAPPER_KEY = "utilscollection_ideogram4_debanner"
IDEOGRAM4_DEBANNER_STATE_KEY = "utilscollection_ideogram4_debanner_state"
IDEOGRAM4_DEBANNER_BUNDLE = Path(__file__).resolve().parent.parent / "models" / "ideogram4_correction_v1.safetensors"


def _register_minimax_h3_projection_folder() -> None:
    path = os.path.join(folder_paths.models_dir, MINIMAX_H3_PROJECTION_FOLDER)
    if MINIMAX_H3_PROJECTION_FOLDER not in folder_paths.folder_names_and_paths:
        folder_paths.folder_names_and_paths[MINIMAX_H3_PROJECTION_FOLDER] = (
            [path],
            {".safetensors"},
        )
    else:
        folder_paths.add_model_folder_path(MINIMAX_H3_PROJECTION_FOLDER, path)


_register_minimax_h3_projection_folder()


def list_minimax_h3_projections() -> list[str]:
    return [
        name
        for name in folder_paths.get_filename_list(MINIMAX_H3_PROJECTION_FOLDER)
        if name.lower().endswith(".safetensors")
    ]


def _projection_scalar(data: dict[str, Any], metadata: dict[str, str], key: str) -> int:
    value = metadata.get(key, data.get(key))
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError(f"Projection {key} must contain one value.")
        value = value.item()
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Projection is missing a valid {key} value.") from exc


def load_minimax_h3_projection(name: str) -> tuple[dict[str, torch.Tensor], int]:
    if not name.lower().endswith(".safetensors"):
        raise ValueError("MiniMax H3 projections must use the .safetensors format.")
    path = folder_paths.get_full_path_or_raise(MINIMAX_H3_PROJECTION_FOLDER, name)
    with safe_open(path, framework="pt", device="cpu") as handle:
        data = {key: handle.get_tensor(key) for key in handle.keys()}
        metadata = handle.metadata() or {}
    required = ("mean_in", "std_in", "mean_out", "std_out")
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"Projection is missing required tensors: {', '.join(missing)}.")
    tap = _projection_scalar(data, metadata, "tap")
    return data, tap


def _projection_layers(data: dict[str, torch.Tensor]) -> list[int]:
    layers = set()
    for key in data:
        if not key.startswith("mlp."):
            continue
        parts = key.split(".")
        if len(parts) != 3 or parts[2] not in {"weight", "bias"}:
            raise ValueError(f"Unsupported projection tensor: {key}.")
        try:
            layers.add(int(parts[1]))
        except ValueError as exc:
            raise ValueError(f"Unsupported projection tensor: {key}.") from exc
    return sorted(layers)


class MiniMaxH3ProjectionModel(torch.nn.Module):
    def __init__(self, data: dict[str, torch.Tensor], tap: int):
        super().__init__()
        self.tap = tap
        for key in ("mean_in", "std_in", "mean_out", "std_out"):
            value = data[key]
            if value.ndim != 1 or not torch.is_floating_point(value):
                raise ValueError(f"Projection {key} must be a floating-point vector.")
            self.register_buffer(key, value.float())
        d_in = self.mean_in.shape[0]
        d_out = self.mean_out.shape[0]
        if self.std_in.shape != self.mean_in.shape or self.std_out.shape != self.mean_out.shape:
            raise ValueError("Projection mean and standard-deviation shapes must match.")
        if torch.any(self.std_in == 0) or torch.any(self.std_out == 0):
            raise ValueError("Projection standard deviations must be nonzero.")

        weight = data.get("W")
        if weight is not None:
            if weight.ndim != 2 or tuple(weight.shape) != (d_in, d_out):
                raise ValueError(
                    f"Projection W must have shape [{d_in}, {d_out}]."
                )
            self.linear_weight = torch.nn.Parameter(weight.float())
        else:
            self.register_parameter("linear_weight", None)

        residual_layers = []
        previous_out = d_in
        layer_indices = _projection_layers(data)
        for position, index in enumerate(layer_indices):
            weight_key = f"mlp.{index}.weight"
            bias_key = f"mlp.{index}.bias"
            if weight_key not in data:
                raise ValueError(f"Projection residual layer {index} has no weight.")
            residual_weight = data[weight_key]
            if residual_weight.ndim != 2 or residual_weight.shape[1] != previous_out:
                raise ValueError(f"Projection residual layer {index} has incompatible dimensions.")
            linear = torch.nn.Linear(
                residual_weight.shape[1],
                residual_weight.shape[0],
                bias=bias_key in data,
                dtype=residual_weight.dtype,
            )
            linear.weight = torch.nn.Parameter(residual_weight)
            if bias_key in data:
                bias = data[bias_key]
                if bias.ndim != 1 or bias.shape[0] != residual_weight.shape[0]:
                    raise ValueError(f"Projection residual layer {index} has an incompatible bias.")
                linear.bias = torch.nn.Parameter(bias)
            residual_layers.append(linear)
            if position < len(layer_indices) - 1:
                residual_layers.append(torch.nn.GELU())
            previous_out = residual_weight.shape[0]
        if residual_layers and previous_out != d_out:
            raise ValueError(f"Projection residual output must have {d_out} dimensions.")
        self.residual = torch.nn.Sequential(*residual_layers) if residual_layers else None
        if self.linear_weight is None and self.residual is None:
            raise ValueError("Projection requires a linear matrix or residual network.")

        sink = data.get("sink_out")
        if sink is not None:
            if sink.ndim != 1 or sink.shape[0] != d_out:
                raise ValueError(f"Projection sink_out must have {d_out} values.")
            self.register_buffer("sink_out", sink.float())
        else:
            self.register_buffer("sink_out", None)
        self.device = torch.device("cpu")

    @property
    def input_dimensions(self) -> int:
        return self.mean_in.shape[0]

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = hidden.float()
        if hidden.shape[-1] != self.input_dimensions:
            raise ValueError(
                f"Connected encoder produces {hidden.shape[-1]} dimensions, but projection expects {self.input_dimensions}."
            )
        normalized = (hidden - self.mean_in) / self.std_in
        projected = normalized @ self.linear_weight if self.linear_weight is not None else None
        if self.residual is not None:
            residual_dtype = self.residual[0].weight.dtype
            residual = self.residual(normalized.to(residual_dtype)).float()
            projected = residual if projected is None else projected + residual
        output = projected * self.std_out + self.mean_out
        if self.sink_out is not None and output.shape[1] > 0:
            output[:, 0] = self.sink_out
        return output


class _MiniMaxH3ProjectedTokenizer:
    clip_name = MINIMAX_H3_PROJECTED_KEY

    def __init__(self, source_tokenizer: Any, source_key: str):
        self.source_tokenizer = source_tokenizer
        self.source_key = source_key
        self.raw_tokenizer = getattr(source_tokenizer, source_key)

    def _text_entries(self, text: str) -> list[tuple]:
        if not text:
            return []
        batches = self.raw_tokenizer.tokenize_with_weights(
            text,
            return_word_ids=False,
            disable_weights=True,
        )
        if len(batches) != 1:
            raise ValueError("MiniMax H3 projected text exceeds the supported prompt length.")
        return list(batches[0])

    @staticmethod
    def _vision_entry(data: torch.Tensor, video_block: bool = False) -> dict[str, Any]:
        entry = {"type": "image", "data": data, "original_type": "image"}
        if video_block:
            entry["minimax_video_block"] = True
        return entry

    def tokenize_with_weights(
        self,
        text: str,
        return_word_ids: bool = False,
        images: list = [],
        minimax_ref_items=None,
        **_kwargs,
    ) -> dict[str, list[list[tuple]]]:
        entries = []

        def add_text(value: str) -> None:
            entries.extend(self._text_entries(value))

        def add_vision(data: torch.Tensor, video_block: bool = False) -> None:
            entries.append((MINIMAX_H3_VISION_START, 1.0))
            entries.append((self._vision_entry(data, video_block), 1.0))
            entries.append((MINIMAX_H3_VISION_END, 1.0))

        if minimax_ref_items:
            counters = {"image": 0, "audio": 0, "video": 0}
            for item in minimax_ref_items:
                kind = item["type"]
                if kind not in counters:
                    raise ValueError(f"Unsupported MiniMax H3 reference type: {kind}.")
                counters[kind] += 1
                if kind == "image":
                    add_text(f"<Picture {counters[kind]}>: ")
                    add_vision(item["data"])
                elif kind == "audio":
                    add_text(f"<Audio {counters[kind]}>: ")
                else:
                    frames = item["data"]
                    timestamps = item.get("timestamps")
                    if timestamps is None:
                        timestamps = [index / 2.0 for index in range(frames.shape[0])]
                    else:
                        timestamps = list(timestamps)
                    if frames.shape[0] % 2:
                        frames = torch.cat([frames, frames[-1:]], dim=0)
                        timestamps.append(timestamps[-1])
                    add_text(f"<Video {counters[kind]}>: ")
                    for index in range(0, frames.shape[0], 2):
                        midpoint = (timestamps[index] + timestamps[index + 1]) / 2.0
                        add_text(f"<{float(midpoint):.1f} seconds>")
                        add_vision(frames[index:index + 2], video_block=True)
        else:
            for index, image in enumerate(images, start=1):
                add_text(f"<Picture {index}>: ")
                add_vision(image)
        add_text(text)
        if not entries:
            entries.append((MINIMAX_H3_PAD_TOKEN, 1.0))
        if return_word_ids:
            entries = [entry + (0,) for entry in entries]
        return {MINIMAX_H3_PROJECTED_KEY: [entries]}


class MiniMaxH3ProjectedCLIP:
    def __init__(
        self,
        base: Any,
        projection_name: str,
        projection_model: MiniMaxH3ProjectionModel | None = None,
        projection_patcher: Any = None,
        original_methods: tuple[Callable, Callable] | None = None,
    ):
        source_key = getattr(base.cond_stage_model, "clip_name", None)
        if source_key not in MINIMAX_H3_SOURCE_KEYS:
            raise ValueError("MiniMax H3 projection requires a Qwen3-VL 4B or 8B text encoder.")
        source_model = getattr(
            base.cond_stage_model,
            getattr(base.cond_stage_model, "clip", ""),
            None,
        )
        if type(source_model).__module__ != "comfy.text_encoders.qwen3vl":
            raise ValueError(
                "MiniMax H3 projection requires the generic Qwen3-VL wrapper. "
                "Load the 4B or 8B encoder with CLIP type minimax."
            )
        if projection_model is None:
            data, tap = load_minimax_h3_projection(projection_name)
            projection_model = MiniMaxH3ProjectionModel(data, tap)
        self._base = base
        self._projection_name = projection_name
        self._projection_model = projection_model
        self._source_key = source_key
        self._capture = {}
        self.tokenizer = _MiniMaxH3ProjectedTokenizer(base.tokenizer, source_key)
        self._base.clip_layer(projection_model.tap)

        if projection_patcher is None:
            offload_device = comfy.model_management.text_encoder_offload_device()
            projection_model.to(offload_device)
            projection_model.device = offload_device
            projection_patcher = comfy.model_patcher.CoreModelPatcher(
                projection_model,
                load_device=base.patcher.load_device,
                offload_device=offload_device,
            )
        self._projection_patcher = projection_patcher
        self._install_encoder_patches(original_methods)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)

    def _install_encoder_patches(
        self, original_methods: tuple[Callable, Callable] | None
    ) -> None:
        owner_name = self._base.cond_stage_model.clip
        submodel = getattr(self._base.cond_stage_model, owner_name)
        transformer = submodel.transformer
        prefix = f"{owner_name}.transformer"
        if original_methods is None:
            original_preprocess = self._base.patcher.get_model_object(f"{prefix}.preprocess_embed")
            original_forward = self._base.patcher.get_model_object(f"{prefix}.forward")
        else:
            original_preprocess, original_forward = original_methods
        self._original_methods = (original_preprocess, original_forward)

        def preprocess_embed(current, embed, device):
            if embed.get("type") == "image" and embed.get("minimax_video_block", False):
                flattened, grid = process_video_block(embed["data"])
                merged, deepstack = current.visual(
                    flattened.to(device, dtype=torch.float32), grid
                )
                return merged, {"grid": grid, "deepstack": deepstack}
            return original_preprocess(embed, device)

        def forward(_current, *args, **kwargs):
            if self._capture.get("active"):
                embeds_info = kwargs.get("embeds_info", [])
                embeds = kwargs.get("embeds")
                if embeds is not None:
                    self._capture["tags"] = token_tags_from_embeds_info(
                        embeds.shape[1], embeds_info
                    )
            return original_forward(*args, **kwargs)

        self._cache_owned_methods = {
            f"{prefix}.preprocess_embed": types.MethodType(preprocess_embed, transformer),
            f"{prefix}.forward": types.MethodType(forward, transformer),
        }
        for path, method in self._cache_owned_methods.items():
            self._base.patcher.add_object_patch(path, method)

    def h3_cache_identity(self):
        return {
            "base": self._base, "model": self._projection_model,
            "patcher": self._projection_patcher, "name": self._projection_name,
            "source": self._source_key, "tap": self._projection_model.tap,
            "original_methods": self._original_methods,
            "owned_methods": self._cache_owned_methods,
        }

    def clone(self):
        return MiniMaxH3ProjectedCLIP(
            self._base.clone(),
            self._projection_name,
            projection_model=self._projection_model,
            projection_patcher=self._projection_patcher.clone(),
            original_methods=self._original_methods,
        )

    def tokenize(self, text: str, return_word_ids: bool = False, **kwargs):
        return self.tokenizer.tokenize_with_weights(text, return_word_ids, **kwargs)

    def _source_tokens(self, tokens: dict) -> dict:
        try:
            batches = tokens[MINIMAX_H3_PROJECTED_KEY]
        except (KeyError, TypeError) as exc:
            raise ValueError("Projected MiniMax H3 CLIP requires qwen3vl_32b tokens.") from exc
        return {self._source_key: batches}

    def _capture_encode(self, function: Callable, tokens: dict):
        self._capture.clear()
        self._capture["active"] = True
        try:
            output = function(self._source_tokens(tokens))
            tags = self._capture.get("tags")
        finally:
            self._capture.clear()
        if not torch.is_tensor(tags):
            raise RuntimeError("Projected MiniMax H3 encoder did not produce modality tags.")
        return output, tags

    def _project(self, hidden: torch.Tensor) -> torch.Tensor:
        comfy.model_management.load_models_gpu([self._projection_patcher])
        device = self._projection_patcher.load_device
        with comfy.model_management.cuda_device_context(device):
            projected = self._projection_model(hidden.to(device))
        return projected.to(comfy.model_management.intermediate_device())

    def encode_from_tokens(self, tokens, return_pooled=False, return_dict=False):
        output, tags = self._capture_encode(
            lambda source: self._base.encode_from_tokens(
                source, return_pooled=True, return_dict=True
            ),
            tokens,
        )
        output["cond"] = self._project(output["cond"])
        output["minimax_token_tags"] = tags
        if return_dict:
            return output
        pooled = output.get("pooled_output")
        if return_pooled:
            return output["cond"], pooled
        return output["cond"]

    def encode_from_tokens_scheduled(
        self, tokens, unprojected=False, add_dict={}, show_pbar=True
    ):
        conditioning, tags = self._capture_encode(
            lambda source: self._base.encode_from_tokens_scheduled(
                source,
                unprojected=unprojected,
                add_dict=add_dict,
                show_pbar=show_pbar,
            ),
            tokens,
        )
        projected = []
        for hidden, metadata in conditioning:
            metadata = metadata.copy()
            metadata["minimax_token_tags"] = tags
            projected.append([self._project(hidden), metadata])
        return projected


def patch_minimax_h3_clip_projection(clip: Any, projection_name: str) -> MiniMaxH3ProjectedCLIP:
    return MiniMaxH3ProjectedCLIP(clip.clone(), projection_name)


PDD_VIDEO_SHIFT = 12.0
PDD_AUDIO_SHIFT = 3.0
PDD_KNOT_TOLERANCE = 1e-4
PDD_DEFAULT_PARTITIONS = {
    5: (8, 8, 8, 4, 4),
    6: (8, 8, 4, 4, 4, 4),
    7: (8, 4, 4, 4, 4, 4, 4),
}
PDD_HEAD_KEYS = ("proj_out.weight", "proj_out.bias", "audio_proj_out.weight", "audio_proj_out.bias")
PDD_WRAPPER_KEY = "utilscollection_minimax_h3_pdd_acc"
PDD_ADDITIONAL_MODEL_KEY = "utilscollection_minimax_h3_pdd_heads"
PDD_FINAL_FORWARD_PATH = "diffusion_model.final_layer.forward"
PDD_BASIS_DIR = Path(__file__).resolve().parent / "assets" / "minimax_h3_pdd"


def shifted_pdd_sigma(shift: float, t):
    return shift * t / (1.0 + (shift - 1.0) * t)


def pdd_fine_sigmas(shift: float, num_steps: int) -> tuple[float, ...]:
    times = torch.linspace(1.0, 0.0, num_steps + 1, dtype=torch.float64)
    values = shifted_pdd_sigma(shift, times)
    values[0] = 1.0
    values[-1] = 0.0
    return tuple(float(value) for value in values)


def resolve_pdd_partition(num_steps: int, nfe: int, partition_text: str = "", trained_block: int = 4) -> tuple[int, ...]:
    text = (partition_text or "").strip()
    if text:
        try:
            sizes = tuple(int(part) for part in text.replace(" ", "").split(",") if part)
        except ValueError as exc:
            raise ValueError(f"partition '{partition_text}' is not a comma-separated integer list") from exc
        if any(size < 1 for size in sizes) or sum(sizes) != num_steps:
            raise ValueError(f"partition {sizes} must contain positive block sizes summing to {num_steps}; received {sum(sizes)}")
    elif num_steps % nfe == 0:
        sizes = (num_steps // nfe,) * nfe
    elif nfe in PDD_DEFAULT_PARTITIONS and sum(PDD_DEFAULT_PARTITIONS[nfe]) == num_steps:
        sizes = PDD_DEFAULT_PARTITIONS[nfe]
    else:
        raise ValueError(f"nfe {nfe} does not divide the {num_steps}-step PDD grid and has no default partition")
    allowed = (trained_block, 2 * trained_block)
    invalid = sorted({size for size in sizes if size not in allowed})
    if invalid:
        raise ValueError(f"partition {sizes} contains block sizes {invalid} outside the trained envelope {allowed}")
    return sizes


def pdd_partition_starts(sizes: tuple[int, ...]) -> tuple[int, ...]:
    starts = []
    offset = 0
    for size in sizes:
        starts.append(offset)
        offset += size
    return tuple(starts)


def pdd_block_boundaries(num_steps: int, sizes: tuple[int, ...]) -> tuple[float, ...]:
    fine = pdd_fine_sigmas(PDD_VIDEO_SHIFT, num_steps)
    return tuple(fine[start] for start in pdd_partition_starts(sizes)) + (0.0,)


def fuse_pdd_heads(bank_weight: torch.Tensor, bank_bias: torch.Tensor, fine: tuple[float, ...], sizes: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    intervals = [fine[index] - fine[index + 1] for index in range(bank_weight.shape[0])]
    weights = bank_weight.to(torch.float64)
    biases = bank_bias.to(torch.float64)
    fused_weights = []
    fused_biases = []
    for start, size in zip(pdd_partition_starts(sizes), sizes):
        indices = range(start, start + size)
        span = sum(intervals[index] for index in indices)
        fused_weights.append(sum((intervals[index] / span) * weights[index] for index in indices).to(torch.float32))
        fused_biases.append(sum((intervals[index] / span) * biases[index] for index in indices).to(torch.float32))
    return torch.stack(fused_weights).contiguous(), torch.stack(fused_biases).contiguous()


def select_pdd_block(sigma: float, bounds: tuple[float, ...], on_off_grid: str) -> int:
    count = len(bounds) - 1
    for index in range(count):
        if abs(sigma - bounds[index]) <= PDD_KNOT_TOLERANCE:
            return index
    if abs(sigma - bounds[-1]) <= PDD_KNOT_TOLERANCE:
        return count - 1
    if on_off_grid == "error":
        expected = ", ".join(f"{value:.6f}" for value in bounds)
        raise ValueError(f"MiniMax H3 PDD evaluated at sigma {sigma:.6f}, outside trained boundaries [{expected}]. Use this node's SIGMAS output with a sampler that evaluates only those boundaries.")
    if sigma >= bounds[0]:
        return 0
    for index in range(count):
        if sigma > bounds[index + 1]:
            return index
    return count - 1


def convert_pdd_lora(state_dict: dict[str, torch.Tensor], alpha: float) -> tuple[dict[str, torch.Tensor], set[str]]:
    converted = {}
    consumed = set()

    def take(key):
        consumed.add(key)
        return state_dict[key]

    def emit(destination, down, up, alpha_value):
        converted[f"{destination}.lora_A.weight"] = down.contiguous()
        converted[f"{destination}.lora_B.weight"] = up.contiguous()
        converted[f"{destination}.alpha"] = torch.tensor(float(alpha_value))

    def convert_qkv(source, destination):
        parts = [(take(f"{source}.attn.to_{name}.lora_down"), take(f"{source}.attn.to_{name}.lora_up")) for name in ("q", "k", "v")]
        rank = parts[0][0].shape[0]
        output = parts[0][1].shape[0]
        down = torch.cat([part[0] for part in parts], dim=0)
        up = torch.zeros(output * 3, rank * 3, dtype=parts[0][1].dtype)
        for index, (_, value) in enumerate(parts):
            up[index * output:(index + 1) * output, index * rank:(index + 1) * rank] = value
        emit(f"{destination}.attn.qkv_proj", down, up, alpha * 3.0)

    def convert_linear(source, destination, half_swap=False):
        down = take(f"{source}.lora_down")
        up = take(f"{source}.lora_up")
        if half_swap:
            midpoint = up.shape[0] // 2
            up = torch.cat((up[midpoint:], up[:midpoint]), dim=0)
        emit(destination, down, up, alpha)

    trunk_blocks = sorted({int(match.group(1)) for key in state_dict if (match := re.match(r"transformer_blocks\.(\d+)\.", key))})
    refiner_blocks = sorted({int(match.group(1)) for key in state_dict if (match := re.match(r"token_refiner\.refiner_blocks\.(\d+)\.", key))})
    for index in trunk_blocks:
        source = f"transformer_blocks.{index}"
        destination = f"diffusion_model.blocks.{index}"
        convert_qkv(source, destination)
        convert_linear(f"{source}.attn.to_out.0", f"{destination}.attn.out_proj")
        convert_linear(f"{source}.ff.net.0.proj", f"{destination}.mlp.fc1", True)
        convert_linear(f"{source}.ff.net.2", f"{destination}.mlp.fc2")
        if f"{source}.adaln_proj.linear.lora_down" in state_dict:
            convert_linear(f"{source}.adaln_proj.linear", f"{destination}.adaln_proj.linear")
    for index in refiner_blocks:
        source = f"token_refiner.refiner_blocks.{index}"
        destination = f"diffusion_model.token_refiner.blocks.{index}"
        convert_qkv(source, destination)
        convert_linear(f"{source}.attn.to_out.0", f"{destination}.attn.out_proj")
        convert_linear(f"{source}.ff.net.0.proj", f"{destination}.mlp.fc1", True)
        convert_linear(f"{source}.ff.net.2", f"{destination}.mlp.fc2")
    return converted, set(state_dict) - consumed


def split_pdd_state_dict(state_dict: dict[str, torch.Tensor], metadata: dict[str, str] | None, filename: str):
    metadata = metadata or {}
    config = {"num_steps": int(metadata.get("pdd_num_steps", 32)), "block_size": int(metadata.get("pdd_block_size", 4)), "alpha": float(metadata.get("lora_alpha", 64.0))}
    missing = [key for key in PDD_HEAD_KEYS if key not in state_dict]
    if missing:
        raise ValueError(f"{filename} is missing PDD head tensors: {missing}")
    heads = tuple(state_dict.pop(key) for key in PDD_HEAD_KEYS)
    video_weight, video_bias, audio_weight, audio_bias = heads
    steps = config["num_steps"]
    valid = video_weight.ndim == 3 and video_bias.shape == video_weight.shape[:2] and audio_weight.ndim == 3 and audio_bias.shape == audio_weight.shape[:2] and video_weight.shape[0] == steps and audio_weight.shape[0] == steps
    if not valid:
        raise ValueError(f"{filename} has invalid PDD head shapes: video {list(video_weight.shape)}, video bias {list(video_bias.shape)}, audio {list(audio_weight.shape)}, audio bias {list(audio_bias.shape)}")
    if any(key.startswith("diffusion_model.") for key in state_dict):
        invalid = [key for key in state_dict if not (key.startswith("diffusion_model.") and key.endswith((".lora_A.weight", ".lora_B.weight", ".alpha", ".diff", ".diff_b")))]
        if invalid:
            raise ValueError(f"{filename} contains unexpected converted keys: {invalid[:4]}")
        config["source_format"] = "converted"
        lora_state = dict(state_dict)
    else:
        config["source_format"] = "original"
        lora_state, leftovers = convert_pdd_lora(state_dict, config["alpha"])
        if leftovers:
            raise ValueError(f"{filename} contains unrecognized keys: {sorted(leftovers)[:4]}")
    return lora_state, heads, config


def pdd_table_sha(table: torch.Tensor) -> str:
    value = table.detach().to(torch.float32).contiguous().cpu()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()[:16]


def rebase_pdd_adaln(lora_state: dict[str, torch.Tensor], center: torch.Tensor, basis: torch.Tensor) -> tuple[dict[str, torch.Tensor], int]:
    rebased = dict(lora_state)
    center64 = center.to(torch.float64)
    basis64 = basis.to(torch.float64)
    count = 0
    for key in list(rebased):
        if not key.endswith(".adaln_proj.linear.lora_A.weight"):
            continue
        module = key[:-len(".lora_A.weight")]
        down = rebased.pop(f"{module}.lora_A.weight").to(torch.float64)
        up = rebased.pop(f"{module}.lora_B.weight").to(torch.float64)
        alpha = rebased.pop(f"{module}.alpha", None)
        scale = float(alpha) / down.shape[0] if alpha is not None else 1.0
        rebased[f"{module}.diff"] = (scale * (up @ (down @ basis64))).to(torch.float32).contiguous()
        rebased[f"{module}.diff_b"] = (scale * (up @ (down @ center64))).to(torch.float32).contiguous()
        count += 1
    return rebased, count


def rebase_pdd_for_pruned_model(model: Any, lora_state: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], str]:
    diffusion_model = model.get_model_object("diffusion_model")
    if not diffusion_model.use_adaln_curves:
        return lora_state, ""
    if not any(key.endswith(".adaln_proj.linear.lora_A.weight") for key in lora_state):
        return lora_state, "pruned model without dense AdalN LoRA modules"
    table = diffusion_model.adaln_t_table.detach().to(torch.float32).cpu()
    candidates = []
    for path in sorted(PDD_BASIS_DIR.glob("basis_*.safetensors")):
        data = comfy.utils.load_torch_file(str(path), safe_load=True)
        candidates.append(path.name)
        basis_table = data["adaln_t_table"]
        if basis_table.shape == table.shape and torch.allclose(basis_table, table, atol=1e-6):
            trunk = path.stem.removeprefix("basis_")
            rebased, count = rebase_pdd_adaln(lora_state, data["c"], data["V"])
            return rebased, f"rebased {count} AdalN modules onto {trunk} curve basis"
    raise ValueError(f"MiniMax H3 PDD cannot match this pruned model's AdalN table ({list(table.shape)}, sha {pdd_table_sha(table)}) against {candidates}")


class MiniMaxH3PDDHeadBank(torch.nn.Module):
    def __init__(self, video_weight: torch.Tensor, video_bias: torch.Tensor, audio_weight: torch.Tensor, audio_bias: torch.Tensor) -> None:
        super().__init__()
        # Core's ModelPatcher loads parameter-backed modules; buffer-only modules
        # are omitted from its load list and remain on the offload device.
        self.register_parameter("video_weight", torch.nn.Parameter(video_weight, requires_grad=False))
        self.register_parameter("video_bias", torch.nn.Parameter(video_bias, requires_grad=False))
        self.register_parameter("audio_weight", torch.nn.Parameter(audio_weight, requires_grad=False))
        self.register_parameter("audio_bias", torch.nn.Parameter(audio_bias, requires_grad=False))

    def project(self, video: torch.Tensor, audio: torch.Tensor, block: int) -> tuple[torch.Tensor, torch.Tensor]:
        video_weight = comfy.ops.cast_to_input(self.video_weight[block], video, copy=False)
        video_bias = comfy.ops.cast_to_input(self.video_bias[block], video, copy=False)
        audio_weight = comfy.ops.cast_to_input(self.audio_weight[block], audio, copy=False)
        audio_bias = comfy.ops.cast_to_input(self.audio_bias[block], audio, copy=False)
        return F.linear(video, video_weight, video_bias), F.linear(audio, audio_weight, audio_bias)


class MiniMaxH3PDDExecutionState:
    def __init__(self) -> None:
        self.sigma = contextvars.ContextVar("utilscollection_minimax_h3_pdd_sigma", default=None)


def make_pdd_diffusion_wrapper(state: MiniMaxH3PDDExecutionState):
    def pdd_diffusion_wrapper(executor, x, timestep, context, transformer_options=None, **kwargs):
        options = transformer_options or {}
        diffusion_model = executor.class_obj
        video_shift = float(options.get("minimax_h3_sigma_shift_video", getattr(diffusion_model, "sigma_shift_video", PDD_VIDEO_SHIFT)))
        audio_shift = float(options.get("minimax_h3_sigma_shift_audio", getattr(diffusion_model, "sigma_shift_audio", PDD_AUDIO_SHIFT)))
        if not (math.isclose(video_shift, PDD_VIDEO_SHIFT, abs_tol=1e-6) and math.isclose(audio_shift, PDD_AUDIO_SHIFT, abs_tol=1e-6)):
            raise ValueError(f"MiniMax H3 PDD requires SigmaShift 12.0/3.0; received {video_shift}/{audio_shift}.")
        payload = kwargs.get("minimax_payload") or {}
        audio_scale = float(payload.get("audio_scale", PDD_VIDEO_SHIFT / PDD_AUDIO_SHIFT))
        if not math.isclose(audio_scale, PDD_VIDEO_SHIFT / PDD_AUDIO_SHIFT, abs_tol=1e-6):
            raise ValueError(f"MiniMax H3 PDD requires audio_scale {PDD_VIDEO_SHIFT / PDD_AUDIO_SHIFT}; received {audio_scale}.")
        token = state.sigma.set(float(timestep.flatten()[0]) / 1000.0)
        try:
            return executor(x, timestep, context, options, **kwargs)
        finally:
            state.sigma.reset(token)
    return pdd_diffusion_wrapper


def make_pdd_final_forward(final_layer: Any, head_bank: MiniMaxH3PDDHeadBank, state: MiniMaxH3PDDExecutionState, bounds: tuple[float, ...], on_off_grid: str, strength: float):
    def pdd_final_forward(self, x, t_emb, video_seg, audio_seg, sigma_v, sample_sigmas, shifts):
        # Keep the PDD block selection tied to the wrapper-validated sampler state;
        # the remaining arguments are part of Core's current final-layer contract.
        sigma = state.sigma.get()
        if sigma is None:
            raise RuntimeError("MiniMax H3 PDD final layer ran outside its Core diffusion wrapper.")
        block = select_pdd_block(sigma, bounds, on_off_grid)
        shift, scale = self.adaln_proj(t_emb)

        def modulate(segment):
            start, stop, row = segment
            return (self.norm(x[start:stop]) * (1.0 + minimax_model._mod_row(scale, row, scale.dtype)) + minimax_model._mod_row(shift, row, shift.dtype)).to(torch.float32)

        video_hidden = modulate(video_seg)
        audio_hidden = modulate(audio_seg)
        video, audio = head_bank.project(video_hidden, audio_hidden, block)
        if strength != 1.0:
            native_video = self.video_out(video_hidden)
            native_audio = self.audio_out(audio_hidden)
            video = native_video + (video - native_video) * strength
            audio = native_audio + (audio - native_audio) * strength
        return video, audio
    return types.MethodType(pdd_final_forward, final_layer)


def patch_minimax_h3_pdd_model(model: Any, pdd_lora: str, nfe: int, partition: str, lora_strength: float, head_strength: float, on_off_grid: str) -> tuple[Any, torch.Tensor]:
    diffusion_model = model.get_model_object("diffusion_model")
    if not isinstance(diffusion_model, minimax_model.MiniMaxH3Model):
        raise ValueError("MiniMax H3 PDD requires a MiniMax H3 diffusion model.")
    model_options = getattr(model, "model_options", {})
    if model_options.get(MINIMAX_H3_PDD_OWNER_KEY):
        raise ValueError("MiniMax H3 PDD is already applied to this model.")
    if PDD_FINAL_FORWARD_PATH in getattr(model, "object_patches", {}):
        raise ValueError("MiniMax H3 PDD will not replace an existing final-layer object patch.")

    path = folder_paths.get_full_path_or_raise("loras", pdd_lora)
    state_dict, metadata = comfy.utils.load_torch_file(path, safe_load=True, return_metadata=True)
    lora_state, heads, config = split_pdd_state_dict(state_dict, metadata, pdd_lora)
    sizes = resolve_pdd_partition(config["num_steps"], int(nfe), partition, config["block_size"])
    lora_state, curve_note = rebase_pdd_for_pruned_model(model, lora_state)
    expected_keys = sum(key.endswith((".lora_A.weight", ".diff", ".diff_b")) for key in lora_state)
    loaded = comfy.lora.load_lora(lora_state, comfy.lora.model_lora_keys_unet(model.model, {}), log_missing=False)

    patched = model.clone()
    applied = patched.add_patches(loaded, lora_strength)
    if len(applied) != expected_keys:
        raise ValueError(f"MiniMax H3 PDD matched {len(applied)}/{expected_keys} patch keys; verify that PDD and diffusion-model trunks match.")
    final_layer = patched.get_model_object("diffusion_model.final_layer")
    video_weight, video_bias, audio_weight, audio_bias = heads
    if tuple(final_layer.video_out.weight.shape) != tuple(video_weight.shape[1:]):
        raise ValueError("PDD video head shape does not match this MiniMax H3 model.")
    if tuple(final_layer.audio_out.weight.shape) != tuple(audio_weight.shape[1:]):
        raise ValueError("PDD audio head shape does not match this MiniMax H3 model.")

    video_fine = pdd_fine_sigmas(PDD_VIDEO_SHIFT, config["num_steps"])
    audio_fine = pdd_fine_sigmas(PDD_AUDIO_SHIFT, config["num_steps"])
    fused_video_weight, fused_video_bias = fuse_pdd_heads(video_weight, video_bias, video_fine, sizes)
    fused_audio_weight, fused_audio_bias = fuse_pdd_heads(audio_weight, audio_bias, audio_fine, sizes)
    head_bank = MiniMaxH3PDDHeadBank(fused_video_weight, fused_video_bias, fused_audio_weight, fused_audio_bias)
    head_patcher = comfy.model_patcher.ModelPatcher(head_bank, load_device=patched.load_device, offload_device=patched.offload_device)
    patched.set_additional_models(PDD_ADDITIONAL_MODEL_KEY, [head_patcher])

    bounds = pdd_block_boundaries(config["num_steps"], sizes)
    execution_state = MiniMaxH3PDDExecutionState()
    patched.add_object_patch(PDD_FINAL_FORWARD_PATH, make_pdd_final_forward(final_layer, head_bank, execution_state, bounds, on_off_grid, head_strength))
    wrapper_type = comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL
    patched.remove_wrappers_with_key(wrapper_type, PDD_WRAPPER_KEY)
    patched.add_wrapper_with_key(wrapper_type, PDD_WRAPPER_KEY, make_pdd_diffusion_wrapper(execution_state))
    patched.model_options[MINIMAX_H3_PDD_OWNER_KEY] = True
    if curve_note:
        LOG.info("MiniMax H3 PDD: %s", curve_note)
    return patched, torch.tensor(bounds, dtype=torch.float32)


# Cache heuristic adapted from ComfyUI-MiniMaxH3-Cache by lihaoyun6:
# https://github.com/lihaoyun6/ComfyUI-MiniMaxH3-Cache (GPL-3.0).
class MiniMaxH3Cache:
    """Reuse the residual produced by the complete MiniMax H3 block stack."""

    def __init__(
        self,
        reuse_threshold: float,
        start_percent: float,
        end_percent: float,
        max_steps: int,
        device: str,
        verbose: bool,
    ) -> None:
        self.reuse_threshold = reuse_threshold
        self.start_percent = start_percent
        self.end_percent = end_percent
        self.max_steps = max_steps
        self.device = device
        self.verbose = verbose
        self.total_steps = 1
        self._residual_buffer: torch.Tensor | None = None
        self._cache_valid = False
        self.reset()

    def reset(self) -> None:
        self._cache_valid = False
        self._residual_buffer = None
        self.previous_feature_signature: torch.Tensor | None = None
        self.layout_signature: tuple[Any, ...] | None = None
        self.last_seen_timestep: float | None = None
        self.step_counter = 0
        self.accumulated_relative_l1 = 0.0
        self.consecutive_skips = 0
        self.run_count = 0
        self.skip_count = 0

    def begin(self, total_steps: int) -> None:
        self.reset()
        self.total_steps = max(1, total_steps)

    def finish(self) -> None:
        if self.verbose and self.run_count + self.skip_count:
            total = self.run_count + self.skip_count
            speedup = total / max(1, self.run_count)
            logging.info(
                "[UtilsCollection MiniMax H3 Cache] Skipped %s/%s block-stack "
                "executions (%.2fx theoretical block-stack speedup).",
                self.skip_count,
                total,
                speedup,
            )
        self.reset()

    @staticmethod
    def _feature_signature(
        hidden_states: torch.Tensor,
        cache_ranges: tuple[tuple[int, int], ...],
    ) -> torch.Tensor:
        max_dim = min(64, hidden_states.shape[-1])
        signatures = []
        for start, end in cache_ranges:
            length = end - start
            if length <= 0:
                continue
            stride = max(1, length // 100)
            sampled = hidden_states[start:end:stride, :max_dim]
            signatures.append(sampled.detach().abs().mean(dim=-1))

        if not signatures:
            stride = max(1, hidden_states.shape[0] // 100)
            sampled = hidden_states[::stride, :max_dim]
            return sampled.detach().abs().mean(dim=-1).float().cpu()
        return torch.cat(signatures).float().cpu()

    @staticmethod
    def _timestep_value(timestep: Any) -> float | None:
        if isinstance(timestep, torch.Tensor):
            if timestep.numel() == 0:
                return None
            return float(timestep.detach().flatten()[0].item())
        if isinstance(timestep, (int, float)):
            return float(timestep)
        return None

    @property
    def cached_residual(self) -> torch.Tensor | None:
        return self._residual_buffer if self._cache_valid else None

    def _store_residual(self, residual: torch.Tensor) -> None:
        if self.device == "cuda" and residual.device.type != "cuda":
            raise ValueError(
                "MiniMax H3 cache device is set to cuda, but the model is not running on CUDA."
            )

        target_device = torch.device("cpu") if self.device == "cpu" else residual.device
        residual = residual.detach()
        try:
            buffer = self._residual_buffer
            if (
                buffer is None
                or buffer.shape != residual.shape
                or buffer.dtype != residual.dtype
                or buffer.device != target_device
            ):
                buffer = torch.empty(residual.shape, dtype=residual.dtype, device=target_device)
                self._residual_buffer = buffer
            buffer.copy_(residual, non_blocking=(target_device.type == "cuda"))
            self._cache_valid = True
        except torch.OutOfMemoryError:
            if self.device == "cuda":
                raise
            self._residual_buffer = residual.to("cpu", copy=True)
            self._cache_valid = True

    def _apply_residual(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = self.cached_residual
        if residual is None:
            return hidden_states
        if residual.device != hidden_states.device or residual.dtype != hidden_states.dtype:
            residual = residual.to(
                device=hidden_states.device,
                dtype=hidden_states.dtype,
                non_blocking=True,
            )
        return hidden_states + residual

    def __call__(self, args: dict[str, Any], extra_options: dict[str, Any]) -> dict[str, Any]:
        original_block = extra_options["original_block"]
        hidden_states = args["img"]
        cache_ranges = tuple(tuple(pair) for pair in args.get("cache_ranges", ()))
        current_layout = (
            tuple(hidden_states.shape),
            hidden_states.dtype,
            hidden_states.device,
            args.get("block_count"),
            cache_ranges,
        )

        if self.layout_signature is None:
            self.layout_signature = current_layout
        elif self.layout_signature != current_layout:
            total_steps = self.total_steps
            self.reset()
            self.total_steps = total_steps
            self.layout_signature = current_layout

        timestep = self._timestep_value(args.get("timestep"))
        if timestep is None:
            return original_block(args)
        if self.last_seen_timestep != timestep:
            self.last_seen_timestep = timestep
            self.step_counter += 1

        progress = self.step_counter / self.total_steps
        in_cache_range = self.start_percent <= progress <= self.end_percent
        skip_reason = "initial step"

        if self.cached_residual is not None and self.previous_feature_signature is not None:
            current_signature = self._feature_signature(hidden_states, cache_ranges)
            current_float = current_signature.float()
            previous_float = self.previous_feature_signature.float()
            difference = (current_float - previous_float).abs().mean().item()
            denominator = previous_float.abs().mean().item() + 1e-6
            self.accumulated_relative_l1 += difference / denominator

            below_threshold = self.accumulated_relative_l1 < self.reuse_threshold
            below_skip_limit = self.consecutive_skips < self.max_steps
            if below_threshold and below_skip_limit and in_cache_range:
                self.skip_count += 1
                self.consecutive_skips += 1
                if self.verbose:
                    logging.info(
                        "[UtilsCollection MiniMax H3 Cache] Step %s SKIP "
                        "(relative L1 %.4f < %.4f).",
                        self.step_counter,
                        self.accumulated_relative_l1,
                        self.reuse_threshold,
                    )
                return {"img": self._apply_residual(hidden_states)}

            reasons = []
            if not below_threshold:
                reasons.append("threshold reached")
            if not below_skip_limit:
                reasons.append("maximum consecutive skips reached")
            if not in_cache_range:
                reasons.append("outside cache range")
            skip_reason = ", ".join(reasons)

        if self.verbose:
            logging.info(
                "[UtilsCollection MiniMax H3 Cache] Step %s RUN (%s).",
                self.step_counter,
                skip_reason,
            )

        self.run_count += 1
        self.consecutive_skips = 0
        self._cache_valid = False
        self.previous_feature_signature = self._feature_signature(hidden_states, cache_ranges)
        start_hidden_states = hidden_states.clone()
        result = original_block(args)
        output = result["img"]
        self._store_residual(output - start_hidden_states)
        self.accumulated_relative_l1 = 0.0
        return result


class MiniMaxH3SamplingScope:
    """Give one cache object an exact outer-sampling lifecycle."""

    def __init__(self, cache: MiniMaxH3Cache) -> None:
        self.cache = cache

    def __call__(self, sample_fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        sigmas = kwargs.get("sigmas")
        if sigmas is None and len(args) > 3:
            sigmas = args[3]
        if not isinstance(sigmas, torch.Tensor) or sigmas.ndim != 1 or len(sigmas) < 2:
            self.cache.reset()
            raise ValueError("MiniMax H3 cache could not read the sampler sigma schedule.")

        self.cache.begin(len(sigmas) - 1)
        try:
            return sample_fn(*args, **kwargs)
        finally:
            self.cache.finish()


def run_minimax_h3_blocks(
    model: minimax_model.MiniMaxH3Model,
    hidden_states: torch.Tensor,
    timestep_embedding: torch.Tensor,
    mod_segments: list[tuple[int, int, int]],
    rope_freqs: torch.Tensor,
    transformer_options: dict[str, Any],
    start: int = 0,
    end: int | None = None,
) -> torch.Tensor:
    """Run a bounded H3 block range while preserving Core block replacements."""

    blocks_replace = transformer_options.get("patches_replace", {}).get("dit", {})
    end = len(model.blocks) if end is None else end
    blocks = list(model.blocks[start:end])
    prefetch_queue = comfy.model_prefetch.make_prefetch_queue(
        blocks, hidden_states.device, transformer_options
    )
    for index in range(start, end):
        block = model.blocks[index]
        comfy.model_prefetch.prefetch_queue_pop(
            prefetch_queue, hidden_states.device, block, malloc_scope="block"
        )
        if ("double_block", index) in blocks_replace:

            def block_wrapper(block_args: dict[str, Any]) -> dict[str, torch.Tensor]:
                return {
                    "img": block(
                        block_args["img"],
                        block_args["t_emb"],
                        block_args["mod_segments"],
                        block_args["rope_freqs"],
                        transformer_options=block_args["transformer_options"],
                    )
                }

            hidden_states = blocks_replace[("double_block", index)](
                {
                    "img": hidden_states,
                    "t_emb": timestep_embedding,
                    "mod_segments": mod_segments,
                    "rope_freqs": rope_freqs,
                    "transformer_options": transformer_options,
                },
                {"original_block": block_wrapper},
            )["img"]
        else:
            hidden_states = block(
                hidden_states,
                timestep_embedding,
                mod_segments,
                rope_freqs,
                transformer_options=transformer_options,
            )
    if prefetch_queue is not None:
        comfy.model_prefetch.prefetch_queue_pop(
            prefetch_queue, hidden_states.device, None, malloc_scope="block"
        )
    return hidden_states


def minimax_h3_block_patch_forward(
    self: minimax_model.MiniMaxH3Model,
    x: list[torch.Tensor],
    timestep: torch.Tensor,
    context: torch.Tensor,
    transformer_options: dict[str, Any] = {},
    minimax_payload: dict[str, Any] | None = None,
    **kwargs: Any,
) -> list[torch.Tensor]:
    """Current Core H3 forward with one model-patchable block-loop boundary."""

    del kwargs
    video_x, audio_x = x[0], x[1]
    orig_t, orig_h, orig_w = video_x.shape[2:5]
    video_x = comfy.ldm.common_dit.pad_to_patch_size(video_x, self.patch_size)
    if video_x.shape[0] != 1:
        raise ValueError("MiniMax H3 supports batch size 1")
    payload = minimax_payload or {}
    device = video_x.device
    dtype = context.dtype

    latent_t, lat_h, lat_w = video_x.shape[2:5]
    audio_t = audio_x.shape[-1]
    text_len = context.shape[1]
    layout = payload.get("layout")
    if layout is None or layout.signature != (text_len, latent_t, lat_h, lat_w, audio_t):
        layout = minimax_model.PackedLayout(
            text_len,
            latent_t,
            lat_h,
            lat_w,
            audio_t,
            keyframes=payload.get("keyframes"),
            refs=payload.get("refs"),
        )

    shift_v = float(
        transformer_options.get("minimax_h3_sigma_shift_video", self.sigma_shift_video)
    )
    shift_a = float(
        transformer_options.get("minimax_h3_sigma_shift_audio", self.sigma_shift_audio)
    )
    sigma_v = (timestep.flatten()[0] / 1000.0).float().clamp(min=1e-6)
    t_v = float(1.0 - sigma_v)
    t_a = float(1.0 - minimax_model.time_shift_sigma(sigma_v, shift_v, shift_a))

    vis_aug = float(
        payload.get("visual_cond_noise_aug", minimax_model.VISUAL_COND_TIMESTEP)
    )
    aud_aug = float(
        payload.get("audio_cond_noise_aug", minimax_model.AUDIO_COND_TIMESTEP)
    )
    has_vis_cond = any(kind in ("cond", "ref_img") for _, _, kind in layout.segments)
    has_aud_cond = any(kind in ("cond_audio", "ref_audio") for _, _, kind in layout.segments)
    seg_t = {
        "text": t_v,
        "video": t_v,
        "audio": t_a,
        "cond": max(t_v, vis_aug),
        "ref_img": max(t_v, vis_aug),
        "cond_audio": max(t_a, aud_aug),
        "ref_audio": max(t_a, aud_aug),
    }
    unique_t = sorted(
        {t_v, t_a}
        | ({seg_t["cond"]} if has_vis_cond else set())
        | ({seg_t["cond_audio"]} if has_aud_cond else set())
    )
    t_row = {value: index for index, value in enumerate(unique_t)}
    seg_tag = {
        "text": 1,
        "video": 0,
        "audio": 2,
        "cond": 0,
        "ref_img": 0,
        "cond_audio": 2,
        "ref_audio": 2,
    }

    text_tags = payload.get("text_token_tags")
    mod_segments = []
    for start, end, kind in layout.segments:
        row_base = t_row[seg_t[kind]] * 3
        if kind == "text" and text_tags is not None:
            tags = text_tags.view(-1).tolist()
            run_start = 0
            for index in range(1, end - start + 1):
                if index == end - start or tags[index] != tags[run_start]:
                    mod_segments.append(
                        (start + run_start, start + index, row_base + int(tags[run_start]))
                    )
                    run_start = index
        else:
            mod_segments.append((start, end, row_base + seg_tag[kind]))

    img_update = layout.img_update.to(device)
    audio_update = layout.audio_update.to(device)
    video_rows = minimax_model.patchify_video(video_x.to(torch.float32), self.patch_size)
    audio_rows = minimax_model.pack_audio(audio_x.to(torch.float32))
    cond_video_rows = self._cond_video_rows(payload, device)
    cond_audio_rows = self._cond_audio_rows(payload, device)

    all_video_rows = video_rows
    if cond_video_rows is not None:
        all_video_rows = torch.empty(
            img_update.shape[0], video_rows.shape[1], dtype=torch.float32, device=device
        )
        all_video_rows[~img_update] = cond_video_rows
        all_video_rows[img_update] = video_rows
    all_audio_rows = audio_rows
    if cond_audio_rows is not None:
        all_audio_rows = torch.empty(
            audio_update.shape[0], audio_rows.shape[1], dtype=torch.float32, device=device
        )
        all_audio_rows[~audio_update] = cond_audio_rows
        all_audio_rows[audio_update] = audio_rows

    video_embed = self.video_patch_proj(all_video_rows).to(dtype)
    audio_embed = self.audio_patch_proj(all_audio_rows).to(dtype)
    text_states = context[0]
    if text_states.shape[-1] != self.hidden_size:
        text_states = self.token_refiner(
            self.condition_proj(text_states), transformer_options=transformer_options
        )

    hidden_states = torch.empty(
        layout.seq_len, self.hidden_size, dtype=dtype, device=device
    )
    video_offset = audio_offset = 0
    for start, end, kind in layout.segments:
        length = end - start
        if kind == "text":
            hidden_states[start:end] = text_states
        elif kind in ("cond", "ref_img", "video"):
            hidden_states[start:end] = video_embed[video_offset : video_offset + length]
            video_offset += length
        else:
            hidden_states[start:end] = audio_embed[audio_offset : audio_offset + length]
            audio_offset += length

    t_values = torch.tensor(unique_t, dtype=torch.float32, device=device)
    if self.use_adaln_curves:
        table = comfy.model_management.cast_to(self.adaln_t_table, device=device)
        position = t_values.clamp(0.0, 1.0) * (table.shape[0] - 1)
        lower = position.floor().long().clamp(max=table.shape[0] - 2)
        timestep_embedding = torch.lerp(
            table[lower], table[lower + 1], (position - lower).unsqueeze(1)
        )
    else:
        timestep_embedding = self.time_embedder(t_values).to(dtype)

    rope_freqs = minimax_model.rope_rotation_table(
        self.rope_freqs(layout.position_ids, device), dtype
    )
    blocks_replace = transformer_options.get("patches_replace", {}).get("dit", {})
    cache_ranges = tuple(
        (start, end)
        for start, end, kind in layout.segments
        if kind in ("audio", "video")
    )
    if ("block_loop", 0) in blocks_replace:

        def block_loop_wrapper(block_args: dict[str, Any]) -> dict[str, torch.Tensor]:
            return {
                "img": run_minimax_h3_blocks(
                    self,
                    block_args["img"],
                    block_args["t_emb"],
                    block_args["mod_segments"],
                    block_args["rope_freqs"],
                    block_args["transformer_options"],
                    block_args.get("start", 0),
                    block_args.get("end"),
                )
            }

        hidden_states = blocks_replace[("block_loop", 0)](
            {
                "img": hidden_states,
                "timestep": timestep,
                "t_emb": timestep_embedding,
                "mod_segments": mod_segments,
                "rope_freqs": rope_freqs,
                "transformer_options": transformer_options,
                "cache_ranges": cache_ranges,
                "target_ranges": tuple(
                    (start, end, kind)
                    for start, end, kind in layout.segments
                    if kind in ("audio", "video")
                ),
                "block_count": len(self.blocks),
            },
            {"original_block": block_loop_wrapper},
        )["img"]
    else:
        hidden_states = run_minimax_h3_blocks(
            self,
            hidden_states,
            timestep_embedding,
            mod_segments,
            rope_freqs,
            transformer_options,
        )

    video_seg = next(
        (start, end, t_row[seg_t["video"]])
        for start, end, kind in layout.segments
        if kind == "video"
    )
    audio_seg = next(
        (start, end, t_row[seg_t["audio"]])
        for start, end, kind in layout.segments
        if kind == "audio"
    )
    video_result, audio_result = self.final_layer(
        hidden_states,
        timestep_embedding,
        video_seg,
        audio_seg,
        sigma_v,
        transformer_options.get("sample_sigmas"),
        (shift_v, shift_a),
    )
    video_out = minimax_model.unpatchify_video(
        video_result,
        latent_t,
        lat_h // 2,
        lat_w // 2,
        self.latents_dim,
        self.patch_size,
    )
    video_out = video_out[:, :, :orig_t, :orig_h, :orig_w]
    audio_out = minimax_model.unpack_audio(audio_result)
    return [
        -video_out.to(video_x.dtype),
        -audio_out.to(audio_x.dtype),
    ]


def patch_minimax_h3_cache_model(
    model: Any,
    reuse_threshold: float,
    start_percent: float,
    end_percent: float,
    max_steps: int,
    device: str,
    verbose: bool,
) -> Any:
    """Return a cloned patcher with a reversible H3 block-loop cache."""

    if start_percent > end_percent:
        raise ValueError("Cache start percent must not exceed end percent.")

    patched_model = model.clone()
    diffusion_model = patched_model.model.diffusion_model
    if not isinstance(diffusion_model, minimax_model.MiniMaxH3Model):
        raise ValueError(
            "MiniMax H3 Cache requires a MiniMax H3 diffusion model; "
            f"received {diffusion_model.__class__.__name__}."
        )

    cache = MiniMaxH3Cache(
        reuse_threshold=reuse_threshold,
        start_percent=start_percent,
        end_percent=end_percent,
        max_steps=max_steps,
        device=device,
        verbose=verbose,
    )
    if hasattr(patched_model, "model_options"):
        patched_model.model_options[MINIMAX_H3_CACHE_OWNER_KEY] = True
    bound_forward = types.MethodType(minimax_h3_block_patch_forward, diffusion_model)
    patched_model.add_object_patch("diffusion_model._forward", bound_forward)
    patched_model.set_model_patch_replace(cache, "dit", "block_loop", 0)
    patched_model.add_wrapper(
        comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
        MiniMaxH3SamplingScope(cache),
    )
    return patched_model


DIFFUSION_CACHE_PRESETS = {
    "balanced": {
        "composition_cutoff": 0.75,
        "detail_protection": 0.15,
        "base_threshold": 0.05,
        "max_noise_span": 0.20,
    },
    "high_fidelity": {
        "composition_cutoff": 0.65,
        "detail_protection": 0.25,
        "base_threshold": 0.03,
        "max_noise_span": 0.12,
    },
    "high_speed": {
        "composition_cutoff": 0.85,
        "detail_protection": 0.10,
        "base_threshold": 0.08,
        "max_noise_span": 0.30,
    },
}


class MiniMaxH3DiffusionCache(MiniMaxH3Cache):
    """Reuse MiniMax H3 block residual anchored to physical diffusion noise phases."""

    def __init__(
        self,
        preset: str = "balanced",
        composition_cutoff: float = 0.75,
        detail_protection: float = 0.15,
        drift_tolerance: float = 1.0,
        max_noise_span: float = 0.20,
        device: str = "auto",
        verbose: bool = False,
    ) -> None:
        self.preset = preset
        self.device = device
        self.verbose = verbose
        self.drift_tolerance = max(0.01, float(drift_tolerance))

        if preset in DIFFUSION_CACHE_PRESETS:
            cfg = DIFFUSION_CACHE_PRESETS[preset]
            self.composition_cutoff = float(cfg["composition_cutoff"])
            self.detail_protection = float(cfg["detail_protection"])
            self.reuse_threshold = float(cfg["base_threshold"]) * self.drift_tolerance
            self.max_noise_span = float(cfg["max_noise_span"])
        else:
            self.composition_cutoff = float(composition_cutoff)
            self.detail_protection = float(detail_protection)
            self.reuse_threshold = 0.05 * self.drift_tolerance
            self.max_noise_span = float(max_noise_span)

        self.start_percent = 0.0
        self.end_percent = 1.0
        self.max_steps = 100
        self.sigmas: torch.Tensor | None = None
        self.sigma_max: float = 1.0
        self.sigma_min: float = 0.0
        self.total_steps: int = 1
        self._residual_buffer: torch.Tensor | None = None
        self._cache_valid = False
        self.cached_at_sigma: float | None = None
        self.reset()

    def reset(self) -> None:
        super().reset()
        self.cached_at_sigma = None

    def begin(self, schedule: Any) -> None:
        self.reset()
        if isinstance(schedule, torch.Tensor) and schedule.ndim == 1 and len(schedule) >= 2:
            self.sigmas = schedule.detach().cpu().float()
            self.total_steps = len(schedule) - 1
            self.sigma_max = float(self.sigmas[0].item())
            self.sigma_min = float(self.sigmas[-1].item())
        elif isinstance(schedule, (int, float)):
            self.total_steps = max(1, int(schedule))
            self.sigmas = None
            self.sigma_max = 1.0
            self.sigma_min = 0.0
        else:
            self.total_steps = 1
            self.sigmas = None
            self.sigma_max = 1.0
            self.sigma_min = 0.0

    def finish(self) -> None:
        if self.verbose and self.run_count + self.skip_count:
            total = self.run_count + self.skip_count
            speedup = total / max(1, self.run_count)
            logging.info(
                "[UtilsCollection MiniMax H3 Diffusion Cache] Skipped %s/%s block-stack "
                "executions (%.2fx speedup).",
                self.skip_count,
                total,
                speedup,
            )
        self.reset()

    def _normalize_sigma(self, timestep: float) -> float:
        if self.sigmas is not None:
            s_max = self.sigma_max
            s_min = self.sigma_min
            high = max(s_max, s_min)
            low = min(s_max, s_min)
            if timestep > 1.5 and high <= 1.05:
                current = timestep / 1000.0
            elif timestep <= 1.5 and high > 50.0:
                current = timestep * 1000.0
            else:
                current = timestep
            span = max(1e-6, high - low)
            norm = (current - low) / span
            return float(max(0.0, min(1.0, norm)))
        else:
            current = timestep / 1000.0 if timestep > 1.5 else timestep
            return float(max(0.0, min(1.0, current)))

    def __call__(self, args: dict[str, Any], extra_options: dict[str, Any]) -> dict[str, Any]:
        original_block = extra_options["original_block"]
        hidden_states = args["img"]
        cache_ranges = tuple(tuple(pair) for pair in args.get("cache_ranges", ()))
        current_layout = (
            tuple(hidden_states.shape),
            hidden_states.dtype,
            hidden_states.device,
            args.get("block_count"),
            cache_ranges,
        )

        if self.layout_signature is None:
            self.layout_signature = current_layout
        elif self.layout_signature != current_layout:
            total_steps = self.total_steps
            sigmas = self.sigmas
            self.reset()
            self.total_steps = total_steps
            self.sigmas = sigmas
            if sigmas is not None:
                self.sigma_max = float(sigmas[0].item())
                self.sigma_min = float(sigmas[-1].item())
            self.layout_signature = current_layout

        timestep = self._timestep_value(args.get("timestep"))
        if timestep is None:
            return original_block(args)
        if self.last_seen_timestep != timestep:
            self.last_seen_timestep = timestep
            self.step_counter += 1

        normalized_noise = self._normalize_sigma(timestep)
        in_cache_phase = self.detail_protection <= normalized_noise <= self.composition_cutoff
        skip_reason = "initial step"
        noise_delta = 0.0

        if self.cached_residual is not None and self.previous_feature_signature is not None:
            current_signature = self._feature_signature(hidden_states, cache_ranges)
            current_float = current_signature.float()
            previous_float = self.previous_feature_signature.float()
            difference = (current_float - previous_float).abs().mean().item()
            denominator = previous_float.abs().mean().item() + 1e-6
            self.accumulated_relative_l1 += difference / denominator

            cached_sigma = self.cached_at_sigma if self.cached_at_sigma is not None else normalized_noise
            noise_delta = abs(cached_sigma - normalized_noise)
            within_noise_budget = noise_delta <= self.max_noise_span
            below_threshold = self.accumulated_relative_l1 < self.reuse_threshold

            if in_cache_phase and within_noise_budget and below_threshold:
                self.skip_count += 1
                self.consecutive_skips += 1
                if self.verbose:
                    logging.info(
                        "[UtilsCollection MiniMax H3 Diffusion Cache] Step %s (noise %.3f) SKIP "
                        "(relative L1 %.4f < %.4f, delta noise %.3f <= %.3f).",
                        self.step_counter,
                        normalized_noise,
                        self.accumulated_relative_l1,
                        self.reuse_threshold,
                        noise_delta,
                        self.max_noise_span,
                    )
                return {"img": self._apply_residual(hidden_states)}

            reasons = []
            if not in_cache_phase:
                if normalized_noise > self.composition_cutoff:
                    reasons.append(
                        f"macro composition phase (noise {normalized_noise:.3f} > {self.composition_cutoff:.3f})"
                    )
                else:
                    reasons.append(
                        f"micro detail phase (noise {normalized_noise:.3f} < {self.detail_protection:.3f})"
                    )
            if not within_noise_budget:
                reasons.append(
                    f"max noise span reached (delta {noise_delta:.3f} > {self.max_noise_span:.3f})"
                )
            if not below_threshold:
                reasons.append(
                    f"drift threshold reached (L1 {self.accumulated_relative_l1:.4f} >= {self.reuse_threshold:.4f})"
                )
            skip_reason = ", ".join(reasons)

        if self.verbose:
            logging.info(
                "[UtilsCollection MiniMax H3 Diffusion Cache] Step %s (noise %.3f) RUN (%s).",
                self.step_counter,
                normalized_noise,
                skip_reason,
            )

        self.run_count += 1
        self.consecutive_skips = 0
        self._cache_valid = False
        self.cached_at_sigma = normalized_noise
        self.previous_feature_signature = self._feature_signature(hidden_states, cache_ranges)
        start_hidden_states = hidden_states.clone()
        result = original_block(args)
        output = result["img"]
        self._store_residual(output - start_hidden_states)
        self.accumulated_relative_l1 = 0.0
        return result


class MiniMaxH3DiffusionSamplingScope:
    """Give one diffusion cache object an exact outer-sampling lifecycle."""

    def __init__(self, cache: MiniMaxH3DiffusionCache) -> None:
        self.cache = cache

    def __call__(self, sample_fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        sigmas = kwargs.get("sigmas")
        if sigmas is None and len(args) > 3:
            sigmas = args[3]
        if not isinstance(sigmas, torch.Tensor) or sigmas.ndim != 1 or len(sigmas) < 2:
            self.cache.reset()
            raise ValueError("MiniMax H3 Diffusion Cache could not read the sampler sigma schedule.")

        self.cache.begin(sigmas)
        try:
            return sample_fn(*args, **kwargs)
        finally:
            self.cache.finish()


def patch_minimax_h3_diffusion_cache_model(
    model: Any,
    preset: str = "balanced",
    composition_cutoff: float = 0.75,
    detail_protection: float = 0.15,
    drift_tolerance: float = 1.0,
    max_noise_span: float = 0.20,
    device: str = "auto",
    verbose: bool = False,
) -> Any:
    """Return a cloned patcher with a physical diffusion-aware H3 block-loop cache."""

    if preset == "custom" and composition_cutoff < detail_protection:
        raise ValueError(
            f"Composition cutoff noise level ({composition_cutoff}) must be greater than "
            f"or equal to detail protection level ({detail_protection})."
        )

    if getattr(model, "model_options", {}).get(MINIMAX_H3_CACHE_OWNER_KEY):
        raise ValueError("MiniMax H3 Diffusion Cache cannot be combined with another MiniMax H3 Cache.")

    patched_model = model.clone()
    diffusion_model = patched_model.model.diffusion_model
    if not isinstance(diffusion_model, minimax_model.MiniMaxH3Model):
        raise ValueError(
            "MiniMax H3 Diffusion Cache requires a MiniMax H3 diffusion model; "
            f"received {diffusion_model.__class__.__name__}."
        )

    cache = MiniMaxH3DiffusionCache(
        preset=preset,
        composition_cutoff=composition_cutoff,
        detail_protection=detail_protection,
        drift_tolerance=drift_tolerance,
        max_noise_span=max_noise_span,
        device=device,
        verbose=verbose,
    )
    if hasattr(patched_model, "model_options"):
        patched_model.model_options[MINIMAX_H3_CACHE_OWNER_KEY] = True
    bound_forward = types.MethodType(minimax_h3_block_patch_forward, diffusion_model)
    patched_model.add_object_patch("diffusion_model._forward", bound_forward)
    patched_model.set_model_patch_replace(cache, "dit", "block_loop", 0)
    patched_model.add_wrapper(
        comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
        MiniMaxH3DiffusionSamplingScope(cache),
    )
    return patched_model

@dataclass(frozen=True, slots=True)
class MiniMaxH3RadialAttentionConfig:
    dense_blocks: int = 1
    dense_start_steps: int = 1
    dense_end_steps: int = 1
    block_size: int = 128
    decay_factor: float = 0.2
    allow_compile: bool = False

    def validate(self) -> MiniMaxH3RadialAttentionConfig:
        if self.dense_blocks < 0:
            raise ValueError("Dense block count must be zero or greater.")
        if self.dense_start_steps < 0 or self.dense_end_steps < 0:
            raise ValueError("Dense step counts must be zero or greater.")
        if self.block_size not in (64, 128):
            raise ValueError("MiniMax H3 Radial block size must be 64 or 128.")
        if not math.isfinite(self.decay_factor) or not 0.0 <= self.decay_factor <= 1.0:
            raise ValueError("MiniMax H3 Radial decay factor must be finite and in [0, 1].")
        if not isinstance(self.allow_compile, bool):
            raise TypeError("MiniMax H3 Radial allow_compile must be a boolean.")
        return self


@dataclass(frozen=True)
class MiniMaxH3SlaAttentionConfig:
    sparsity: float = 0.9
    block_size: int = 32
    dense_tail_steps: int = 1
    protect_reference_media: str = "Light"
    dense_backend: str = "comfy_kitchen"
    minimum_sequence_length: int = 8192
    dense_steps: str = "0"
    protect_audio: bool = True
    disable_fp16_accumulation: bool = True
    stabilize_routing: bool = False

    def validate(self) -> MiniMaxH3SlaAttentionConfig:
        if not math.isfinite(self.sparsity) or not 0.0 <= self.sparsity <= 0.95:
            raise ValueError("MiniMax H3 SLA sparsity must be finite and in [0, 0.95].")
        if self.block_size not in (32, 64, 128):
            raise ValueError("MiniMax H3 SLA block size must be 32, 64, or 128.")
        if self.minimum_sequence_length < 0:
            raise ValueError("MiniMax H3 SLA minimum sequence length must be zero or greater.")
        if self.dense_tail_steps < 0:
            raise ValueError("MiniMax H3 SLA dense tail steps must be zero or greater.")
        for value, label in (
            (self.protect_audio, "audio protection"),
            (self.disable_fp16_accumulation, "FP16/BF16 accumulation control"),
            (self.stabilize_routing, "routing stabilization"),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"MiniMax H3 SLA {label} must be a boolean.")
        if self.protect_reference_media not in SLA_REFERENCE_PROTECTION_MODES:
            raise ValueError("MiniMax H3 SLA reference-media protection must be Off, Light, or Heavy Enforcement.")
        if self.dense_backend not in SLA_DENSE_BACKENDS:
            raise ValueError("Choose a supported MiniMax H3 SLA dense backend.")
        _parse_h3_sla_dense_steps(self.dense_steps)
        return self


UNIFIED_ATTENTION_MODES = (
    "disabled",
    "FlashAttention",
    "SageAttention",
    "Sparse / MiniMax H3 SLA",
    "Sparse / MiniMax H3 Radial",
)

CUSTOM_SAGE_MODES = (
    "auto",
    "sageattn_qk_int8_pv_fp16_cuda",
    "sageattn_qk_int8_pv_fp16_triton",
    "sageattn_qk_int8_pv_fp8_cuda",
    "sageattn_qk_int8_pv_fp8_cuda++",
    "sageattn3",
    "sageattn3_per_block_mean",
)

SLA_REFERENCE_PROTECTION_MODES = ("Off", "Light", "Heavy Enforcement")
SLA_DENSE_BACKENDS = ("comfy_kitchen", "auto", "pytorch", *CUSTOM_SAGE_MODES[1:])
SLA_REFERENCE_LIGHT_SPARSITY = 0.85


def _call_attention_function(attention_function: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    def override(_original: Callable[..., torch.Tensor], *args: Any, **kwargs: Any) -> torch.Tensor:
        wrapped = getattr(attention_function, "__wrapped__", attention_function)
        return wrapped(*args, **kwargs)

    return override


def _call_attention_with_fallback(attention_function: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    def override(original: Callable[..., torch.Tensor], *args: Any, **kwargs: Any) -> torch.Tensor:
        return attention_function(original, *args, **kwargs)

    return override


def _attention_as_nhd(q, k, v, heads: int, already_hnd: bool):
    if already_hnd:
        batch, _, _, head_dim = q.shape
        return (q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)), batch, head_dim
    batch, _, hidden_size = q.shape
    head_dim = hidden_size // heads
    return (q.reshape(batch, -1, heads, head_dim), k.reshape(batch, -1, heads, head_dim), v.reshape(batch, -1, heads, head_dim)), batch, head_dim


def _attention_from_nhd(output, batch: int, heads: int, head_dim: int, hnd_output: bool):
    return output.transpose(1, 2) if hnd_output else output.reshape(batch, -1, heads * head_dim)


def _sage_mask(mask):
    if mask is None:
        return None
    if mask.ndim == 2:
        return mask[None]
    return mask[:, None] if mask.ndim == 3 else mask


def _make_sage_backend(mode: str, allow_compile: bool) -> Callable[..., torch.Tensor]:
    if mode == "auto":
        from sageattention import sageattn

        def sage_function(q, k, v, **kwargs):
            return sageattn(q, k, v, **kwargs)
    elif mode == "sageattn_qk_int8_pv_fp16_cuda":
        from sageattention import sageattn_qk_int8_pv_fp16_cuda

        def sage_function(q, k, v, **kwargs):
            return sageattn_qk_int8_pv_fp16_cuda(q, k, v, pv_accum_dtype="fp32", **kwargs)
    elif mode == "sageattn_qk_int8_pv_fp16_triton":
        from sageattention import sageattn_qk_int8_pv_fp16_triton

        def sage_function(q, k, v, **kwargs):
            return sageattn_qk_int8_pv_fp16_triton(q, k, v, pv_accum_dtype="fp32", **kwargs)
    elif mode in ("sageattn_qk_int8_pv_fp8_cuda", "sageattn_qk_int8_pv_fp8_cuda++"):
        from sageattention import sageattn_qk_int8_pv_fp8_cuda

        accumulation = "fp32+fp16" if mode.endswith("++") else "fp32+fp32"

        def sage_function(q, k, v, **kwargs):
            return sageattn_qk_int8_pv_fp8_cuda(q, k, v, pv_accum_dtype=accumulation, **kwargs)
    elif mode in ("sageattn3", "sageattn3_per_block_mean"):
        from sageattn3 import sageattn3_blackwell

        def sage_function(q, k, v, **kwargs):
            tensor_layout = kwargs.pop("tensor_layout", "NHD")
            if tensor_layout == "NHD":
                q, k, v = (tensor.transpose(1, 2) for tensor in (q, k, v))
            result = sageattn3_blackwell(
                q,
                k,
                v,
                per_block_mean=mode == "sageattn3_per_block_mean",
                **kwargs,
            )
            return result.transpose(1, 2) if tensor_layout == "NHD" else result
    else:
        raise ValueError(f"Unsupported custom SageAttention mode: {mode}")

    if not allow_compile:
        sage_function = torch.compiler.disable()(sage_function)

    def attention(q, k, v, heads, mask=None, skip_reshape=False, skip_output_reshape=False, **kwargs):
        if kwargs.get("low_precision_attention", True) is False:
            return comfy.ldm.modules.attention.attention_pytorch.__wrapped__(
                q, k, v, heads, mask=mask, skip_reshape=skip_reshape, skip_output_reshape=skip_output_reshape, **kwargs
            )
        input_dtype = v.dtype
        if q.dtype == torch.float32 or k.dtype == torch.float32 or v.dtype == torch.float32:
            q, k, v = q.to(torch.float16), k.to(torch.float16), v.to(torch.float16)
        (q, k, v), batch, head_dim = _attention_as_nhd(q, k, v, heads, skip_reshape)
        output = sage_function(q, k, v, attn_mask=_sage_mask(mask), is_causal=False, tensor_layout="NHD").to(input_dtype)
        return _attention_from_nhd(output, batch, heads, head_dim, skip_output_reshape)

    return attention


def _parse_h3_sla_dense_steps(value: str) -> frozenset[int]:
    if not isinstance(value, str):
        raise TypeError("MiniMax H3 SLA dense steps must be a comma-separated string.")
    steps = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            if "-" in item:
                first, last = (int(part.strip()) for part in item.split("-", 1))
                if first < 0 or last < 0:
                    raise ValueError
                steps.update(range(min(first, last), max(first, last) + 1))
            else:
                step = int(item)
                if step < 0:
                    raise ValueError
                steps.add(step)
        except ValueError:
            logging.getLogger(__name__).warning("MiniMax H3 SLA ignoring invalid dense step %r.", item)
    return frozenset(steps)


def _resolve_h3_sla_dense_backend(backend: str) -> Callable[..., torch.Tensor] | None:
    if backend == "auto":
        return None
    if backend == "pytorch":
        return _call_attention_function(comfy.ldm.modules.attention.attention_pytorch)
    if backend == "comfy_kitchen":
        for name in ("attention_comfy_kitchen_int8", "attention_ck", "attention_composable_kernel"):
            candidate = getattr(comfy.ldm.modules.attention, name, None)
            if callable(candidate):
                return _call_attention_function(candidate)
        raise RuntimeError("MiniMax H3 SLA dense backend comfy_kitchen is unavailable in this ComfyUI installation.")
    if backend in CUSTOM_SAGE_MODES:
        try:
            return _call_attention_function(_make_sage_backend(backend, False))
        except (AttributeError, ImportError) as error:
            raise RuntimeError(f"MiniMax H3 SLA dense backend {backend} is unavailable: {error}") from error
    raise ValueError("Choose a supported MiniMax H3 SLA dense backend.")


def _make_flash_backend(allow_compile: bool, cast_dtype: torch.dtype) -> Callable[..., torch.Tensor]:
    try:
        from flash_attn import flash_attn_func
        is_v3 = False
    except ImportError:
        try:
            from flash_attn_interface import flash_attn_func
            is_v3 = True
        except ImportError as error:
            raise RuntimeError("Custom FlashAttention requires flash_attn or flash_attn_interface.") from error

    def flash_function(q, k, v):
        result = flash_attn_func(q, k, v, causal=False) if is_v3 else flash_attn_func(q, k, v, dropout_p=0.0, causal=False)
        return result[0] if isinstance(result, tuple) else result

    if not allow_compile:
        flash_function = torch.compiler.disable()(flash_function)

    def attention(q, k, v, heads, mask=None, skip_reshape=False, skip_output_reshape=False, **_kwargs):
        if mask is not None:
            raise RuntimeError("Custom FlashAttention does not support attention masks.")
        input_dtype = v.dtype
        if q.dtype == torch.float32 or k.dtype == torch.float32 or v.dtype == torch.float32:
            q, k, v = q.to(cast_dtype), k.to(cast_dtype), v.to(cast_dtype)
        (q, k, v), batch, head_dim = _attention_as_nhd(q, k, v, heads, skip_reshape)
        output = flash_function(q, k, v).to(input_dtype)
        return _attention_from_nhd(output, batch, heads, head_dim, skip_output_reshape)

    return attention


def _model_compute_dtype(model: Any) -> torch.dtype:
    diffusion_model = model.get_model_object("diffusion_model")
    get_dtype = getattr(diffusion_model, "get_dtype_inference", None)
    dtype = get_dtype() if callable(get_dtype) else torch.float16
    return dtype if dtype in (torch.float16, torch.bfloat16) else torch.float16


def _ensure_transformer_options(model: Any) -> dict[str, Any]:
    options = dict(getattr(model, "model_options", {}).get("transformer_options", {}))
    model.model_options["transformer_options"] = options
    return options


def _h3_sage_forward(self, x, rope_freqs=None, transformer_options=None):
    if x.device.type != "cuda":
        raise RuntimeError("MiniMax H3 memory optimizations require CUDA.")
    try:
        from sageattention import sageattn_qk_int8_pv_fp8_cuda
    except ImportError as error:
        raise RuntimeError("MiniMax H3 memory optimizations require SageAttention.") from error

    token_count = x.shape[0]
    inner = self.heads * self.head_dim
    q, k, v = self.qkv_proj(x).split(inner, dim=-1)
    q = q.view(1, token_count, self.heads, self.head_dim).contiguous()
    k = k.view(1, token_count, self.heads, self.head_dim).contiguous()
    v = v.view(1, token_count, self.heads, self.head_dim).contiguous()
    if rope_freqs is not None:
        qw = comfy.model_management.cast_to(self.q_norm.weight, device=x.device)
        kw = comfy.model_management.cast_to(self.k_norm.weight, device=x.device)
        comfy.quant_ops.ck.rms_rope_split_half_(q, k, rope_freqs, qw, kw, epsilon=self.q_norm.eps, rot_dim=rope_freqs.shape[-3] * 2)
    else:
        q = self.q_norm(q.view(token_count, self.heads, self.head_dim)).unsqueeze(0).contiguous()
        k = self.k_norm(k.view(token_count, self.heads, self.head_dim)).unsqueeze(0).contiguous()
    attended = sageattn_qk_int8_pv_fp8_cuda(q, k, v, is_causal=False, tensor_layout="NHD", pv_accum_dtype="fp32+fp32")
    return self.out_proj(attended.to(x.dtype).flatten(2).squeeze(0))


def _h3_sparse_step(transformer_options: dict[str, Any], timestep: torch.Tensor, config: MiniMaxH3RadialAttentionConfig) -> bool:
    sigmas = transformer_options.get("sample_sigmas")
    if not isinstance(sigmas, torch.Tensor) or sigmas.ndim != 1 or len(sigmas) < 2:
        raise ValueError("MiniMax H3 Radial requires the sampler sigma schedule.")
    current = timestep.flatten()[0].to(sigmas.dtype)
    if current <= sigmas[-1]:
        return False
    step_index = torch.searchsorted(-sigmas, -current, right=True) - 1
    step_index = int(step_index.clamp(0, len(sigmas) - 2).item())
    return config.dense_start_steps <= step_index < len(sigmas) - 1 - config.dense_end_steps


def _h3_radial_block_mask(state: dict[str, Any], device: torch.device) -> torch.Tensor:
    block_size = state["config"].block_size
    total_blocks = math.ceil(state["sequence_length"] / block_size)
    plan = torch.ones((total_blocks, total_blocks), dtype=torch.bool, device=device)
    video_start, video_end = state["video_start"], state["video_end"]
    first_video_block = (video_start + block_size - 1) // block_size
    last_video_block = video_end // block_size - 1
    if last_video_block < first_video_block:
        return plan

    plan[first_video_block : last_video_block + 1, first_video_block : last_video_block + 1] = False
    frame_tokens = state["tokens_per_frame"]
    frame_count = state["frame_count"]

    def describe(block_index: int):
        start = max(video_start, block_index * block_size) - video_start
        end = min(video_end, (block_index + 1) * block_size) - video_start
        return min(frame_count - 1, start // frame_tokens), start % frame_tokens, (end - 1) % frame_tokens

    for output_block in range(first_video_block, last_video_block + 1):
        output_frame, output_left, output_right = describe(output_block)
        for input_block in range(first_video_block, last_video_block + 1):
            input_frame, input_left, input_right = describe(input_block)
            frame_distance = abs(output_frame - input_frame)
            if frame_distance == 0:
                plan[output_block, input_block] = True
                continue
            radius = frame_tokens // 2 if frame_distance == 1 else max(
                block_size,
                int((2 ** frame_tokens.bit_length()) * state["config"].decay_factor / 2**frame_distance),
            )
            plan[output_block, input_block] = output_left <= input_right + radius and input_left <= output_right + radius
    return plan


def _convert_sparse_mask(mask: torch.Tensor, block_size: int) -> torch.Tensor:
    compute_capability = "sm{}{}".format(*torch.cuda.get_device_capability(mask.device))
    if block_size == 128:
        expand_axis = 0 if compute_capability == "sm90" else 1
        return mask.repeat_interleave(2, dim=expand_axis)
    rows, columns = mask.shape
    if compute_capability == "sm90":
        return mask.reshape(rows, columns // 2, 2).any(dim=2)
    return mask.reshape(rows // 2, 2, columns).any(dim=1)


def _run_h3_radial_attention(self, x, rope_freqs, state: dict[str, Any]) -> torch.Tensor:
    if x.device.type != "cuda":
        raise RuntimeError("MiniMax H3 Radial requires CUDA.")
    try:
        from spas_sage_attn import block_sparse_sage2_attn_cuda
    except ImportError as error:
        raise RuntimeError("MiniMax H3 Radial requires spas_sage_attn.") from error

    sequence_length = x.shape[0]
    block_size = state["config"].block_size
    padded_length = math.ceil(sequence_length / block_size) * block_size
    q, k, v = self.qkv_proj(x).split(self.heads * self.head_dim, dim=-1)
    q = q.view(1, sequence_length, self.heads, self.head_dim)
    k = k.view(1, sequence_length, self.heads, self.head_dim)
    v = v.view(1, sequence_length, self.heads, self.head_dim)
    if rope_freqs is not None:
        qw = comfy.model_management.cast_to(self.q_norm.weight, device=x.device)
        kw = comfy.model_management.cast_to(self.k_norm.weight, device=x.device)
        comfy.quant_ops.ck.rms_rope_split_half_(q, k, rope_freqs, qw, kw, epsilon=self.q_norm.eps, rot_dim=rope_freqs.shape[-3] * 2)
    else:
        q = self.q_norm(q)
        k = self.k_norm(k)
    if padded_length != sequence_length:
        padding = (0, 0, 0, 0, 0, padded_length - sequence_length)
        q, k, v = (torch.nn.functional.pad(tensor, padding) for tensor in (q, k, v))
    mask = state.setdefault("masks", {}).get((x.device, padded_length))
    if mask is None:
        mask = _convert_sparse_mask(_h3_radial_block_mask(state, x.device), block_size)
        state["masks"][(x.device, padded_length)] = mask
    mask_id = mask.unsqueeze(0).unsqueeze(0).expand(1, self.heads, -1, -1).to(torch.int8)
    output = block_sparse_sage2_attn_cuda(q, k, v, mask_id=mask_id, tensor_layout="NHD", output_dtype=x.dtype)
    return self.out_proj(output[:, :sequence_length].to(x.dtype).reshape(sequence_length, self.heads * self.head_dim))


def _make_h3_radial_attention_forward(original: Callable[..., torch.Tensor], config: MiniMaxH3RadialAttentionConfig):
    def forward(self, x, rope_freqs=None, transformer_options=None):
        transformer_options = {} if transformer_options is None else transformer_options
        state = transformer_options.get(MINIMAX_H3_RADIAL_STATE_KEY)
        if state is None or not _h3_sparse_step(transformer_options, state["timestep"], config):
            return original(x, rope_freqs=rope_freqs, transformer_options=transformer_options)
        return _run_h3_radial_attention(self, x, rope_freqs, state)

    return forward


def _make_h3_radial_wrapper(config: MiniMaxH3RadialAttentionConfig):
    def wrapper(executor, x, timestep, context, transformer_options=None, minimax_payload=None, **kwargs):
        transformer_options = {} if transformer_options is None else transformer_options
        model = executor.class_obj
        video_x, audio_x = x
        video_x = comfy.ldm.common_dit.pad_to_patch_size(video_x, model.patch_size)
        latent_t, latent_h, latent_w = video_x.shape[2:5]
        payload = minimax_payload or {}
        layout = payload.get("layout")
        signature = (context.shape[1], latent_t, latent_h, latent_w, audio_x.shape[-1])
        if layout is None or layout.signature != signature:
            layout = minimax_model.PackedLayout(*signature, keyframes=payload.get("keyframes"), refs=payload.get("refs"))
        video_start, video_end, _ = next(segment for segment in layout.segments if segment[2] == "video")
        options = dict(transformer_options)
        options[MINIMAX_H3_RADIAL_STATE_KEY] = {
            "config": config,
            "timestep": timestep,
            "sequence_length": layout.seq_len,
            "video_start": video_start,
            "video_end": video_end,
            "frame_count": latent_t,
            "tokens_per_frame": (video_end - video_start) // latent_t,
            "masks": {},
        }
        return executor(x, timestep, context, options, minimax_payload=minimax_payload, **kwargs)

    return wrapper


def _patch_h3_radial_attention(model: Any, config: MiniMaxH3RadialAttentionConfig) -> Any:
    config.validate()
    patched = model.clone()
    diffusion_model = patched.get_model_object("diffusion_model")
    if not isinstance(diffusion_model, minimax_model.MiniMaxH3Model):
        raise ValueError("MiniMax H3 Radial requires a MiniMax H3 diffusion model.")
    attention_head_dims = {block.attn.head_dim for block in diffusion_model.blocks}
    if not attention_head_dims <= {64, 128}:
        raise ValueError("MiniMax H3 Radial requires a 64 or 128 dimension attention head.")
    options = _ensure_transformer_options(patched)
    options[UNIFIED_ATTENTION_OWNER_KEY] = "Sparse / MiniMax H3 Radial"
    patched.remove_wrappers_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, MINIMAX_H3_RADIAL_WRAPPER_KEY)
    patched.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, MINIMAX_H3_RADIAL_WRAPPER_KEY, _make_h3_radial_wrapper(config))
    for index, block in enumerate(diffusion_model.blocks[config.dense_blocks :], start=config.dense_blocks):
        original = block.attn.forward
        patched_forward = _make_h3_radial_attention_forward(original, config)
        if config.allow_compile:
            patched_forward = torch.compile(patched_forward, dynamic=True)
        patched_forward = types.MethodType(patched_forward, block.attn)
        patched.add_object_patch(f"diffusion_model.blocks.{index}.attn.forward", patched_forward)
    return patched


def _sla_block_ranges(ranges: tuple[tuple[int, int], ...], block_size: int, block_count: int) -> tuple[tuple[int, int], ...]:
    blocks = []
    for start, stop in ranges:
        first = max(0, int(start)) // block_size
        last = min(block_count, (max(0, int(stop)) + block_size - 1) // block_size)
        if first < last:
            blocks.append((first, last))
    return tuple(blocks)


def _sla_pool_tokens(value: torch.Tensor, block_size: int) -> torch.Tensor:
    batch, sequence, heads, head_dim = value.shape
    block_count = math.ceil(sequence / block_size)
    padded = block_count * block_size
    if padded != sequence:
        value = F.pad(value, (0, 0, 0, 0, 0, padded - sequence))
    pooled = value.transpose(1, 2).reshape(batch, heads, block_count, block_size, head_dim).sum(dim=3)
    counts = torch.arange(block_count, device=value.device).mul(block_size).add(block_size).clamp(max=sequence)
    counts = counts.sub(torch.arange(block_count, device=value.device).mul(block_size)).to(pooled.dtype)
    return pooled / counts.view(1, 1, block_count, 1)


def _sla_select_blocks(q: torch.Tensor, k: torch.Tensor, config: MiniMaxH3SlaAttentionConfig, protected_ranges: tuple[tuple[int, int], ...], reference_ranges: tuple[tuple[int, int], ...], previous: torch.Tensor | None) -> tuple[torch.Tensor, int, torch.Tensor | None]:
    pooled_q = _sla_pool_tokens(q, config.block_size).float()
    key_block = 64 if config.block_size == 128 else config.block_size
    pooled_k = _sla_pool_tokens(k, key_block) - k.mean(dim=1, dtype=torch.float32)[:, :, None, :]
    scores = pooled_q @ pooled_k.transpose(-1, -2)
    key_blocks = scores.shape[-1]
    base_count = max(1, min(key_blocks, int((1.0 - config.sparsity) * key_blocks)))
    forced = torch.zeros(key_blocks, dtype=torch.bool, device=scores.device)
    for first, last in _sla_block_ranges(protected_ranges, key_block, key_blocks):
        forced[first:last] = True
    if config.stabilize_routing and previous is not None and previous.shape[:3] == scores.shape[:3]:
        scale = scores.detach().abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
        scores.scatter_add_(-1, previous.long(), scale.expand_as(previous) * 0.01)
    reference_count = 0
    reference_sparsity = {
        "Off": None,
        "Light": SLA_REFERENCE_LIGHT_SPARSITY,
        "Heavy Enforcement": 0.0,
    }[config.protect_reference_media]
    if reference_sparsity is not None:
        for first, last in _sla_block_ranges(reference_ranges, key_block, key_blocks):
            keep = max(1, min(last - first, math.ceil((1.0 - reference_sparsity) * (last - first))))
            selected_reference = torch.topk(scores[..., first:last], keep, dim=-1, sorted=False).indices + first
            scores.scatter_(-1, selected_reference, float("inf"))
            reference_count += keep
    if forced.any():
        scores[..., forced] = float("inf")
    selected_count = min(key_blocks, base_count + int(forced.sum().item()) + reference_count)
    selected = torch.topk(scores, selected_count, dim=-1, sorted=False).indices.to(torch.int32).contiguous()
    history = selected[..., : min(4, selected_count)].contiguous() if config.stabilize_routing else None
    return selected, selected_count, history


SLA_TRITON_LAUNCH_LADDER = {
    (32, 32): ((4, 2), (2, 1), (4, 1), (2, 2)),
    (64, 64): ((4, 1), (4, 3), (8, 3), (8, 1)),
    (64, 128): ((4, 2), (8, 2), (4, 1)),
    (128, 64): ((8, 3), (4, 3), (8, 2), (4, 1)),
    (128, 128): ((8, 2), (4, 2), (8, 1), (4, 1)),
}
_SLA_TRITON_CHOSEN: dict[tuple[int, int, int], tuple[int, int]] = {}


if triton is not None:
    @triton.jit
    def _sla_attention_kernel(Q, K, V, scale: tl.constexpr, topk: tl.constexpr, lookup, output, heads: tl.constexpr, query_length: tl.constexpr, key_length: tl.constexpr, query_blocks: tl.constexpr, head_dim: tl.constexpr, query_block: tl.constexpr, key_block: tl.constexpr):
        query_block_index = tl.program_id(0).to(tl.int64)
        batch_head_index = tl.program_id(1).to(tl.int64)
        batch_index = batch_head_index // heads
        head_index = batch_head_index % heads
        hidden = heads * head_dim
        query_offset = batch_index * query_length * hidden + head_index * head_dim
        key_offset = batch_index * key_length * hidden + head_index * head_dim
        lookup_offset = (batch_head_index * query_blocks + query_block_index) * topk
        row = query_block_index * query_block + tl.arange(0, query_block)
        column = tl.arange(0, key_block)
        dimension = tl.arange(0, head_dim)
        query_ptr = Q + query_offset + row[:, None] * hidden + dimension[None, :]
        output_ptr = output + query_offset + row[:, None] * hidden + dimension[None, :]
        running_max = tl.full([query_block], -float("inf"), tl.float32)
        running_sum = tl.zeros([query_block], tl.float32)
        running_output = tl.zeros([query_block, head_dim], tl.float32)
        query = tl.load(query_ptr, mask=row[:, None] < query_length, other=0.0)
        for selection_index in tl.range(topk):
            key_block_index = tl.load(lookup + lookup_offset + selection_index).to(tl.int64)
            key_start = key_block_index * key_block
            key_valid = key_start + column < key_length
            key_ptr = K + key_offset + (key_start + column)[None, :] * hidden + dimension[:, None]
            value_ptr = V + key_offset + (key_start + column)[:, None] * hidden + dimension[None, :]
            key = tl.load(key_ptr, mask=key_valid[None, :], other=0.0)
            logits = tl.dot(query, key) * (scale * 1.4426950408889634)
            logits = tl.where(key_valid[None, :], logits, -float("inf"))
            value = tl.load(value_ptr, mask=key_valid[:, None], other=0.0)
            local_max = tl.max(logits, 1)
            next_max = tl.maximum(running_max, local_max)
            probabilities = tl.math.exp2(logits - next_max[:, None])
            alpha = tl.math.exp2(running_max - next_max)
            running_output = running_output * alpha[:, None] + tl.dot(probabilities.to(value.dtype), value)
            running_sum = running_sum * alpha + tl.sum(probabilities, 1)
            running_max = next_max
        tl.store(output_ptr, (running_output / running_sum[:, None]).to(output.type.element_ty), mask=row[:, None] < query_length)


def _sla_sparse_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, lookup: torch.Tensor, selected_count: int, config: MiniMaxH3SlaAttentionConfig) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("MiniMax H3 SLA requires Triton.")
    batch, sequence, heads, head_dim = q.shape
    key_block = 64 if config.block_size == 128 else config.block_size
    query_blocks = triton.cdiv(sequence, config.block_size)
    output = torch.empty_like(q)
    key = (config.block_size, key_block, head_dim)
    ladder = (_SLA_TRITON_CHOSEN[key],) if key in _SLA_TRITON_CHOSEN else SLA_TRITON_LAUNCH_LADDER[(config.block_size, key_block)]
    last_error = None
    for num_warps, num_stages in ladder:
        try:
            _sla_attention_kernel[(query_blocks, batch * heads)](
                q, k, v, head_dim**-0.5, selected_count, lookup, output,
                heads, sequence, k.shape[1], query_blocks, head_dim,
                config.block_size, key_block, num_warps=num_warps, num_stages=num_stages,
            )
        except triton.runtime.errors.OutOfResources as error:
            last_error = error
            continue
        _SLA_TRITON_CHOSEN[key] = (num_warps, num_stages)
        return output
    raise last_error if last_error is not None else RuntimeError("MiniMax H3 SLA found no viable Triton launch configuration.")


def _sla_sampler_step_index(transformer_options: dict[str, Any], timestep: torch.Tensor) -> tuple[int, int] | None:
    sigmas = transformer_options.get("sample_sigmas")
    if not isinstance(sigmas, torch.Tensor) or sigmas.ndim != 1 or len(sigmas) < 2:
        return None
    current = timestep.flatten()[0].to(sigmas.dtype)
    index = int((torch.searchsorted(-sigmas, -current, right=True) - 1).clamp(0, len(sigmas) - 2).item())
    return index, len(sigmas) - 1


def _sla_dense_step(transformer_options: dict[str, Any], timestep: torch.Tensor, config: MiniMaxH3SlaAttentionConfig) -> str | None:
    step = _sla_sampler_step_index(transformer_options, timestep)
    if step is None:
        if config.dense_tail_steps > 0 or _parse_h3_sla_dense_steps(config.dense_steps):
            return "sampling schedule is unavailable"
        return None
    index, total_steps = step
    if index in _parse_h3_sla_dense_steps(config.dense_steps):
        return f"sampling step {index} is configured dense"
    if config.dense_tail_steps > 0 and index >= total_steps - config.dense_tail_steps:
        return f"one of the final {config.dense_tail_steps} sampler steps is configured dense"
    return None


def _report_sla_fallback(state: dict[str, Any], reason: str, level: int = logging.INFO) -> None:
    reported = state.setdefault("reported_fallbacks", set())
    if reason in reported:
        return
    reported.add(reason)
    logging.getLogger(__name__).log(level, "MiniMax H3 SLA using dense attention: %s", reason)


def _run_h3_sla_dense_attention(state: dict[str, Any], original: Callable[..., torch.Tensor], *args: Any, **kwargs: Any) -> torch.Tensor:
    backend = state.get("dense_backend")
    if backend is None:
        return original(*args, **kwargs)
    return backend(original, *args, **kwargs)


def _make_h3_sla_override(config: MiniMaxH3SlaAttentionConfig, state: dict[str, Any]) -> Callable[..., torch.Tensor]:
    def override(original, q, k, v, heads, mask=None, skip_reshape=False, skip_output_reshape=False, **kwargs):
        options = kwargs.get("transformer_options") or {}
        current = options.get(MINIMAX_H3_SLA_STATE_KEY)
        reason = None
        if state.get("sparse_disabled_reason") is not None:
            reason = state["sparse_disabled_reason"]
        elif current is None:
            reason = "SLA step state is unavailable"
        elif current.get("dense", True):
            reason = current.get("dense_reason", "this sampler step is configured dense")
        elif mask is not None:
            reason = "the attention call has a mask"
        elif not skip_reshape or q.ndim != 4 or q.shape[-1] != 128:
            reason = "the attention call is not MiniMax H3 packed self-attention"
        elif q.dtype not in (torch.float16, torch.bfloat16):
            reason = f"the attention dtype is {q.dtype}"
        elif q.shape[2] < config.minimum_sequence_length:
            reason = f"sequence length {q.shape[2]} is below {config.minimum_sequence_length}"
        elif q.device.type != "cuda":
            reason = "the attention call is not on CUDA"
        if reason is not None:
            _report_sla_fallback(state, reason)
            return _run_h3_sla_dense_attention(state, original, q, k, v, heads, mask=mask, skip_reshape=skip_reshape, skip_output_reshape=skip_output_reshape, **kwargs)
        try:
            q_nhd, k_nhd, v_nhd = (value.transpose(1, 2).contiguous() for value in (q, k, v))
            call_index = current["call_index"]
            current["call_index"] = call_index + 1
            lookup, selected_count, history = _sla_select_blocks(
                q_nhd, k_nhd, config, current["protected_ranges"], current["reference_ranges"],
                state["history"].get(call_index),
            )
            output = _sla_sparse_attention(q_nhd, k_nhd, v_nhd, lookup, selected_count, config)
            if history is not None:
                state["history"][call_index] = history
            return output.transpose(1, 2) if skip_output_reshape else output.reshape(q.shape[0], q.shape[2], -1)
        except Exception as error:
            state["history"].clear()
            state["sparse_disabled_reason"] = f"the sparse kernel failed with {error.__class__.__name__}: {error}"
            _report_sla_fallback(state, state["sparse_disabled_reason"], logging.WARNING)
            return _run_h3_sla_dense_attention(state, original, q, k, v, heads, mask=mask, skip_reshape=skip_reshape, skip_output_reshape=skip_output_reshape, **kwargs)
    return override


def _set_h3_sla_reduced_precision_accumulation(enabled: bool) -> dict[str, bool]:
    backend = torch.backends.cuda.matmul
    previous = {}
    for name in ("allow_fp16_reduced_precision_reduction", "allow_bf16_reduced_precision_reduction"):
        if hasattr(backend, name):
            previous[name] = getattr(backend, name)
            setattr(backend, name, enabled)
    return previous


def _restore_h3_sla_reduced_precision_accumulation(previous: dict[str, bool]) -> None:
    backend = torch.backends.cuda.matmul
    for name, value in previous.items():
        setattr(backend, name, value)


class MiniMaxH3SlaSamplingScope:
    def __init__(self, state: dict[str, Any], config: MiniMaxH3SlaAttentionConfig) -> None:
        self.state = state
        self.config = config

    def __call__(self, sample_fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        self.state["history"].clear()
        self.state["reported_fallbacks"].clear()
        self.state["sparse_disabled_reason"] = None
        previous = _set_h3_sla_reduced_precision_accumulation(False) if self.config.disable_fp16_accumulation else {}
        try:
            return sample_fn(*args, **kwargs)
        finally:
            self.state["history"].clear()
            _restore_h3_sla_reduced_precision_accumulation(previous)


def _make_h3_sla_wrapper(config: MiniMaxH3SlaAttentionConfig, state: dict[str, Any]):
    def wrapper(executor, x, timestep, context, transformer_options=None, minimax_payload=None, **kwargs):
        options = dict(transformer_options or {})
        layout = (minimax_payload or {}).get("layout")
        protected_ranges = []
        reference_ranges = []
        for start, stop, kind in getattr(layout, "segments", ()) or ():
            if config.protect_audio and kind in ("text", "audio", "cond_audio", "ref_audio"):
                protected_ranges.append((int(start), int(stop)))
            elif kind in ("cond", "ref_img", "ref_video", "video_ref"):
                reference_ranges.append((int(start), int(stop)))
        dense_reason = state.get("sparse_disabled_reason")
        if dense_reason is None and layout is None:
            dense_reason = "MiniMax H3 layout metadata is unavailable"
        elif dense_reason is None:
            dense_reason = _sla_dense_step(options, timestep, config)
        if config.disable_fp16_accumulation:
            _set_h3_sla_reduced_precision_accumulation(False)
        current = {
            "dense": dense_reason is not None,
            "dense_reason": dense_reason,
            "call_index": 0,
            "protected_ranges": tuple(protected_ranges),
            "reference_ranges": tuple(reference_ranges),
        }
        if dense_reason is not None:
            state["history"].clear()
            _report_sla_fallback(state, dense_reason)
        options[MINIMAX_H3_SLA_STATE_KEY] = current
        if minimax_payload is not None:
            kwargs["minimax_payload"] = minimax_payload
        try:
            return executor(x, timestep, context, options, **kwargs)
        except Exception:
            state["history"].clear()
            raise
    return wrapper


def _patch_h3_sla_attention(model: Any, config: MiniMaxH3SlaAttentionConfig) -> Any:
    config.validate()
    if triton is None:
        raise RuntimeError("MiniMax H3 SLA requires Triton.")
    patched = model.clone()
    diffusion_model = patched.get_model_object("diffusion_model")
    if not isinstance(diffusion_model, minimax_model.MiniMaxH3Model):
        raise ValueError("MiniMax H3 SLA requires a MiniMax H3 diffusion model.")
    if {block.attn.head_dim for block in diffusion_model.blocks} != {128}:
        raise ValueError("MiniMax H3 SLA requires 128-dimensional attention heads.")
    state = {
        "history": {},
        "reported_fallbacks": set(),
        "sparse_disabled_reason": None,
        "dense_backend": _resolve_h3_sla_dense_backend(config.dense_backend),
    }
    options = _ensure_transformer_options(patched)
    options[UNIFIED_ATTENTION_OWNER_KEY] = "Sparse / MiniMax H3 SLA"
    options["optimized_attention_override"] = _call_attention_with_fallback(_make_h3_sla_override(config, state))
    patched.remove_wrappers_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, MINIMAX_H3_SLA_WRAPPER_KEY)
    patched.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, MINIMAX_H3_SLA_WRAPPER_KEY, _make_h3_sla_wrapper(config, state))
    patched.remove_wrappers_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, MINIMAX_H3_SLA_SAMPLING_SCOPE_KEY)
    patched.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, MINIMAX_H3_SLA_SAMPLING_SCOPE_KEY, MiniMaxH3SlaSamplingScope(state, config))
    logging.getLogger(__name__).info(
        "MiniMax H3 SLA installed: sparsity=%.2f, block_size=%d, minimum_sequence_length=%d, dense_steps=%s, dense_tail_steps=%d, protect_audio=%s, protect_reference_media=%s, dense_backend=%s, disable_fp16_accumulation=%s, stabilize_routing=%s",
        config.sparsity,
        config.block_size,
        config.minimum_sequence_length,
        config.dense_steps,
        config.dense_tail_steps,
        config.protect_audio,
        config.protect_reference_media,
        config.dense_backend,
        config.disable_fp16_accumulation,
        config.stabilize_routing,
    )
    return patched


def patch_unified_attention_model(model: Any, attention_mode: dict[str, Any]) -> Any:
    mode = attention_mode.get("attention_mode")
    if mode not in UNIFIED_ATTENTION_MODES:
        raise ValueError("Choose a supported attention mode.")
    if mode == "disabled":
        return model
    if mode == "Sparse / MiniMax H3 Radial":
        config = attention_mode.get("minimax_h3_radial_config")
        if not isinstance(config, MiniMaxH3RadialAttentionConfig):
            raise ValueError("Sparse / MiniMax H3 Radial requires MiniMax H3 Radial Attention Config.")
        return _patch_h3_radial_attention(model, config)
    if mode == "Sparse / MiniMax H3 SLA":
        config = attention_mode.get("minimax_h3_sla_config")
        if config is None:
            config = MiniMaxH3SlaAttentionConfig()
        if not isinstance(config, MiniMaxH3SlaAttentionConfig):
            raise ValueError("Sparse / MiniMax H3 SLA requires MiniMax H3 SLA Attention Config.")
        config = replace(
            config,
            sparsity=attention_mode.get("sparsity", 0.9),
            block_size=attention_mode.get("block_size", 32),
            dense_tail_steps=attention_mode.get("dense_tail_steps", 1),
            protect_reference_media=attention_mode.get("protect_reference_media", "Light"),
            dense_backend=attention_mode.get("dense_backend", "comfy_kitchen"),
        )
        return _patch_h3_sla_attention(model, config)

    patched = model.clone()
    options = _ensure_transformer_options(patched)
    options[UNIFIED_ATTENTION_OWNER_KEY] = mode
    if mode == "FlashAttention":
        options["optimized_attention_override"] = _call_attention_function(
            _make_flash_backend(bool(attention_mode.get("allow_compile", False)), _model_compute_dtype(patched))
        )
    elif mode == "SageAttention":
        h3_memory_optimizations = bool(attention_mode.get("h3_memory_optimizations", False))
        options["optimized_attention_override"] = _call_attention_function(
            _make_sage_backend(attention_mode.get("sage_mode", "auto"), bool(attention_mode.get("allow_compile", False)))
        )
        if h3_memory_optimizations:
            diffusion_model = patched.get_model_object("diffusion_model")
            if not isinstance(diffusion_model, minimax_model.MiniMaxH3Model):
                raise ValueError("MiniMax H3 memory optimizations require a MiniMax H3 diffusion model.")
            for index, block in enumerate(diffusion_model.blocks):
                patched_forward = types.MethodType(_h3_sage_forward, block.attn)
                patched.add_object_patch(f"diffusion_model.blocks.{index}.attn.forward", patched_forward)
    return patched


def load_ideogram4_debanner_directions() -> dict[int, torch.Tensor]:
    if not IDEOGRAM4_DEBANNER_BUNDLE.is_file():
        raise FileNotFoundError(f"Ideogram 4 debanner bundle is missing: {IDEOGRAM4_DEBANNER_BUNDLE}")
    with safe_open(str(IDEOGRAM4_DEBANNER_BUNDLE), framework="pt", device="cpu") as handle:
        return {index: handle.get_tensor(f"direction.block_{index:02d}") for index in range(25, 29)}


def _validate_ideogram4_debanner_directions(directions: dict[int, torch.Tensor]) -> None:
    for index in range(25, 29):
        direction = directions.get(index)
        if not torch.is_tensor(direction) or tuple(direction.shape) != (1, 8, 8, 4608):
            raise ValueError(f"Ideogram 4 debanner direction for block {index} must have shape [1, 8, 8, 4608].")


def _make_ideogram4_debanner_wrapper(strength: float):
    def wrapper(executor, x, timesteps, context=None, attention_mask=None, transformer_options=None, **kwargs):
        _, _, grid_height, grid_width = x.shape
        options = {} if transformer_options is None else dict(transformer_options)
        options[IDEOGRAM4_DEBANNER_STATE_KEY] = {
            "grid_height": grid_height,
            "grid_width": grid_width,
            "image_token_count": grid_height * grid_width,
            "strength": strength,
            "is_conditional": context is not None,
        }
        return executor(x, timesteps, context, attention_mask, options, **kwargs)

    return wrapper


def _make_ideogram4_debanner_forward(original: Callable[..., torch.Tensor], direction: torch.Tensor):
    def forward(self, x, attn_mask, freqs_cis, adaln_input, transformer_options={}):
        output = original(x, attn_mask, freqs_cis, adaln_input, transformer_options=transformer_options)
        state = transformer_options.get(IDEOGRAM4_DEBANNER_STATE_KEY)
        if state is None or not state["is_conditional"]:
            return output

        grid_height = state["grid_height"]
        grid_width = state["grid_width"]
        image_token_count = state["image_token_count"]
        image_offset = output.shape[1] - image_token_count
        spatial_direction = torch.nn.functional.interpolate(
            direction.permute(0, 3, 1, 2), size=(grid_height, grid_width), mode="nearest"
        ).permute(0, 2, 3, 1).reshape(1, image_token_count, -1).to(device=output.device, dtype=output.dtype)
        image_tokens = output[:, image_offset:]
        original_norm = torch.linalg.vector_norm(image_tokens, dim=-1, keepdim=True)
        corrected = torch.nn.functional.normalize(image_tokens - state["strength"] * spatial_direction, p=2, dim=-1) * original_norm
        return torch.cat((output[:, :image_offset], corrected), dim=1)

    return forward


def patch_ideogram4_debanner_model(model: Any, directions: dict[int, torch.Tensor], strength: float) -> Any:
    strength = float(strength)
    if strength < 0.0:
        raise ValueError("Ideogram 4 debanner strength must be nonnegative.")
    _validate_ideogram4_debanner_directions(directions)

    diffusion_model = model.get_model_object("diffusion_model")
    if not isinstance(diffusion_model, ideogram4_model.Ideogram4Transformer2DModel):
        raise ValueError("Ideogram 4 Debanner requires an Ideogram4Transformer2DModel diffusion model.")

    patched = model.clone()
    patched.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
        IDEOGRAM4_DEBANNER_WRAPPER_KEY,
        _make_ideogram4_debanner_wrapper(strength),
    )
    for index in range(25, 29):
        block = diffusion_model.layers[index]
        patched_forward = types.MethodType(_make_ideogram4_debanner_forward(block.forward, directions[index]), block)
        patched.add_object_patch(f"diffusion_model.layers.{index}.forward", patched_forward)
    return patched


def patch_ideogram4_debanner(model: Any, strength: float) -> Any:
    return patch_ideogram4_debanner_model(model, load_ideogram4_debanner_directions(), strength)
