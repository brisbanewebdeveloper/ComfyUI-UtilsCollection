"""Native video-sequence alignment helpers.

This module adapts the complete comparison/correlation portion of the
``video-offset-finder`` reference implementation to ComfyUI BHWC tensors.
It intentionally has no file-decoder or third-party hashing dependency.
"""

from dataclasses import dataclass
from enum import Enum
import math
import os

import torch
import torch.nn.functional as F


VIDEO_EXTENSIONS = {
    ".avi", ".flv", ".m2ts", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg",
    ".mpg", ".mts", ".ts", ".webm", ".wmv",
}


def normalize_video_path(path: str) -> str:
    if not isinstance(path, str):
        return ""
    path = path.strip().replace("\\", "/")
    return os.path.abspath(os.path.normpath(path)) if path else ""


def list_video_files(directory: str) -> list[str]:
    return sorted(
        os.path.join(directory, name)
        for name in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, name))
        and os.path.splitext(name)[1].lower() in VIDEO_EXTENSIONS
    )


class VideoCompareType(str, Enum):
    PHASH = "phash"
    DHASH = "dhash"
    AHASH = "ahash"
    WHASH = "whash"
    SAD = "sad"


@dataclass(frozen=True)
class VideoCorrelationResult:
    offset_frames: int
    distance: float
    second_best_distance: float | None
    overlap_frames: int


def _validate_frames(frames: torch.Tensor) -> torch.Tensor:
    if (
        not torch.is_tensor(frames)
        or frames.ndim != 4
        or frames.shape[0] < 1
        or min(frames.shape[1:3]) < 1
        or frames.shape[3] < 3
        or not torch.isfinite(frames).all()
    ):
        raise ValueError("Video alignment frames must be a finite BHWC RGB tensor.")
    return frames[..., :3].to(torch.float32).clamp(0, 1)


def _gray(frames: torch.Tensor, height: int, width: int) -> torch.Tensor:
    samples = F.interpolate(
        frames.movedim(-1, 1), size=(height, width), mode="area"
    )
    return (
        0.2126 * samples[:, :1]
        + 0.7152 * samples[:, 1:2]
        + 0.0722 * samples[:, 2:3]
    ).squeeze(1)


def _dct_basis(size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    index = torch.arange(size, device=device, dtype=dtype)
    basis = torch.cos(math.pi / size * (index[:, None] + 0.5) * index[None, :])
    basis[0] *= math.sqrt(1.0 / size)
    basis[1:] *= math.sqrt(2.0 / size)
    return basis


def _phash(frames: torch.Tensor, hash_size: int) -> torch.Tensor:
    side = hash_size * 4
    pixels = _gray(frames, side, side)
    basis = _dct_basis(side, pixels.device, pixels.dtype)
    coefficients = basis @ pixels @ basis.t()
    low = coefficients[:, :hash_size, :hash_size]
    median = low.flatten(1).median(dim=1, keepdim=True).values
    return (low.flatten(1) > median).to(torch.uint8)


def _dhash(frames: torch.Tensor, hash_size: int) -> torch.Tensor:
    pixels = _gray(frames, hash_size, hash_size + 1)
    return (pixels[..., 1:] > pixels[..., :-1]).flatten(1).to(torch.uint8)


def _ahash(frames: torch.Tensor, hash_size: int) -> torch.Tensor:
    pixels = _gray(frames, hash_size, hash_size)
    return (pixels > pixels.flatten(1).mean(dim=1, keepdim=True).view(-1, 1, 1)).flatten(1).to(torch.uint8)


def _whash(frames: torch.Tensor, hash_size: int) -> torch.Tensor:
    size = hash_size * 2
    pixels = _gray(frames, size, size)
    while pixels.shape[-1] > hash_size:
        pixels = (
            pixels[..., 0::2, 0::2]
            + pixels[..., 1::2, 0::2]
            + pixels[..., 0::2, 1::2]
            + pixels[..., 1::2, 1::2]
        ) * 0.25
    return (pixels > pixels.flatten(1).mean(dim=1, keepdim=True).view(-1, 1, 1)).flatten(1).to(torch.uint8)


def _sad(frames: torch.Tensor, width: int = 64, height: int = 64) -> torch.Tensor:
    return (_gray(frames, height, width) * 255.0).round().to(torch.int16).flatten(1)


def compute_video_signatures(
    frames: torch.Tensor,
    compare_type: VideoCompareType | str = VideoCompareType.PHASH,
    hash_size: int = 16,
) -> torch.Tensor:
    """Compute one native signature row per BHWC video frame."""
    frames = _validate_frames(frames)
    compare_type = VideoCompareType(compare_type)
    if isinstance(hash_size, bool) or not isinstance(hash_size, int) or hash_size < 2:
        raise ValueError("Video alignment hash_size must be an integer of at least 2.")
    if compare_type == VideoCompareType.PHASH:
        return _phash(frames, hash_size)
    if compare_type == VideoCompareType.DHASH:
        return _dhash(frames, hash_size)
    if compare_type == VideoCompareType.AHASH:
        return _ahash(frames, hash_size)
    if compare_type == VideoCompareType.WHASH:
        return _whash(frames, hash_size)
    return _sad(frames)


def _candidate_offsets(
    reference_count: int,
    query_count: int,
    min_overlap_fraction: float,
    min_overlap_frames: int,
    min_offset_frames: int | None,
    max_offset_frames: int | None,
) -> tuple[range, int]:
    if reference_count < 1 or query_count < 1:
        raise ValueError("Cannot correlate empty video signatures.")
    shortest = min(reference_count, query_count)
    required = min(shortest, max(min_overlap_frames, math.ceil(shortest * min_overlap_fraction)))
    lower_bound = -query_count + required if min_offset_frames is None else min_offset_frames
    upper_bound = reference_count - required if max_offset_frames is None else max_offset_frames
    lower = max(-query_count + required, lower_bound)
    upper = min(reference_count - required, upper_bound)
    if lower > upper:
        raise ValueError("No video offsets satisfy the overlap bounds.")
    return range(lower, upper + 1), required


def cross_correlate_video_signatures(
    reference: torch.Tensor,
    query: torch.Tensor,
    compare_type: VideoCompareType | str = VideoCompareType.PHASH,
    min_overlap_fraction: float = 0.5,
    min_overlap_frames: int = 2,
    min_offset_frames: int | None = None,
    max_offset_frames: int | None = None,
) -> VideoCorrelationResult:
    """Find the globally best temporal offset and report ambiguity."""
    compare_type = VideoCompareType(compare_type)
    reference = reference.to(torch.uint8) if compare_type != VideoCompareType.SAD else reference.to(torch.int16)
    query = query.to(reference.dtype)
    if not 0 < min_overlap_fraction <= 1:
        raise ValueError("min_overlap_fraction must be in the interval (0, 1].")
    offsets, _ = _candidate_offsets(
        reference.shape[0], query.shape[0], min_overlap_fraction,
        min_overlap_frames, min_offset_frames, max_offset_frames,
    )
    candidates: list[tuple[float, int, int]] = []
    for offset in offsets:
        reference_start = max(offset, 0)
        query_start = max(-offset, 0)
        overlap = min(reference.shape[0] - reference_start, query.shape[0] - query_start)
        if overlap < 1:
            continue
        first = reference[reference_start:reference_start + overlap]
        second = query[query_start:query_start + overlap]
        if compare_type == VideoCompareType.SAD:
            distance = (first.to(torch.float32) - second.to(torch.float32)).abs().mean(dim=1)
        else:
            distance = (first != second).to(torch.float32).mean(dim=1)
        candidates.append((float(distance.mean().item()), offset, overlap))
    if not candidates:
        raise ValueError("No valid video alignment candidates were found.")
    candidates.sort(key=lambda item: (item[0], -item[2], abs(item[1])))
    best_distance, best_offset, best_overlap = candidates[0]
    return VideoCorrelationResult(
        best_offset,
        best_distance,
        candidates[1][0] if len(candidates) > 1 else None,
        best_overlap,
    )


def find_video_overlap(
    previous: torch.Tensor,
    current: torch.Tensor,
    maximum_overlap_frames: int,
    compare_type: VideoCompareType | str = VideoCompareType.PHASH,
    hash_size: int = 16,
) -> VideoCorrelationResult:
    """Find the best previous-tail/current-head overlap.

    Unlike whole-video offset search, a boundary join has an explicit
    suffix/prefix relationship. Every admissible overlap is therefore scored
    independently, while the result still reports the runner-up distance for
    ambiguity detection.
    """
    previous = _validate_frames(previous)
    current = _validate_frames(current)
    maximum_overlap_frames = min(maximum_overlap_frames, previous.shape[0], current.shape[0] - 1)
    if maximum_overlap_frames < 1:
        return VideoCorrelationResult(0, float("inf"), None, 0)
    previous_signatures = compute_video_signatures(previous, compare_type, hash_size)
    current_signatures = compute_video_signatures(current, compare_type, hash_size)
    candidates: list[tuple[float, int]] = []
    for overlap in range(maximum_overlap_frames, 0, -1):
        first = previous_signatures[-overlap:]
        second = current_signatures[:overlap]
        if compare_type == VideoCompareType.SAD:
            distance = (first.to(torch.float32) - second.to(torch.float32)).abs().mean().item() / 255.0
        else:
            distance = (first != second).to(torch.float32).mean().item()
        candidates.append((float(distance), overlap))
    candidates.sort(key=lambda item: (item[0], -item[1]))
    best_distance, best_overlap = candidates[0]
    return VideoCorrelationResult(
        best_overlap,
        best_distance,
        candidates[1][0] if len(candidates) > 1 else None,
        best_overlap,
    )


def audio_overlap_similarity(
    previous: torch.Tensor,
    current: torch.Tensor,
    overlap_frames: int,
    fps: int,
    sample_rate: int,
) -> float | None:
    """Compare suffix/prefix audio over a video-frame duration."""
    sample_count = round(overlap_frames * sample_rate / fps)
    if sample_count < max(256, sample_rate // 10):
        return None
    first = previous[:, -sample_count:].to(torch.float32)
    second = current[:, :sample_count].to(torch.float32)
    first = first - first.mean(dim=-1, keepdim=True)
    second = second - second.mean(dim=-1, keepdim=True)
    if first.norm() < 1e-5 or second.norm() < 1e-5:
        return None
    correlation = F.cosine_similarity(first.reshape(1, -1), second.reshape(1, -1)).item()
    return max(0.0, min(1.0, (correlation + 1.0) * 0.5))
