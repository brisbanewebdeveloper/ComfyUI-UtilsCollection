import pytest
import torch
import pathlib
import sys
import types

CUSTOM_NODE_ROOT = pathlib.Path(__file__).parents[1]
PACKAGE_NAME = "utils_collection_video_helpers_test"
package = types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(CUSTOM_NODE_ROOT)]
sys.modules.setdefault(PACKAGE_NAME, package)

from utils_collection_video_helpers_test.helpers.video_helpers import (
    VideoCompareType,
    compute_video_signatures,
    cross_correlate_video_signatures,
    find_video_overlap,
)


@pytest.mark.parametrize("compare_type", list(VideoCompareType))
def test_video_signatures_support_all_reference_comparison_modes(compare_type):
    frames = torch.rand(6, 32, 48, 3, generator=torch.Generator().manual_seed(3))
    signatures = compute_video_signatures(frames, compare_type)
    assert signatures.shape[0] == frames.shape[0]
    assert signatures.ndim == 2


def test_video_cross_correlation_reports_offset_and_runner_up():
    frames = torch.rand(8, 16, 16, 3, generator=torch.Generator().manual_seed(4))
    signatures = compute_video_signatures(frames, VideoCompareType.SAD)
    result = cross_correlate_video_signatures(signatures, signatures)
    assert result.offset_frames == 0
    assert result.distance == 0.0
    assert result.second_best_distance is not None
    assert result.overlap_frames == frames.shape[0]


def test_video_overlap_finds_suffix_prefix_without_single_frame_match():
    previous = torch.rand(8, 16, 16, 3, generator=torch.Generator().manual_seed(5))
    current = torch.cat(
        (previous[-4:], torch.rand(3, 16, 16, 3, generator=torch.Generator().manual_seed(6))),
        dim=0,
    )
    result = find_video_overlap(previous, current, maximum_overlap_frames=6, compare_type="sad")
    assert result.overlap_frames == 4
    assert result.distance == 0.0
