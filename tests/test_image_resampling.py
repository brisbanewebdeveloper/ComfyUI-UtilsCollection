import pathlib
import sys
import types

import pytest
import torch


CUSTOM_NODE_ROOT = pathlib.Path(__file__).parents[1]
PACKAGE_NAME = "utils_collection_image_resampling_test"
package = types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(CUSTOM_NODE_ROOT)]
sys.modules.setdefault(PACKAGE_NAME, package)

from utils_collection_image_resampling_test.helpers.image_helpers import (
    downscale_nohalo_lohalo,
    halo_downscale_dimensions,
)
from utils_collection_image_resampling_test.helpers import image_helpers
from utils_collection_image_resampling_test.nodes.image_nodes import UC_NoHaloLoHaloDownscale


def test_nohalo_lohalo_schema_contract():
    schema = UC_NoHaloLoHaloDownscale.define_schema()
    assert schema.node_id == "UC_NoHaloLoHaloDownscale"
    assert [input_.id for input_ in schema.inputs] == ["image", "method", "megapixels", "multiple"]
    assert schema.inputs[1].options == ["lohalo", "nohalo"]


def test_megapixel_dimensions_preserve_scale_and_alignment():
    width, height, scale, offset_x, offset_y = halo_downscale_dimensions(2048, 1024, 0.5, 16)
    assert (width, height) == (1024, 512)
    assert scale == pytest.approx(0.5)
    assert offset_x == pytest.approx(0.0)
    assert offset_y == pytest.approx(0.0)


def test_alignment_uses_uniform_cover_scale_and_center_crop():
    width, height, scale, offset_x, offset_y = halo_downscale_dimensions(1000, 777, 0.25, 64)
    assert width % 64 == 0
    assert height % 64 == 0
    assert scale == pytest.approx(max(width / 1000, height / 777))
    assert offset_x == pytest.approx((1000 - width / scale) * 0.5)
    assert offset_y == pytest.approx((777 - height / scale) * 0.5)


def test_downscaler_rejects_enlargement():
    with pytest.raises(ValueError, match="requires an output smaller"):
        halo_downscale_dimensions(64, 64, 1.0, 16)


@pytest.mark.parametrize("method", ["lohalo", "nohalo"])
def test_constant_image_is_preserved(method):
    image = torch.full((2, 128, 192, 3), 0.375)
    result = downscale_nohalo_lohalo(image, method, 0.01, 16)
    assert result.shape == (2, 80, 128, 3)
    torch.testing.assert_close(result, torch.full_like(result, 0.375), atol=2e-5, rtol=2e-5)


def test_lohalo_preserves_rgba_dtype_and_linear_alpha():
    image = torch.zeros((1, 128, 192, 4), dtype=torch.float16)
    image[..., :3] = 0.25
    image[..., 3] = 0.75
    result = downscale_nohalo_lohalo(image, "lohalo", 0.01, 16)
    assert result.dtype == torch.float16
    torch.testing.assert_close(result[..., 3], torch.full_like(result[..., 3], 0.75))


def test_nohalo_remains_locally_bounded():
    generator = torch.Generator().manual_seed(1234)
    image = torch.rand((1, 128, 192, 3), generator=generator)
    result = downscale_nohalo_lohalo(image, "nohalo", 0.01, 16)
    assert torch.isfinite(result).all()
    assert result.min() >= image.min() - 1e-6
    assert result.max() <= image.max() + 1e-6


@pytest.mark.parametrize("method", ["lohalo", "nohalo"])
def test_tiled_and_single_tile_results_match(monkeypatch, method):
    image = torch.linspace(0.0, 1.0, 128 * 192 * 3).reshape(1, 128, 192, 3)
    expected = downscale_nohalo_lohalo(image, method, 0.01, 16)
    monkeypatch.setattr(image_helpers, "_HALO_WORKSPACE_BYTES", 64 * 1024)
    tiled = downscale_nohalo_lohalo(image, method, 0.01, 16)
    torch.testing.assert_close(tiled, expected)
