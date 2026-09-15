import pathlib
import sys
import types

import pytest
import torch


CUSTOM_NODE_ROOT = pathlib.Path(__file__).parents[1]
PACKAGE_NAME = "utils_collection_bbox_test"
package = types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(CUSTOM_NODE_ROOT)]
sys.modules.setdefault(PACKAGE_NAME, package)

from utils_collection_bbox_test.nodes.utils_nodes import (
    UC_AdjustBoundingBox,
    UC_ExtractBoundingBox,
    UC_ExtractImage,
    UC_ExtractMask,
    UC_Ideogram4BoundingBoxCrop,
)
from utils_collection_bbox_test.helpers.image_helpers import mask_to_bounding_box
from utils_collection_bbox_test.nodes.image_nodes import UC_MaskToBoundingBox


def test_mask_to_bounding_box_uses_core_dictionary_shape():
    mask = torch.ones((1, 512, 560), dtype=torch.float32)

    bounding_box, cropped_image = mask_to_bounding_box(mask)

    assert bounding_box == {"x": 0, "y": 0, "width": 560, "height": 512}
    assert cropped_image is None


def test_mask_to_bounding_box_converts_inclusive_corners_to_extents_and_crops():
    mask = torch.zeros((1, 200, 240), dtype=torch.float32)
    mask[:, 98:142, 68:172] = 1.0
    image = torch.rand((1, 200, 240, 3))

    output = UC_MaskToBoundingBox.execute(mask, False, image).args

    assert output[0] == {"x": 68, "y": 98, "width": 104, "height": 44}
    torch.testing.assert_close(output[1], image[:, 98:142, 68:172, :])


def test_mask_to_bounding_box_rejects_empty_mask():
    with pytest.raises(ValueError, match="at least one nonzero pixel"):
        mask_to_bounding_box(torch.zeros((1, 16, 16)))


def test_mask_to_bounding_box_schema_uses_core_bounding_box_output():
    schema = UC_MaskToBoundingBox.define_schema()

    assert schema.node_id == "UC_MaskToBoundingBox"
    assert [input_.id for input_ in schema.inputs] == ["mask", "invert", "image"]
    assert schema.outputs[0].io_type == "BOUNDING_BOX"


def _result(data, index=0, expansion=0, axis="both", multiple="0"):
    return UC_AdjustBoundingBox.execute(data, index, expansion, axis, multiple).args[0]


def test_extract_and_uniform_expansion():
    data = {"detections": [{"x": 10, "y": 20, "width": 30, "height": 40}]}
    assert _result(data, expansion=5) == {"x": 5, "y": 15, "width": 40, "height": 50}


def test_axis_expansion_and_multiple_alignment():
    data = [{"x": 100, "y": 200, "width": 31, "height": 33}]
    assert _result(data, expansion=3, axis="horizontal", multiple="16") == {
        "x": 92,
        "y": 193,
        "width": 48,
        "height": 48,
    }


def test_index_selection_and_json_input():
    data = '[{"x": 1, "y": 2, "width": 8, "height": 9}, {"x": 4, "y": 5, "width": 6, "height": 7}]'
    assert _result(data, index=1) == {"x": 4, "y": 5, "width": 6, "height": 7}


def test_extract_bounding_box_appends_core_compatible_box():
    output = UC_ExtractBoundingBox.execute(
        [[{"x": 4, "y": 5, "width": 6, "height": 7}]], 0,
    ).args

    assert output == (
        4,
        5,
        6,
        7,
        {"x": 4, "y": 5, "width": 6, "height": 7},
    )


def test_invalid_box_rejected():
    with pytest.raises(ValueError, match="positive width and height"):
        _result({"x": 0, "y": 0, "width": 0, "height": 8})


def test_extract_mask_selects_batch_index():
    masks = torch.stack((torch.zeros(3, 4), torch.ones(3, 4)))
    extracted = UC_ExtractMask.execute(masks, 1).args[0]

    assert extracted.shape == (1, 3, 4)
    assert extracted.all()


def test_extract_mask_rejects_out_of_range_index():
    with pytest.raises(ValueError, match=r"Index 2 is out of range.*1 mask"):
        UC_ExtractMask.execute(torch.zeros(1, 3, 4), 2)


def test_extract_image_selects_batch_index_and_preserves_alpha():
    first = torch.zeros(1, 3, 4, 4)
    second = torch.zeros(1, 7, 5, 4)
    second[..., :3] = 0.25
    second[..., 3] = 0.75
    images = [first, second]

    extracted = UC_ExtractImage.execute(images, [1]).args[0]

    assert extracted.shape == (1, 7, 5, 4)
    assert torch.allclose(extracted, second)


def test_extract_image_consumes_native_image_list_as_one_input():
    schema = UC_ExtractImage.define_schema()

    assert schema.is_input_list is True
    assert schema.inputs[0].io_type == "IMAGE"
    assert schema.outputs[0].io_type == "IMAGE"


def test_extract_image_rejects_out_of_range_index():
    with pytest.raises(ValueError, match=r"Index 2 is out of range.*1 image"):
        UC_ExtractImage.execute([torch.zeros(1, 3, 4, 4)], [2])


def test_ideogram_bbox_crop_outputs_crop_and_normalized_string():
    image = torch.zeros(1, 10, 20, 3)
    image[:, 1:6, 5:15] = 1

    crop, ig4_bbox, bounding_box = UC_Ideogram4BoundingBoxCrop.execute(
        image,
        [[{"x": 5, "y": 1, "width": 10, "height": 5}]],
        0,
    ).args

    assert crop.shape == (1, 5, 10, 3)
    assert crop.all()
    assert ig4_bbox == "[100,250,600,750]"
    assert bounding_box == {"x": 5, "y": 1, "width": 10, "height": 5}


def test_ideogram_bbox_crop_clips_to_the_exact_image_region():
    image = torch.ones(1, 10, 20, 3)
    crop, ig4_bbox, bounding_box = UC_Ideogram4BoundingBoxCrop.execute(
        image,
        [[{"x": -5, "y": 1, "width": 15, "height": 20}]],
        0,
    ).args

    assert crop.shape == (1, 9, 10, 3)
    assert ig4_bbox == "[100,0,1000,500]"
    assert bounding_box == {"x": 0, "y": 1, "width": 10, "height": 9}
