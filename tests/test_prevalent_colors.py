import pathlib
import sys
import types

import torch


CUSTOM_NODE_ROOT = pathlib.Path(__file__).parents[1]
PACKAGE_NAME = "utils_collection_prevalent_colors_test"
package = types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(CUSTOM_NODE_ROOT)]
sys.modules.setdefault(PACKAGE_NAME, package)

from utils_collection_prevalent_colors_test.helpers.color_palette_helpers import (
    _describe_palette,
    _nearest_color_names,
    _render_palette_grid,
    extract_prevalent_hex_colors,
)
from utils_collection_prevalent_colors_test.nodes.image_nodes import (
    UC_ExtractPrevalentColors,
)


def test_schema_exposes_requested_controls_and_list_output():
    schema = UC_ExtractPrevalentColors.define_schema()

    assert schema.node_id == "UC_ExtractPrevalentColors"
    assert schema.display_name == "Extract Prevalent Colors"
    assert [value.id for value in schema.inputs] == [
        "image",
        "color_count",
        "prefix_hash",
    ]
    assert schema.inputs[1].default == 8
    assert schema.inputs[2].default is True
    assert [output.id for output in schema.outputs] == [
        "colors",
        "palette_grid",
        "color_names",
        "palette_description",
    ]
    assert [output.is_output_list for output in schema.outputs] == [
        False,
        False,
        False,
        False,
    ]


def test_colors_are_ordered_by_prevalence():
    image = torch.zeros((1, 10, 10, 3))
    image[:, :7, :, 0] = 1.0
    image[:, 7:, :, 2] = 1.0

    assert extract_prevalent_hex_colors(image, 2, True) == "#FF0000, #0000FF"


def test_low_count_merges_similar_shades_before_distinct_color():
    image = torch.zeros((1, 10, 10, 3))
    image[:, :4, :, 0] = 1.0
    image[:, 4:7, :, 0] = 0.98
    image[:, 4:7, :, 1] = 0.01
    image[:, 7:, :, 2] = 1.0

    palette = extract_prevalent_hex_colors(image, 2, True).split(", ")

    assert palette[0] in {"#FF0000", "#FD0100"}
    assert palette[1] == "#0000FF"


def test_similar_shades_combine_their_prevalence_before_ranking():
    image = torch.zeros((1, 10, 10, 3))
    image[:, :2, :, 0] = 1.0
    image[:, 2:4, :, 0] = 0.98
    image[:, 2:4, :, 1] = 0.01
    image[:, 4:6, :, 0] = 0.96
    image[:, 4:6, :, 1] = 0.02
    image[:, 6:, :, 2] = 1.0

    palette = extract_prevalent_hex_colors(image, 2, True).split(", ")

    assert palette[0] != "#0000FF"
    assert palette[1] == "#0000FF"


def test_higher_count_preserves_more_variation_within_color_family():
    image = torch.zeros((1, 10, 10, 3))
    image[:, :3, :, 0] = 1.0
    image[:, 3:6, :, 0] = 0.8
    image[:, 3:6, :, 1] = 0.08
    image[:, 6:, :, 2] = 1.0

    two_colors = extract_prevalent_hex_colors(image, 2, True).split(", ")
    three_colors = extract_prevalent_hex_colors(image, 3, True).split(", ")

    assert len(two_colors) == 2
    assert len(three_colors) == 3
    assert sum(color != "#0000FF" for color in two_colors) == 1
    assert sum(color != "#0000FF" for color in three_colors) == 2


def test_image_batch_returns_one_combined_palette_and_grid():
    images = torch.stack(
        [
            torch.ones((4, 4, 3)) * torch.tensor([1.0, 0.0, 0.0]),
            torch.ones((4, 4, 3)) * torch.tensor([0.0, 1.0, 0.0]),
        ]
    )

    output = UC_ExtractPrevalentColors.execute(images, 4, False)

    palette, grid, color_names, palette_description = output.args
    assert set(palette.split(", ")) == {"FF0000", "00FF00"}
    assert grid.shape == (1, 128, 256, 3)
    assert len(color_names.split(", ")) == 2
    assert "palette with" in palette_description


def test_requested_count_is_a_maximum_when_image_has_fewer_colors():
    image = torch.ones((1, 32, 32, 3)) * 0.5

    assert extract_prevalent_hex_colors(image, 8, True) == "#808080"


def test_palette_grid_layout_and_transparent_unused_cell():
    colors = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
        ]
    )

    grid = _render_palette_grid(colors)

    assert grid.shape == (1, 256, 384, 3)
    assert torch.equal(grid[0, 64, 64], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.equal(grid[0, 64, 192], torch.tensor([0.0, 1.0, 0.0]))
    assert torch.equal(grid[0, 192, 64], torch.tensor([1.0, 1.0, 0.0]))
    assert torch.equal(grid[0, 192, 192], torch.tensor([1.0, 0.0, 1.0]))
    assert torch.equal(grid[0, 192, 320], torch.zeros(3))


def test_palette_grid_dimensions_follow_distributed_layout():
    expected_shapes = {
        2: (1, 128, 256, 3),
        3: (1, 128, 384, 3),
        4: (1, 256, 256, 3),
        5: (1, 256, 384, 3),
        6: (1, 256, 384, 3),
    }

    for color_count, shape in expected_shapes.items():
        colors = torch.linspace(0.0, 1.0, color_count)[:, None].expand(-1, 3)
        assert _render_palette_grid(colors).shape == shape


def test_nearest_names_are_separate_and_follow_palette_order():
    colors = torch.tensor(
        [
            [229.0 / 255.0, 0.0, 0.0],
            [3.0 / 255.0, 67.0 / 255.0, 223.0 / 255.0],
        ]
    )

    assert _nearest_color_names(colors) == "red, blue"


def test_palette_description_reports_computed_semantics_without_surface_claims():
    warm_colors = torch.tensor([[1.0, 0.3, 0.0], [0.8, 0.15, 0.0]])
    cool_colors = torch.tensor([[0.0, 0.2, 1.0], [0.0, 0.6, 0.8]])
    weights = torch.tensor([3.0, 1.0])

    warm_description = _describe_palette(warm_colors, weights)
    cool_description = _describe_palette(cool_colors, weights)

    assert warm_description.startswith("warm,")
    assert cool_description.startswith("cool,")
    assert "matte" not in warm_description
    assert "gloss" not in warm_description
