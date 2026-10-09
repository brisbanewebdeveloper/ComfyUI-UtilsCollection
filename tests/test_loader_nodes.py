import pytest
import re

from ..helpers.loader_helpers import (
    extract_transformer_block_index,
    filter_lora_patches,
    parse_block_ranges,
    parse_layer_patterns,
)
from ..nodes.loader_nodes import UC_LoraLoaderModelOnly


def test_parse_block_ranges_valid():
    assert parse_block_ranges("") == set()
    assert parse_block_ranges("   ") == set()
    assert parse_block_ranges("1, 2, 3") == {1, 2, 3}
    assert parse_block_ranges("1,2,3") == {1, 2, 3}
    assert parse_block_ranges("1,2,4-5,7") == {1, 2, 4, 5, 7}
    assert parse_block_ranges(" 0 - 2 , 5 , 8 - 10 ") == {0, 1, 2, 5, 8, 9, 10}
    assert parse_block_ranges("5-3") == {3, 4, 5}


def test_parse_block_ranges_invalid():
    with pytest.raises(ValueError, match="expected integer"):
        parse_block_ranges("1, abc, 3")

    with pytest.raises(ValueError, match="expected integers"):
        parse_block_ranges("1, 2-xyz")

    with pytest.raises(ValueError, match="Invalid block range syntax"):
        parse_block_ranges("1, 2-3-4")


def test_parse_layer_patterns_valid():
    assert parse_layer_patterns("") == []
    assert parse_layer_patterns("   \n\n  ") == []

    patterns = parse_layer_patterns("img_attn\n  single_blocks.*linear  \n")
    assert len(patterns) == 2
    assert isinstance(patterns[0], re.Pattern)
    assert patterns[0].pattern == "img_attn"
    assert patterns[1].pattern == "single_blocks.*linear"


def test_parse_layer_patterns_invalid():
    with pytest.raises(ValueError, match="Invalid layer regex pattern"):
        parse_layer_patterns("img_attn\n[invalid_regex")


def test_extract_transformer_block_index():
    assert extract_transformer_block_index(
        "diffusion_model.input_blocks.1.1.transformer_blocks.0.attn1.to_q.weight"
    ) == 0
    assert extract_transformer_block_index(
        "diffusion_model.output_blocks.4.1.transformer_blocks.2.attn1.to_q.weight"
    ) == 2
    assert extract_transformer_block_index(
        "diffusion_model.double_blocks.3.img_attn.qkv.weight"
    ) == 3
    assert extract_transformer_block_index(
        "diffusion_model.single_blocks.15.linear1.weight"
    ) == 15
    assert extract_transformer_block_index(
        "diffusion_model.joint_blocks.5.x_block.attn.qkv.weight"
    ) == 5
    assert extract_transformer_block_index(
        "diffusion_model.blocks.4.self_attn.q.weight"
    ) == 4
    assert extract_transformer_block_index(
        "diffusion_model.layers.7.attention.qkv.weight"
    ) == 7
    assert extract_transformer_block_index(
        "diffusion_model.transformer_blocks.9.attn1.to_q.weight"
    ) == 9

    assert extract_transformer_block_index("diffusion_model.time_in.in_layer.weight") is None
    assert extract_transformer_block_index("diffusion_model.final_layer.linear.weight") is None
    assert extract_transformer_block_index("diffusion_model.txtfusion.weight") is None


def test_filter_lora_patches_empty_filters():
    patches = {
        "diffusion_model.double_blocks.0.img_attn.qkv.weight": 1,
        "diffusion_model.double_blocks.1.img_attn.qkv.weight": 2,
        "diffusion_model.time_in.in_layer.weight": 3,
    }
    assert filter_lora_patches(patches, "", "", whitelist=True) == patches
    assert filter_lora_patches(patches, "", "", whitelist=False) == patches


def test_filter_lora_patches_layer_regex_only():
    patches = {
        "diffusion_model.double_blocks.0.img_attn.qkv.weight": 1,
        "diffusion_model.double_blocks.0.txt_attn.qkv.weight": 2,
        "diffusion_model.time_in.in_layer.weight": 3,
    }
    whitelisted = filter_lora_patches(patches, "img_attn\ntime_in", "", whitelist=True)
    assert set(whitelisted.keys()) == {
        "diffusion_model.double_blocks.0.img_attn.qkv.weight",
        "diffusion_model.time_in.in_layer.weight",
    }

    blacklisted = filter_lora_patches(patches, "img_attn\ntime_in", "", whitelist=False)
    assert set(blacklisted.keys()) == {
        "diffusion_model.double_blocks.0.txt_attn.qkv.weight",
    }


def test_filter_lora_patches_block_ranges_only():
    patches = {
        "diffusion_model.double_blocks.0.img_attn.qkv.weight": 1,
        "diffusion_model.double_blocks.1.img_attn.qkv.weight": 2,
        "diffusion_model.double_blocks.2.img_attn.qkv.weight": 3,
        "diffusion_model.time_in.in_layer.weight": 4,
    }
    whitelisted = filter_lora_patches(patches, "", "0, 2", whitelist=True)
    assert set(whitelisted.keys()) == {
        "diffusion_model.double_blocks.0.img_attn.qkv.weight",
        "diffusion_model.double_blocks.2.img_attn.qkv.weight",
    }

    blacklisted = filter_lora_patches(patches, "", "0, 2", whitelist=False)
    assert set(blacklisted.keys()) == {
        "diffusion_model.double_blocks.1.img_attn.qkv.weight",
        "diffusion_model.time_in.in_layer.weight",
    }


def test_filter_lora_patches_combined_and():
    patches = {
        "diffusion_model.double_blocks.0.img_attn.qkv.weight": 1,
        "diffusion_model.double_blocks.0.txt_attn.qkv.weight": 2,
        "diffusion_model.double_blocks.1.img_attn.qkv.weight": 3,
        "diffusion_model.double_blocks.1.txt_attn.qkv.weight": 4,
        "diffusion_model.time_in.in_layer.weight": 5,
        "diffusion_model.final_layer.linear.weight": 6,
    }

    # Whitelist: block 0 only AND img_attn; plus time_in explicitly specified in regex
    whitelisted = filter_lora_patches(
        patches, "img_attn\ntime_in", "0", whitelist=True
    )
    assert set(whitelisted.keys()) == {
        "diffusion_model.double_blocks.0.img_attn.qkv.weight",
        "diffusion_model.time_in.in_layer.weight",
    }

    # Blacklist: block 0 img_attn and time_in excluded; rest retained
    blacklisted = filter_lora_patches(
        patches, "img_attn\ntime_in", "0", whitelist=False
    )
    assert set(blacklisted.keys()) == {
        "diffusion_model.double_blocks.0.txt_attn.qkv.weight",
        "diffusion_model.double_blocks.1.img_attn.qkv.weight",
        "diffusion_model.double_blocks.1.txt_attn.qkv.weight",
        "diffusion_model.final_layer.linear.weight",
    }


def test_uc_lora_loader_model_only_schema():
    schema = UC_LoraLoaderModelOnly.define_schema()
    assert schema.node_id == "UC_LoraLoaderModelOnly"
    assert schema.display_name == "Load LoRA for Model Only (Filtered)"
    assert schema.category == "advanced/model"
    input_names = [inp.id for inp in schema.inputs]
    assert input_names == [
        "model",
        "lora_name",
        "strength_model",
        "layer_filter",
        "block_filter",
        "whitelist",
    ]
    output_names = [out.id for out in schema.outputs]
    assert output_names == ["model"]
