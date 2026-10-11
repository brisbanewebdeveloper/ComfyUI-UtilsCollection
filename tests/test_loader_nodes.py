import pytest
import re

from ..helpers.loader_helpers import (
    apply_lora_stack_to_model,
    build_lora_stack_dynamic_combo,
    extract_lora_stack_entries,
    extract_transformer_block_index,
    filter_lora_patches,
    parse_block_ranges,
    parse_layer_patterns,
)
from ..nodes.loader_nodes import (
    UC_LoraLoaderModelOnly,
    UC_LoraStack,
    UC_LoraStackLoaderModelOnly,
)


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


def test_uc_lora_stack_loader_model_only_schema():
    schema = UC_LoraStackLoaderModelOnly.define_schema()
    assert schema.node_id == "UC_LoraStackLoaderModelOnly"
    assert schema.display_name == "Load LoRA Stack (Model Only)"
    assert schema.category == "advanced/model"
    assert "lora stack" in schema.search_aliases

    input_names = [inp.id for inp in schema.inputs]
    assert input_names == ["model", "add_lora"]
    output_names = [out.id for out in schema.outputs]
    assert output_names == ["model"]

    # Verify DynamicCombo structure
    combo = schema.inputs[1]
    assert combo.id == "add_lora"
    assert len(combo.options) == 2
    assert combo.options[0].key == "False"
    assert combo.options[0].inputs == []
    assert combo.options[1].key == "True"

    # Level 1 inputs inside option "True"
    sub_inputs = combo.options[1].inputs
    sub_ids = [inp.id for inp in sub_inputs]
    assert sub_ids == ["lora_name", "strength_model", "enabled", "add_lora_2"]

    # Verify depth 15 termination
    curr = combo
    for depth in range(1, 16):
        true_opt = next(o for o in curr.options if o.key == "True")
        ids = [inp.id for inp in true_opt.inputs]
        assert "lora_name" in ids
        assert "strength_model" in ids
        assert "enabled" in ids
        if depth < 15:
            assert f"add_lora_{depth + 1}" in ids
            curr = next(inp for inp in true_opt.inputs if inp.id == f"add_lora_{depth + 1}")
        else:
            assert len(ids) == 3  # Terminal level has no next add_lora combo

    # Verify serialization without circular reference
    dict_repr = combo.as_dict()
    assert dict_repr is not None
    assert "options" in dict_repr


def test_extract_lora_stack_entries():
    # 1. Disabled/empty
    assert extract_lora_stack_entries(None) == []
    assert extract_lora_stack_entries("False") == []
    assert extract_lora_stack_entries({"add_lora": "False"}) == []

    # 2. Single active LoRA
    single = {
        "add_lora": "True",
        "lora_name": "style_alpha.safetensors",
        "strength_model": 0.85,
        "enabled": True,
    }
    entries = extract_lora_stack_entries(single)
    assert len(entries) == 1
    assert entries[0]["lora_name"] == "style_alpha.safetensors"
    assert entries[0]["strength_model"] == 0.85
    assert entries[0]["strength"] == 0.85
    assert entries[0]["enabled"] is True

    # 3. Multi-level nested LoRAs with disabled item
    nested = {
        "add_lora": "True",
        "lora_name": "first.safetensors",
        "strength_model": 1.0,
        "enabled": True,
        "add_lora_2": {
            "add_lora_2": "True",
            "lora_name": "second.safetensors",
            "strength_model": 0.5,
            "enabled": False,
            "add_lora_3": {
                "add_lora_3": "True",
                "lora_name": "third.safetensors",
                "strength_model": -0.3,
                "enabled": True,
                "add_lora_4": {
                    "add_lora_4": "False",
                },
            },
        },
    }
    entries3 = extract_lora_stack_entries(nested)
    assert len(entries3) == 3
    assert entries3[0]["lora_name"] == "first.safetensors"
    assert entries3[0]["enabled"] is True
    assert entries3[1]["lora_name"] == "second.safetensors"
    assert entries3[1]["enabled"] is False
    assert entries3[2]["lora_name"] == "third.safetensors"
    assert entries3[2]["strength_model"] == -0.3

    # 4. Fallback flat kwargs
    kw_entries = extract_lora_stack_entries(
        add_lora="True",
        lora_name="flat.safetensors",
        strength_model=0.75,
        enabled=True,
    )
    assert len(kw_entries) == 1
    assert kw_entries[0]["lora_name"] == "flat.safetensors"
    assert kw_entries[0]["strength_model"] == 0.75


def test_apply_lora_stack_to_model(monkeypatch):
    load_calls = []
    patch_calls = []

    def mock_get_full_path_or_raise(folder, name):
        return f"/mock/path/{name}"

    def mock_load_torch_file(path, safe_load=True, return_metadata=True):
        load_calls.append(path)
        return ({"weight": path}, {"meta": path})

    def mock_load_lora_for_models(model, clip, lora, strength_model, strength_clip, lora_metadata=None):
        patch_calls.append((lora, strength_model, lora_metadata))
        return (f"patched({model}+{lora['weight']})", None)

    import folder_paths
    import comfy.utils
    import comfy.sd

    monkeypatch.setattr(folder_paths, "get_full_path_or_raise", mock_get_full_path_or_raise)
    monkeypatch.setattr(comfy.utils, "load_torch_file", mock_load_torch_file)
    monkeypatch.setattr(comfy.sd, "load_lora_for_models", mock_load_lora_for_models)

    entries = [
        {"lora_name": "lora_a.safetensors", "strength_model": 0.8, "enabled": True},
        {"lora_name": "lora_b.safetensors", "strength_model": 0.5, "enabled": False},  # skipped
        {"lora_name": "lora_c.safetensors", "strength_model": 0.0, "enabled": True},   # skipped
        {"lora_name": "lora_d.safetensors", "strength_model": 0.4, "enabled": True},
    ]

    model = "base_model"
    res_model, cache = apply_lora_stack_to_model(model, entries)

    # Only lora_a and lora_d should have been loaded and patched
    assert len(load_calls) == 2
    assert "/mock/path/lora_a.safetensors" in load_calls
    assert "/mock/path/lora_d.safetensors" in load_calls
    assert len(patch_calls) == 2
    assert patch_calls[0][1] == 0.8
    assert patch_calls[1][1] == 0.4
    assert res_model == "patched(patched(base_model+/mock/path/lora_a.safetensors)+/mock/path/lora_d.safetensors)"

    # Second pass should reuse cache without reloading files
    load_calls.clear()
    res_model2, cache2 = apply_lora_stack_to_model(model, [entries[0]], cache=cache)
    assert len(load_calls) == 0  # Reused from cache!
    assert "lora_a.safetensors" in res_model2


def test_uc_lora_stack_loader_model_only_execution(monkeypatch):
    import folder_paths
    import comfy.utils
    import comfy.sd

    monkeypatch.setattr(folder_paths, "get_full_path_or_raise", lambda f, n: f"/mock/{n}")
    monkeypatch.setattr(comfy.utils, "load_torch_file", lambda p, safe_load=True, return_metadata=True: ({"p": p}, None))
    monkeypatch.setattr(
        comfy.sd,
        "load_lora_for_models",
        lambda m, c, l, sm, sc, lora_metadata=None: (f"{m}+lora({sm})", None),
    )

    # 1. False add_lora returns base model untouched
    out = UC_LoraStackLoaderModelOnly.execute("my_model", add_lora="False")
    assert out.args[0] == "my_model"

    # 2. None add_lora returns base model untouched
    out = UC_LoraStackLoaderModelOnly.execute("my_model", add_lora=None)
    assert out.args[0] == "my_model"

    # 3. Nested stack with 2 active LoRAs
    stack_data = {
        "add_lora": "True",
        "lora_name": "lora_1.safetensors",
        "strength_model": 1.0,
        "enabled": True,
        "add_lora_2": {
            "add_lora_2": "True",
            "lora_name": "lora_2.safetensors",
            "strength_model": 0.5,
            "enabled": True,
            "add_lora_3": {
                "add_lora_3": "False",
            },
        },
    }
    out = UC_LoraStackLoaderModelOnly.execute("my_model", add_lora=stack_data)
    assert out.args[0] == "my_model+lora(1.0)+lora(0.5)"


def test_uc_lora_stack_with_comfy_v3_pipeline(monkeypatch):
    from comfy_api.latest._io import (
        build_nested_inputs,
        create_input_dict_v1,
        get_finalized_class_inputs,
    )
    import folder_paths
    import comfy.utils
    import comfy.sd

    monkeypatch.setattr(folder_paths, "get_full_path_or_raise", lambda f, n: f"/mock/{n}")
    monkeypatch.setattr(comfy.utils, "load_torch_file", lambda p, safe_load=True, return_metadata=True: ({"p": p}, None))
    monkeypatch.setattr(
        comfy.sd,
        "load_lora_for_models",
        lambda m, c, l, sm, sc, lora_metadata=None: (f"{m}+{l['p']}({sm})", None),
    )

    schema = UC_LoraStackLoaderModelOnly.define_schema()
    schema_dict = create_input_dict_v1(schema.inputs)

    # Simulate live inputs from ComfyUI frontend
    live_inputs = {
        "model": "base_diffusion_model",
        "add_lora": "True",
        "add_lora.lora_name": "comic_style.safetensors",
        "add_lora.strength_model": 0.8,
        "add_lora.enabled": True,
        "add_lora.add_lora_2": "True",
        "add_lora.add_lora_2.lora_name": "detail_enhancer.safetensors",
        "add_lora.add_lora_2.strength_model": 0.4,
        "add_lora.add_lora_2.enabled": True,
        "add_lora.add_lora_2.add_lora_3": "False",
    }

    finalized, _, v3_data = get_finalized_class_inputs(schema_dict, live_inputs)
    nested_inputs = build_nested_inputs(live_inputs, v3_data)

    out = UC_LoraStackLoaderModelOnly.execute(**nested_inputs)
    assert out.args[0] == "base_diffusion_model+/mock/comic_style.safetensors(0.8)+/mock/detail_enhancer.safetensors(0.4)"


def test_uc_lora_stack_schema():
    schema = UC_LoraStack.define_schema()
    assert schema.node_id == "UC_LoraStack"
    assert schema.display_name == "LoRA Stack"
    assert schema.category == "advanced/model"

    # Only add_lora input (no Model socket)
    input_names = [inp.id for inp in schema.inputs]
    assert input_names == ["add_lora"]

    # Output matches COMFY_DYNAMICGROUP_V3
    assert len(schema.outputs) == 1
    assert schema.outputs[0].id == "loras"
    assert schema.outputs[0].get_io_type() == "COMFY_DYNAMICGROUP_V3"

    # Verify template uses 'strength' matching _lora_template()
    combo = schema.inputs[0]
    true_opt = next(o for o in combo.options if o.key == "True")
    field_ids = [inp.id for inp in true_opt.inputs]
    assert "lora_name" in field_ids
    assert "strength" in field_ids
    assert "enabled" in field_ids
    assert "add_lora_2" in field_ids


def test_uc_lora_stack_execution():
    from comfy_extras.nodes_lora_stack import LoadLoraModel

    # 1. Disabled returns empty list
    out_empty = UC_LoraStack.execute(add_lora="False")
    assert out_empty.args[0] == []

    # 2. Active stack
    stack_data = {
        "add_lora": "True",
        "lora_name": "first.safetensors",
        "strength": 0.8,
        "enabled": True,
        "add_lora_2": {
            "add_lora_2": "True",
            "lora_name": "second.safetensors",
            "strength": -0.5,
            "enabled": False,
            "add_lora_3": {
                "add_lora_3": "False",
            },
        },
    }
    out = UC_LoraStack.execute(add_lora=stack_data)
    loras = out.args[0]
    assert len(loras) == 2
    assert loras[0] == {
        "lora_name": "first.safetensors",
        "strength": 0.8,
        "strength_model": 0.8,
        "enabled": True,
    }
    assert loras[1] == {
        "lora_name": "second.safetensors",
        "strength": -0.5,
        "strength_model": -0.5,
        "enabled": False,
    }

    # 3. Verify the output list directly feeds into LoadLoraModel.execute
    applied_calls = []

    def mock_load_file(name):
        return ("mock_lora", {"meta": "data"})

    def mock_apply(model, clip, lora, strength_model, strength_clip, lora_metadata=None):
        applied_calls.append((lora, strength_model))
        return (f"{model}+patched({strength_model})", None)

    import comfy_extras.nodes_lora_stack as nls
    orig_load = nls._load_lora_file
    orig_apply = nls.comfy.sd.load_lora_for_models
    try:
        nls._load_lora_file = mock_load_file
        nls.comfy.sd.load_lora_for_models = mock_apply

        res = LoadLoraModel.execute("base_model", loras)
        assert res.result == ("base_model+patched(0.8)",)
        assert len(applied_calls) == 1  # second was disabled so not applied
    finally:
        nls._load_lora_file = orig_load
        nls.comfy.sd.load_lora_for_models = orig_apply
