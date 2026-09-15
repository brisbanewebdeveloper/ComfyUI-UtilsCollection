import pathlib
import sys
import types

import pytest
import torch


CUSTOM_NODE_ROOT = pathlib.Path(__file__).parents[1]
PACKAGE_NAME = "utils_collection_visual_consensus_test"
package = types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(CUSTOM_NODE_ROOT)]
sys.modules.setdefault(PACKAGE_NAME, package)

from comfy.cli_args import args as cli_args

prior_cpu = cli_args.cpu
cli_args.cpu = True
try:
    from utils_collection_visual_consensus_test.helpers import encoder_helpers
    from utils_collection_visual_consensus_test.nodes.encoder_nodes import (
        AdvancedConsensusConfig,
        UC_AdvancedConsensusConfiguration,
        UC_AdvancedVisConEncoder,
        UC_AdvancedVisConEncoderTokenFusion,
        UC_ConditioningConsensusBlend,
        UC_TextConsensusBlendConfig,
        UC_VisualFusionConfig,
        UC_VisualConsensusConfiguration,
        VisualFusionConfig,
        VisualConsensusConfig,
    )
finally:
    cli_args.cpu = prior_cpu


def _inputs(node):
    return {value.id: value for value in node.define_schema().inputs}


def test_fresh_node_ids_names_and_socket_types():
    expected = {
        UC_AdvancedVisConEncoder: (
            "UC_AdvancedVisConEncoder",
            "Advanced Visual Consensus Encoder",
        ),
        UC_AdvancedVisConEncoderTokenFusion: (
            "UC_AdvancedVisConEncoderTokenFusion",
            "Advanced Visual Consensus Encoder (TokenFusion)",
        ),
        UC_VisualConsensusConfiguration: (
            "UC_VisualConsensusConfiguration",
            "Visual Consensus Configuration",
        ),
        UC_AdvancedConsensusConfiguration: (
            "UC_AdvancedConsensusConfiguration",
            "Advanced Consensus Configuration",
        ),
    }
    for node, (node_id, display_name) in expected.items():
        schema = node.define_schema()
        assert schema.node_id == node_id
        assert schema.display_name == display_name

    assert AdvancedConsensusConfig.io_type == "ADVANCED_CONSENSUS_CONFIG"
    assert VisualFusionConfig.io_type == "VISUAL_FUSION_CONFIG"
    assert VisualConsensusConfig.io_type == "VISUAL_CONSENSUS_CONFIG"
    assert _inputs(UC_AdvancedVisConEncoder)["visual_consensus_config"].optional is False
    assert list(_inputs(UC_AdvancedVisConEncoderTokenFusion)) == list(
        _inputs(UC_AdvancedVisConEncoder)
    )
    assert _inputs(UC_AdvancedVisConEncoder)["fusion_method"].default == "conds_fusion"
    assert _inputs(UC_AdvancedVisConEncoderTokenFusion)["fusion_method"].default == "token_fusion"
    assert "fuses visual and DeepStack tokens before one conditioning encode" in (
        _inputs(UC_AdvancedVisConEncoderTokenFusion)[
            "visual_consensus_config"
        ].tooltip
    )


def test_joint_schema_accepts_complete_child_configs():
    inputs = _inputs(UC_VisualConsensusConfiguration)
    assert list(inputs) == ["visual_fusion_config", "consensus_config"]
    assert inputs["visual_fusion_config"].io_type == "VISUAL_FUSION_CONFIG"
    assert inputs["consensus_config"].io_type == "ADVANCED_CONSENSUS_CONFIG"


def test_joint_config_uses_authoritative_child_configs():
    visual = UC_VisualFusionConfig.execute(
        "spatial-block-interleave",
        7,
        0.2,
        True,
        "fresh.safetensors",
        99,
        "legacy-flat",
        "block-interleave",
        True,
        0.4,
    ).args[0]
    consensus = UC_AdvancedConsensusConfiguration.execute(
        "custom",
        "linear",
        "mean",
        "index",
        0.2,
        -0.1,
        1.2,
        0.7,
        False,
        1.3,
        True,
        True,
        0.8,
        True,
        7,
        96,
    ).args[0]
    config = UC_VisualConsensusConfiguration.execute(visual, consensus).args[0]

    assert config["enable_spatial_fusion"] is True
    assert config["enable_consensus"] is True
    assert config["visual"]["visual_fusion_method"] == "spatial-block-interleave"
    assert config["visual"]["visual_block_size"] == 7
    assert config["visual"]["dither_secondary_pattern"] == "block-interleave"
    assert config["consensus"] == consensus
    assert config["consensus"]["blend_preset"] == "custom"


def test_advanced_consensus_inherits_text_config_and_adds_sampling_controls():
    inputs = _inputs(UC_AdvancedConsensusConfiguration)
    base_inputs = _inputs(UC_TextConsensusBlendConfig)
    assert list(inputs) == [*base_inputs, "resolution_samples", "sample_offset"]
    assert inputs["sample_offset"].default == 32
    assert inputs["sample_offset"].min == 32
    assert inputs["sample_offset"].max == 512
    assert inputs["sample_offset"].step == 32
    preset = inputs["blend_preset"]
    assert preset.default == "baseline"
    assert "custom" in preset.options
    assert "baseline" in preset.options
    assert "off" in preset.options

    config = UC_AdvancedConsensusConfiguration.execute(
        "high_clarity",
        "linear",
        "mean",
        "index",
        0.1,
        -0.2,
        1.0,
        0.5,
        False,
        1.4,
        True,
        True,
        0.65,
        True,
        5,
        64,
    ).args[0]
    assert config["blend_preset"] == "high_clarity"
    assert config["position_weight"] == 0.65
    assert config["preserve_common_prefix"] is True
    assert config["global_scale"] == 1.4
    assert config["resolution_samples"] == 5
    assert config["sample_offset"] == 64
    resolved = encoder_helpers.resolve_consensus_blend_settings(config)
    assert resolved["blend_preset"] == "high_clarity"
    assert resolved["power_alpha"] == 3.0
    assert resolved["position_weight"] == 0.65
    assert resolved["preserve_common_prefix"] is True
    assert resolved["global_scale"] == 1.4


def test_resolution_samples_reaches_consensus_config():
    visual = UC_VisualFusionConfig.execute("off", 2, 0.5).args[0]
    consensus = UC_AdvancedConsensusConfiguration.execute(
        "baseline", "consensus", "median", "similarity", 0.4, 0.0, 2.0,
        0.0, True, 1.0, False, False, 0.0, False, 7, 128,
    ).args[0]
    config = UC_VisualConsensusConfiguration.execute(visual, consensus).args[0]

    assert config["consensus"]["resolution_samples"] == 7
    assert config["consensus"]["sample_offset"] == 128
    assert config["enable_spatial_fusion"] is False
    assert config["enable_consensus"] is True


def test_one_batched_socket_is_one_lane_of_separate_sources():
    images = torch.arange(3 * 2 * 2 * 1).reshape(3, 2, 2, 1)
    lanes, references = encoder_helpers.build_visual_consensus_batch_lanes(
        {"image0": images}
    )
    assert len(lanes) == 1
    assert len(lanes[0]) == 3
    assert len(references) == 3
    assert all(image.shape == (1, 2, 2, 1) for image in lanes[0])
    separate_lanes, _ = encoder_helpers.build_visual_consensus_batch_lanes(
        {
            "image0": images[0:1],
            "image1": images[1:2],
            "image2": images[2:3],
        }
    )
    assert len(separate_lanes) == 1
    assert all(
        torch.equal(batched, separate)
        for batched, separate in zip(lanes[0], separate_lanes[0])
    )


def test_multiple_batches_pair_by_index_and_broadcast_singletons():
    first = torch.tensor([[[[1.0]]], [[[2.0]]], [[[3.0]]]])
    second = torch.tensor([[[[10.0]]], [[[20.0]]], [[[30.0]]]])
    singleton = torch.tensor([[[[99.0]]]])
    lanes, _ = encoder_helpers.build_visual_consensus_batch_lanes(
        {"image0": first, "image1": singleton, "image2": second}
    )
    assert [
        [float(source.item()) for source in lane]
        for lane in lanes
    ] == [
        [1.0, 99.0, 10.0],
        [2.0, 99.0, 20.0],
        [3.0, 99.0, 30.0],
    ]


def test_multiple_unequal_batches_fail_concisely():
    with pytest.raises(ValueError, match="batch lengths must match or be 1"):
        encoder_helpers.build_visual_consensus_batch_lanes(
            {
                "image0": torch.zeros(2, 1, 1, 1),
                "image1": torch.zeros(3, 1, 1, 1),
            }
        )


def test_complete_conditioning_helper_matches_consensus_math(monkeypatch):
    monkeypatch.setattr(
        encoder_helpers.comfy.model_management,
        "get_torch_device",
        lambda: torch.device("cpu"),
    )
    monkeypatch.setattr(
        encoder_helpers.comfy.model_management,
        "intermediate_dtype",
        lambda: torch.float32,
    )
    first = [[torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]), {"pooled_output": torch.tensor([[1.0, 0.0]])}]]
    second = [[torch.tensor([[[3.0, 0.0], [0.0, 3.0]]]), {"pooled_output": torch.tensor([[3.0, 0.0]])}]]
    config = {"blend_preset": "custom", "blend_method": "linear", "global_scale": 1.0}
    actual = encoder_helpers.blend_complete_conditionings([first, second], config)
    expected_tensor, expected_pooled = encoder_helpers.blend_text_vectors(
        {"a": first[0][0], "b": second[0][0]},
        config,
        {"a": first[0][1]["pooled_output"], "b": second[0][1]["pooled_output"]},
        device=torch.device("cpu"),
        compute_dtype=torch.float32,
    )
    assert torch.equal(actual[0][0], expected_tensor)
    assert torch.equal(actual[0][1]["pooled_output"], expected_pooled)


def test_complete_conditioning_consensus_preserves_minimax_reference_tags(monkeypatch):
    monkeypatch.setattr(
        encoder_helpers.comfy.model_management,
        "get_torch_device",
        lambda: torch.device("cpu"),
    )
    monkeypatch.setattr(
        encoder_helpers.comfy.model_management,
        "intermediate_dtype",
        lambda: torch.float32,
    )
    short_tags = torch.tensor([1, 0])
    long_tags = torch.tensor([1, 0, 0, 1])
    short = [[torch.ones(1, 2, 3), {"minimax_token_tags": short_tags}]]
    long = [[torch.ones(1, 4, 3), {"minimax_token_tags": long_tags}]]

    result = encoder_helpers.blend_complete_conditionings(
        [short, long],
        {"blend_preset": "custom", "blend_method": "linear"},
    )

    assert result[0][0].shape[1] == 4
    assert torch.equal(result[0][1]["minimax_token_tags"], long_tags)


def test_complete_conditioning_consensus_rejects_bad_minimax_tags(monkeypatch):
    monkeypatch.setattr(
        encoder_helpers.comfy.model_management,
        "get_torch_device",
        lambda: torch.device("cpu"),
    )
    monkeypatch.setattr(
        encoder_helpers.comfy.model_management,
        "intermediate_dtype",
        lambda: torch.float32,
    )
    bad = [[torch.ones(1, 3, 2), {"minimax_token_tags": torch.ones(2)}]]
    other = [[torch.ones(1, 2, 2), {"minimax_token_tags": torch.ones(2)}]]

    with pytest.raises(ValueError, match="modality tags do not match"):
        encoder_helpers.blend_complete_conditionings(
            [bad, other],
            {"blend_preset": "custom", "blend_method": "linear"},
        )


def _execution_config(spatial=True, consensus=True, samples=1):
    return {
        "enable_spatial_fusion": spatial,
        "enable_consensus": consensus,
        "visual": {
            "visual_fusion_method": "spatial-checkerboard",
            "visual_encoder_path": "grid-deepstack",
        },
        "consensus": {
            "blend_preset": "baseline",
            "resolution_samples": samples,
        },
    }


def _mock_execution_boundaries(monkeypatch):
    encoded = []
    fused = []
    blended = []

    def encode(clip, source, resolution, prompt, path):
        encoded.append((float(source.flatten()[0]), resolution, path))
        tensor = torch.tensor([[[float(source.flatten()[0])]]])
        conditioning = [[tensor, {}]]
        return {"conditioning": conditioning}

    def spatial(branches, config, clip, allow_export):
        fused.append((len(branches), allow_export))
        return branches[0]["conditioning"]

    def consensus(conditionings, config):
        blended.append((len(conditionings), config))
        return conditionings[0]

    monkeypatch.setattr(
        encoder_helpers, "_encode_visual_consensus_source", encode
    )
    monkeypatch.setattr(
        encoder_helpers, "_spatially_fuse_visual_consensus_sources", spatial
    )
    monkeypatch.setattr(
        encoder_helpers, "blend_complete_conditionings", consensus
    )
    return encoded, fused, blended


def _execute_mocked(image_inputs, config):
    return encoder_helpers.execute_advanced_visual_consensus(
        object(),
        "prompt",
        "",
        384,
        image_inputs,
        config,
        "Original",
        "off",
        None,
        1.0,
        8,
        lambda clip, conditioning, latents, mode: conditioning,
    )


def test_token_fusion_variant_fuses_before_one_resolution_encode(monkeypatch):
    tokenized = []
    fused = []
    blended = []

    def tokenize(_clip, source, resolution, _prompt):
        value = float(source.flatten()[0])
        tokenized.append((value, resolution))
        return {"qwen": [[(int(value), 1.0)]]}

    def encode(_clip, token_sources, config, **kwargs):
        fused.append((len(token_sources), config["save_blended_embeds"], kwargs))
        return [[torch.ones(1, 2, 3), {}]]

    def consensus(conditionings, config):
        blended.append((len(conditionings), config))
        return conditionings[0]

    monkeypatch.setattr(
        encoder_helpers, "_tokenize_visual_consensus_source", tokenize
    )
    monkeypatch.setattr(
        encoder_helpers, "encode_token_fused_visual_sources", encode
    )
    monkeypatch.setattr(
        encoder_helpers,
        "_encode_visual_consensus_source",
        lambda *_args, **_kwargs: pytest.fail("legacy full encode must not run"),
    )
    monkeypatch.setattr(
        encoder_helpers, "blend_complete_conditionings", consensus
    )
    config = _execution_config()
    config["visual"]["save_blended_embeds"] = True
    images = torch.tensor([[[[1.0]]], [[[2.0]]], [[[3.0]]]])

    encoder_helpers.execute_advanced_visual_consensus(
        object(),
        "prompt",
        "",
        384,
        {"image0": images},
        config,
        "Original",
        "off",
        None,
        1.0,
        8,
        lambda clip, conditioning, latents, mode: conditioning,
        token_fusion=True,
    )

    assert tokenized == [(1.0, 384), (2.0, 384), (3.0, 384)]
    assert fused == [
        (
            3,
            True,
            {"visual_encoder_path": "grid-deepstack"},
        )
    ]
    assert blended[0][0] == 1


def test_one_requested_resolution_sample_is_not_automatically_expanded(monkeypatch):
    encoded, fused, blended = _mock_execution_boundaries(monkeypatch)
    images = torch.tensor([[[[1.0]]], [[[2.0]]], [[[3.0]]]])
    _execute_mocked({"image0": images}, _execution_config())

    assert len(encoded) == 3
    assert [resolution for _, resolution, _ in encoded[:3]] == [384, 384, 384]
    assert [count for count, _ in fused] == [3]
    assert blended[0][0] == 1


def test_one_requested_resolution_sample_is_exact_across_batch_lanes(monkeypatch):
    encoded, fused, blended = _mock_execution_boundaries(monkeypatch)
    first = torch.tensor([[[[1.0]]], [[[2.0]]], [[[3.0]]]])
    second = torch.tensor([[[[4.0]]], [[[5.0]]], [[[6.0]]]])
    _execute_mocked(
        {"image0": first, "image1": second},
        _execution_config(samples=1),
    )

    assert len(encoded) == 6
    assert [count for count, _ in fused] == [2, 2, 2]
    assert blended[0][0] == 3


def test_consensus_only_retains_every_source_resolution_conditioning(monkeypatch):
    encoded, fused, blended = _mock_execution_boundaries(monkeypatch)
    images = torch.tensor([[[[1.0]]], [[[2.0]]], [[[3.0]]]])
    _execute_mocked(
        {"image0": images},
        _execution_config(spatial=False, consensus=True),
    )

    assert len(encoded) == 3
    assert fused == []
    assert blended[0][0] == 3


def test_configured_five_resolution_samples_reach_consensus(monkeypatch):
    encoded, fused, blended = _mock_execution_boundaries(monkeypatch)
    _execute_mocked(
        {"image0": torch.ones(1, 2, 2, 1)},
        _execution_config(samples=5),
    )

    assert len(encoded) == 5
    assert len(fused) == 5
    assert blended[0][0] == 5


def test_both_disabled_encodes_only_first_source_at_base_resolution(monkeypatch):
    encoded, fused, blended = _mock_execution_boundaries(monkeypatch)
    images = torch.tensor([[[[1.0]]], [[[2.0]]], [[[3.0]]]])
    output = _execute_mocked(
        {"image0": images},
        _execution_config(spatial=False, consensus=False),
    )

    assert encoded == [(1.0, 384, "grid-deepstack")]
    assert fused == []
    assert blended == []
    assert output[0][0].item() == 1.0


def test_original_resolution_accepts_one_exact_sample(monkeypatch):
    encoded, fused, blended = _mock_execution_boundaries(monkeypatch)
    encoder_helpers.execute_advanced_visual_consensus(
        object(),
        "prompt",
        "",
        0,
        {"image0": torch.ones(1, 2, 2, 1)},
        _execution_config(samples=1),
        "Original",
        "off",
        None,
        1.0,
        8,
        lambda clip, conditioning, latents, mode: conditioning,
    )

    assert encoded == [(1.0, None, "grid-deepstack")]
    assert fused == [(1, True)]
    assert blended[0][0] == 1


def _source_branch(source):
    value = float(source.flatten()[0])
    tensor = torch.tensor([[[-1.0], [value], [value], [value], [value], [-2.0]]])
    pooled = torch.tensor([[value, value + 0.5]])
    metadata = {"pooled_output": pooled, "shared": "metadata"}
    return {
        "conditioning": [[tensor, metadata]],
        "tensor": tensor,
        "metadata": metadata,
        "pooled": pooled,
        "tokens": {"qwen": [[(0, 1.0)]]},
        "visual_range": (1, 5),
        "grid": (2, 2),
        "image": source,
    }


def test_minimax_h3_rejects_consensus_disabled_multiple_lanes():
    class Clip:
        @staticmethod
        def tokenize(_text, **_kwargs):
            return {"qwen3vl_32b": [[(151643, 1.0)]]}

    with pytest.raises(ValueError, match="MiniMax H3 supports batch size 1"):
        encoder_helpers.execute_advanced_visual_consensus(
            Clip(),
            "prompt",
            "",
            384,
            {
                "image0": torch.zeros(2, 1, 1, 1),
                "image1": torch.zeros(2, 1, 1, 1),
            },
            _execution_config(spatial=True, consensus=False),
            "Original",
            "off",
            None,
            1.0,
            8,
            lambda _clip, conditioning, _latents, _mode: conditioning,
        )


def test_minimax_h3_rejects_generic_reference_latents():
    class Clip:
        @staticmethod
        def tokenize(_text, **_kwargs):
            return {"qwen3vl_32b": [[(151643, 1.0)]]}

    with pytest.raises(ValueError, match="MiniMax H3 reference latents require Core"):
        encoder_helpers.execute_advanced_visual_consensus(
            Clip(),
            "prompt",
            "",
            384,
            {},
            _execution_config(),
            "Original",
            "single",
            None,
            1.0,
            8,
            lambda _clip, conditioning, _latents, _mode: conditioning,
        )


def _execute_source_distinct(monkeypatch, image_inputs, config):
    monkeypatch.setattr(
        encoder_helpers,
        "_encode_visual_consensus_source",
        lambda _clip, source, _resolution, _prompt, _path: _source_branch(source),
    )
    monkeypatch.setattr(
        encoder_helpers.comfy.model_management,
        "get_torch_device",
        lambda: torch.device("cpu"),
    )
    monkeypatch.setattr(
        encoder_helpers.comfy.model_management,
        "intermediate_dtype",
        lambda: torch.float32,
    )
    return _execute_mocked(image_inputs, config)


def test_original_resolution_rejects_multiple_adjacent_samples(monkeypatch):
    _mock_execution_boundaries(monkeypatch)
    with pytest.raises(
        ValueError,
        match="Original VLM resolution cannot construct adjacent resolution samples",
    ):
        encoder_helpers.execute_advanced_visual_consensus(
            object(),
            "prompt",
            "",
            0,
            {"image0": torch.ones(1, 2, 2, 1)},
            _execution_config(samples=3),
            "Original",
            "off",
            None,
            1.0,
            8,
            lambda clip, conditioning, latents, mode: conditioning,
        )


def test_consensus_disabled_preserves_execution_lanes_as_batch(monkeypatch):
    encoded, fused, blended = _mock_execution_boundaries(monkeypatch)
    first = torch.tensor([[[[1.0]]], [[[2.0]]], [[[3.0]]]])
    second = torch.tensor([[[[4.0]]], [[[5.0]]], [[[6.0]]]])
    output = _execute_mocked(
        {"image0": first, "image1": second},
        _execution_config(spatial=True, consensus=False),
    )

    assert len(encoded) == 6
    assert [count for count, _ in fused] == [2, 2, 2]
    assert blended == []
    assert output[0][0].shape[0] == 3
    assert output[0][0].flatten().tolist() == [1.0, 2.0, 3.0]


def test_multiple_batches_contribute_every_source_to_each_spatial_lane(monkeypatch):
    first = torch.tensor([[[[1.0]]], [[[2.0]]]])
    second = torch.tensor([[[[10.0]]], [[[20.0]]]])
    config = _execution_config(spatial=True, consensus=False)
    config["visual"].update(
        {
            "visual_fusion_method": "spatial-checkerboard",
            "visual_block_size": 1,
            "dither_ratio": 0.5,
            "seed": 0,
            "dither_secondary_pattern": "checkerboard",
            "dither_mask_cleanup": False,
            "spatial_perturbation": 0.0,
        }
    )

    output = _execute_source_distinct(
        monkeypatch, {"image0": first, "image1": second}, config
    )
    visual = output[0][0][:, 1:5, 0]

    assert visual.shape == (2, 4)
    assert set(visual[0].tolist()) == {1.0, 10.0}
    assert set(visual[1].tolist()) == {2.0, 20.0}
    assert not set(visual[0].tolist()) & {2.0, 20.0}
    assert not set(visual[1].tolist()) & {1.0, 10.0}


def test_integrated_consensus_matches_standalone_complete_conditioning(monkeypatch):
    first = torch.tensor([[[[1.0]]], [[[2.0]]]])
    second = torch.tensor([[[[10.0]]], [[[20.0]]]])
    config = _execution_config(spatial=True, consensus=True)
    config["visual"].update(
        {
            "visual_fusion_method": "spatial-checkerboard",
            "visual_block_size": 1,
            "dither_ratio": 0.5,
            "seed": 0,
            "dither_secondary_pattern": "checkerboard",
            "dither_mask_cleanup": False,
            "spatial_perturbation": 0.0,
        }
    )
    config["consensus"] = {
        "blend_preset": "custom",
        "blend_method": "linear",
        "global_scale": 1.0,
        "resolution_samples": 1,
    }
    completed = []
    real_blend = encoder_helpers.blend_complete_conditionings

    def record_and_blend(conditionings, blend_config):
        completed.extend(conditionings)
        return real_blend(conditionings, blend_config)

    monkeypatch.setattr(
        encoder_helpers, "blend_complete_conditionings", record_and_blend
    )
    integrated = _execute_source_distinct(
        monkeypatch, {"image0": first, "image1": second}, config
    )
    standalone = UC_ConditioningConsensusBlend.execute(
        {f"conditioning_{index}": value for index, value in enumerate(completed)},
        config["consensus"],
    ).result[0]

    assert len(completed) == 2
    assert torch.equal(integrated[0][0], standalone[0][0])
    assert integrated[0][0].dtype == standalone[0][0].dtype == torch.float32
    assert torch.equal(
        integrated[0][1]["pooled_output"], standalone[0][1]["pooled_output"]
    )
    assert integrated[0][1]["shared"] == standalone[0][1]["shared"] == "metadata"
