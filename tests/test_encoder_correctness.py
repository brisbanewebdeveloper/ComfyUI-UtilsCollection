import inspect
import pathlib
import sys
import types
from fractions import Fraction

import pytest
import numpy as np
import torch


CUSTOM_NODE_ROOT = pathlib.Path(__file__).parents[1]
PACKAGE_NAME = "utils_collection_encoder_test"
package = types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(CUSTOM_NODE_ROOT)]
sys.modules.setdefault(PACKAGE_NAME, package)

from comfy.cli_args import args as cli_args
import comfy.model_base
import comfy.patcher_extension

prior_cpu = cli_args.cpu
cli_args.cpu = True
try:
    from utils_collection_encoder_test import encoder_helpers, encoder_nodes
    from utils_collection_encoder_test.encoder_nodes import (
        TextEncodeKrea2SystemEditScaledAdv,
        TextEncodeKrea2SysEditScaledAdvAttn,
        UC_AdvancedMiniMaxH3ImageToVideo,
        UC_AdvancedMiniMaxH3ImageToVideoCombined,
        UC_MiniMaxH3MediaConfig,
        UC_MiniMaxH3FirstFrameReferences,
        UC_AdvancedVisualConditioningEncode,
        UC_AttentionBiasTextEncode,
        UC_ConditioningConsensusBlend,
        UC_TextConsensusBlendConfig,
        UC_Krea2TokenAttentionWeight,
        UC_Qwen3VLInputEmbeds,
        UC_VisualFusionConfig,
        UC_VLMInputEmbeds,
    )
finally:
    cli_args.cpu = prior_cpu


VAE_MULTIPLE_ENCODERS = (
    "UC_ScaledBiasTextEncodeLtxv2SystemPrompt",
    "TextEncodeSystemEditPlus",
    "TextEncodeSystemEditPlusAdvanced",
    "TextEncodeKrea2SystemEditPlusAdvanced",
    "TextEncodeEditPlusAdvanced",
    "TextEncodeGemmaSystemEditPlusAdvanced",
    "UC_TextEncodeLtxv2SystemPrompt",
    "TextEncodeKrea2SystemEditScaledAdv",
    "TextEncodeKrea2SysEditScaledAdvAttn",
)


def test_power_blend_preset_matches_declared_widget_values():
    preset_input = next(value for value in UC_TextConsensusBlendConfig.define_schema().inputs if value.id == "blend_preset")

    assert "power_blend" in preset_input.options
    assert encoder_helpers.POWER_BLEND_PRESET == {
        "method": "consensus",
        "type": "median",
        "align": "similarity",
        "alignment_threshold": 0.9,
        "thresh": 0.75,
        "alpha": 8.0,
        "beta": 0.0,
        "norm": True,
        "scale": 1.0,
        "dsc": True,
        "soft_comfort": False,
    }


def test_text_blend_config_exposes_position_and_prefix_controls():
    inputs = {value.id: value for value in UC_TextConsensusBlendConfig.define_schema().inputs}
    assert inputs["position_weight"].default == 0.0
    assert inputs["preserve_common_prefix"].default is False
    config = UC_TextConsensusBlendConfig.execute(
        "power_blend", "consensus", "median", "similarity", 0.4, 0.0,
        2.0, 0.0, True, 1.0, position_weight=0.75,
        preserve_common_prefix=True,
    ).args[0]
    assert config["position_weight"] == 0.75
    assert config["preserve_common_prefix"] is True


def test_vae_reference_image_uses_configurable_dimension_multiple():
    samples = torch.zeros(1, 3, 101, 205)

    original = encoder_helpers.prepare_vae_reference_image(samples, None, 32)
    targeted = encoder_helpers.prepare_vae_reference_image(samples, 1024, 64)

    assert original.shape[-2:] == (96, 192)
    assert targeted.shape[-2] % 64 == 0
    assert targeted.shape[-1] % 64 == 0
    with pytest.raises(ValueError, match="at least 4"):
        encoder_helpers.prepare_vae_reference_image(samples, None, 3)


@pytest.mark.parametrize("mode", ["single", "parallel-single"])
def test_krea2_reference_latents_are_attached_to_conditioning(mode):
    krea2_stage = type("Krea2CLIP", (), {})()
    clip = types.SimpleNamespace(cond_stage_model=krea2_stage)
    reference = torch.zeros(1, 16, 8, 8)
    conditioning = [[torch.zeros(1, 2, 4), {}]]

    result = encoder_nodes.apply_parallel_ref_latents(
        clip, conditioning, [reference], mode
    )

    assert len(result) == 1
    assert result[0][1]["reference_latents"] == [reference]


def test_reference_latent_encoders_append_configurable_multiple():
    for class_name in VAE_MULTIPLE_ENCODERS:
        schema = getattr(encoder_nodes, class_name).define_schema()
        control = next(value for value in schema.inputs if value.id == "vae_dimension_multiple")
        assert control.default == 8
        assert control.min == 4
        assert control.step == 4
        assert control.advanced


def test_advanced_visual_encoder_applies_non_default_vae_alignment(monkeypatch):
    encoded_shapes = []

    class VAE:
        @staticmethod
        def encode(image):
            encoded_shapes.append(tuple(image.shape))
            return torch.zeros(1, 4, 1, 1)

    monkeypatch.setattr(
        encoder_nodes, "prepare_vlm_image", lambda image, _resolution: image
    )
    monkeypatch.setattr(
        encoder_nodes,
        "encode_embedding_classical_scaled_bias",
        lambda *_args, **_kwargs: [[torch.ones(1, 2, 3), {}]],
    )

    UC_AdvancedVisualConditioningEncode.execute(
        types.SimpleNamespace(),
        prompt="",
        system_prompt="",
        vlm_resolution=384,
        image_inputs={"image_1": torch.zeros(1, 101, 205, 3)},
        visual_fusion_config={"visual_fusion_method": "off"},
        vae_resolution="Original",
        ref_latent_mode="single",
        vae=VAE(),
        vae_dimension_multiple=32,
    )

    assert encoded_shapes == [(1, 96, 192, 3)]


def test_expression_grammar_and_nonfinite_rejection():
    value = torch.tensor([1.0, 2.0])
    assert torch.equal(encoder_helpers.evaluate_tensor_expression("clamp(a * 2, 0, 3)", {"a": value}), torch.tensor([2.0, 3.0]))
    with pytest.raises(ValueError, match="Unsupported expression element"):
        encoder_helpers.evaluate_tensor_expression("a.__class__", {"a": value})
    with pytest.raises(ValueError, match="NaN or infinite"):
        encoder_helpers.evaluate_tensor_expression("a / 0", {"a": value})


def test_visual_fusion_config_selects_real_encoder_path():
    config = UC_VisualFusionConfig.execute(
        visual_fusion_method="spatial-checkerboard",
        visual_block_size=2,
        dither_ratio=0.5,
        seed=0,
        visual_encoder_path="legacy-flat",
    )[0]
    assert config["visual_encoder_path"] == "legacy-flat"


def test_legacy_flat_temporarily_disables_grid_and_deepstack_inputs():
    class Transformer:
        @staticmethod
        def build_image_inputs(embeds, embeds_info):
            return "grid", "mask", "deepstack"

    transformer = Transformer()
    clip = types.SimpleNamespace(
        cond_stage_model=types.SimpleNamespace(
            clip_model=types.SimpleNamespace(transformer=transformer),
        ),
    )

    with encoder_helpers.qwen3vl_visual_encoder_path(clip, "legacy-flat"):
        assert transformer.build_image_inputs(None, None) == (None, None, None)

    assert transformer.build_image_inputs(None, None) == ("grid", "mask", "deepstack")


def test_inline_image_placeholders_honor_legacy_flat_encoder_path():
    class Transformer:
        @staticmethod
        def build_image_inputs(embeds, embeds_info):
            return "grid", "mask", "deepstack"

    transformer = Transformer()

    class Clip:
        cond_stage_model = types.SimpleNamespace(
            clip_model=types.SimpleNamespace(transformer=transformer),
        )

        @staticmethod
        def tokenize(*_args, **_kwargs):
            assert transformer.build_image_inputs(None, None) == (None, None, None)
            return {"fake": [[(1, 1.0)]]}

        @staticmethod
        def encode_from_tokens_scheduled(_tokens):
            assert transformer.build_image_inputs(None, None) == (None, None, None)
            return [[torch.ones(1, 1, 1), {}]]

    UC_AdvancedVisualConditioningEncode.execute(
        Clip(),
        prompt="image_input_1",
        system_prompt="",
        vlm_resolution=0,
        image_inputs={"image_1": torch.ones(1, 2, 2, 3)},
        visual_fusion_config={
            "visual_fusion_method": "off",
            "visual_encoder_path": "legacy-flat",
        },
    )

    assert transformer.build_image_inputs(None, None) == ("grid", "mask", "deepstack")


class _MiniMaxH3TestClip:
    def __init__(self):
        self.encoded_tokens = []
        self.tokenize_calls = []

    @staticmethod
    def _text_entries(text):
        return [] if not text else [(text, 1.0)]

    def tokenize(self, text, images=None, minimax_ref_items=None, **_kwargs):
        self.tokenize_calls.append(
            {
                "text": text,
                "images": images,
                "minimax_ref_items": minimax_ref_items,
            }
        )
        if minimax_ref_items is not None:
            entries = []
            for item in minimax_ref_items:
                if item["type"] == "image":
                    entries.extend(self._text_entries("<Picture 1>: "))
                    entries.extend([(151652, 1.0), ({"type": "image", "data": item["data"]}, 1.0), (151653, 1.0)])
                elif item["type"] == "video":
                    entries.extend(self._text_entries("<Video 1>: "))
                    frames = item["data"]
                    timestamps = list(item["timestamps"])
                    if frames.shape[0] % 2 == 1:
                        frames = torch.cat([frames, frames[-1:]], dim=0)
                        timestamps.append(timestamps[-1])
                    for index in range(0, frames.shape[0], 2):
                        timestamp = (timestamps[index] + timestamps[index + 1]) / 2
                        entries.extend(self._text_entries(f"<{float(timestamp):.1f} seconds>"))
                        entries.extend([(151652, 1.0), ({"type": "image", "data": frames[index:index + 2], "minimax_video_block": True}, 1.0), (151653, 1.0)])
            entries.extend(self._text_entries(text))
            return {"qwen3vl_32b": [entries]}
        entries = []
        for index, image in enumerate(images or []):
            entries.extend(self._text_entries(f"<Picture {index + 1}>: "))
            entries.extend(
                [
                    (151652, 1.0),
                    ({"type": "image", "data": image}, 1.0),
                    (151653, 1.0),
                ]
            )
        entries.extend(self._text_entries(text))
        if not entries:
            entries = [(151643, 1.0)]
        return {"qwen3vl_32b": [entries]}

    def encode_from_tokens_scheduled(self, tokens):
        self.encoded_tokens.append(tokens)
        entries = tokens["qwen3vl_32b"][0]
        tag_values = []
        for entry in entries:
            span = (
                encoder_helpers._qwen3vl_image_span(entry)
                if encoder_helpers.is_image_token(entry)
                else 1
            )
            tag_values.extend(
                [0 if encoder_helpers.is_image_token(entry) or entry[0] in (151652, 151653) else 1] * span
            )
        length = len(tag_values)
        tags = torch.tensor(tag_values)
        return [[torch.ones(1, length, 4), {"minimax_token_tags": tags}]]


class _RecordingMiniMaxVAE:
    def __init__(self):
        self.images = []

    def encode(self, image):
        self.images.append(image)
        return torch.full((1, 4, 1, 1), float(image.mean()))


class _MiniMaxH3TestPatcher:
    def __init__(self, model=None):
        self.model = model or object.__new__(comfy.model_base.MiniMaxH3)
        self.wrappers = {}
        self.clone_calls = 0

    def clone(self):
        self.clone_calls += 1
        cloned = _MiniMaxH3TestPatcher(self.model)
        cloned.wrappers = {
            wrapper_type: {
                key: values.copy() for key, values in keyed.items()
            }
            for wrapper_type, keyed in self.wrappers.items()
        }
        return cloned

    def add_wrapper_with_key(self, wrapper_type, key, wrapper):
        self.wrappers.setdefault(wrapper_type, {}).setdefault(key, []).append(wrapper)

    def remove_wrappers_with_key(self, wrapper_type, key):
        self.wrappers.get(wrapper_type, {}).pop(key, None)

    def get_wrappers(self, wrapper_type, key):
        return self.wrappers.get(wrapper_type, {}).get(key, [])


def test_minimax_h3_prompt_tokens_preserve_inline_order_and_raw_syntax():
    clip = _MiniMaxH3TestClip()
    first = torch.tensor([1.0])
    second = torch.tensor([2.0])
    prompt = encoder_helpers.format_minimax_h3_prompt(
        f"first {encoder_helpers.VISION_BLOCK} then {encoder_helpers.VISION_BLOCK}",
        "system",
    )

    tokens = encoder_helpers.tokenize_minimax_h3_prompt(
        clip, prompt, [second, first]
    )["qwen3vl_32b"][0]
    image_entries = [entry[0]["data"] for entry in tokens if encoder_helpers.is_image_token(entry)]
    text = "".join(entry[0] for entry in tokens if isinstance(entry[0], str))

    assert torch.equal(image_entries[0], second)
    assert torch.equal(image_entries[1], first)
    assert text == "system\nfirst <Picture 1>:  then <Picture 2>: "
    assert "<|im_start|>" not in text


def test_minimax_h3_implicit_picture_stays_before_system_text():
    formatted = encoder_helpers.format_minimax_h3_prompt(
        encoder_helpers.VISION_BLOCK + "prompt", "system"
    )
    assert formatted == encoder_helpers.VISION_BLOCK + "system\nprompt"


def test_advanced_visual_encoder_uses_minimax_inline_picture_syntax(monkeypatch):
    clip = _MiniMaxH3TestClip()
    first = torch.ones(1, 2, 2, 3)
    second = torch.full((1, 2, 2, 3), 2.0)
    monkeypatch.setattr(encoder_nodes, "prepare_vlm_image", lambda image, _resolution: image)

    output = UC_AdvancedVisualConditioningEncode.execute(
        clip,
        prompt="second image_input_2 then image_input_1",
        system_prompt="system",
        vlm_resolution=0,
        image_inputs={"image_1": first, "image_2": second},
        visual_fusion_config={"visual_fusion_method": "off"},
    ).args[0]

    entries = clip.encoded_tokens[-1]["qwen3vl_32b"][0]
    images = [entry[0]["data"] for entry in entries if encoder_helpers.is_image_token(entry)]
    text = "".join(entry[0] for entry in entries if isinstance(entry[0], str))
    assert torch.equal(images[0], second)
    assert torch.equal(images[1], first)
    assert "<Picture 1>: " in text and "<Picture 2>: " in text
    assert "<|im_start|>" not in text
    assert output[0][1]["minimax_token_tags"].numel() == output[0][0].shape[1]


def test_advanced_visual_encoder_rejects_generic_minimax_reference_latents():
    with pytest.raises(ValueError, match="MiniMax H3 reference latents require Core"):
        UC_AdvancedVisualConditioningEncode.execute(
            _MiniMaxH3TestClip(),
            prompt="prompt",
            system_prompt="",
            vlm_resolution=0,
            image_inputs={},
            visual_fusion_config={"visual_fusion_method": "off"},
            ref_latent_mode="single",
        )


def test_advanced_minimax_h3_node_schema_separates_visual_roles():
    schema = UC_AdvancedMiniMaxH3ImageToVideo.define_schema()
    inputs = {value.id: value for value in schema.inputs}

    assert schema.node_id == "UC_AdvancedMiniMaxH3ImageToVideo"
    assert schema.display_name == "Advanced MiniMax H3 Image to Video"
    assert [value.id for value in schema.inputs] == [
        "clip",
        "vae",
        "first_frame",
        "last_frame",
        "prompt",
        "width",
        "height",
        "length",
        "visual_fusion_config",
        "multiplier",
        "ref_image_size",
        "vlm_resolution",
        "vlm_video_resolution",
        "reference_images",
        "fusion_images",
        "media_config",
        "video",
        "audio",
        "audio_vae",
    ]
    assert inputs["vae"].optional is True
    assert inputs["first_frame"].optional is True
    assert inputs["last_frame"].optional is True
    assert inputs["video"].optional is True
    assert inputs["audio"].optional is True
    assert inputs["audio_vae"].optional is True
    assert "system_prompt" not in inputs
    assert "keyframe_mode" not in inputs
    assert inputs["reference_images"].template.names == [
        f"reference_image_{index}" for index in range(1, 33)
    ]
    assert inputs["fusion_images"].template.names == [
        f"fusion_image_{index}" for index in range(1, 33)
    ]
    assert inputs["reference_images"].template.min == 0
    assert inputs["fusion_images"].template.min == 0
    assert inputs["ref_image_size"].options == ["match", "max", "none"]
    assert inputs["ref_image_size"].default == "match"
    assert inputs["vlm_resolution"].default == 384
    assert "independent" in inputs["vlm_resolution"].tooltip.lower()
    assert inputs["vlm_video_resolution"].default == 384
    assert "more visual tokens" in inputs["vlm_video_resolution"].tooltip
    fusion_tooltip = inputs["fusion_images"].tooltip
    assert "socket N targets Picture N" in fusion_tooltip
    assert "broadcasts to every reference Picture" in fusion_tooltip
    assert "flattened fusion images pair by index" in fusion_tooltip
    assert "Without frames or references" in fusion_tooltip
    assert "native-reference mode ignores them" in fusion_tooltip
    assert "Video blocks are never fusion targets" in fusion_tooltip
    assert "cannot be combined with explicit first/last frame inputs" in (
        inputs["reference_images"].tooltip
    )
    assert [output.display_name for output in schema.outputs] == [
        "positive",
        None,
    ]


def test_minimax_h3_media_config_schema_and_payload():
    schema = UC_MiniMaxH3MediaConfig.define_schema()
    inputs = {value.id: value for value in schema.inputs}
    assert schema.is_input_list is True
    assert inputs["timestamps"].optional is True
    assert inputs["timestamp_format"].default == "0.0s"
    assert inputs["structure"].default == encoder_helpers.MINIMAX_H3_MEDIA_STRUCTURE
    assert inputs["structure"].default == "<<picture>>: <<visual>>"
    assert "At <<time>>, <<picture>>: <<visual>> (from <<shot>>)" in inputs["structure"].tooltip
    assert inputs["video_fps"].default == 2
    assert inputs["video_fps"].min == 1
    assert inputs["video_fps"].max == 24
    assert "video_structure" not in inputs
    assert "audio" not in inputs
    assert "audio_vae" not in inputs
    assert "video_images" not in inputs
    payload = encoder_helpers.build_minimax_h3_media_config([["0; 1.2"]], video_fps=[5])
    assert payload["schema_version"] == 3
    assert payload["timestamps_seconds"] == (Fraction(0), Fraction(6, 5))
    assert payload["video_fps"] == 5
    assert payload["timestamp_format"] == "0.0s"
    assert payload["structure"] == encoder_helpers.MINIMAX_H3_MEDIA_STRUCTURE
    assert "video_structure" not in payload
    assert "audio" not in payload
    assert "audio_vae" not in payload
    default_payload = encoder_helpers.build_minimax_h3_media_config(None)
    assert default_payload["timestamps_seconds"] == (Fraction(0),)
    assert default_payload["default_single_visual"] is True


def test_minimax_h3_default_media_config_requires_one_visual():
    with pytest.raises(ValueError, match="requires exactly one visual source"):
        encoder_helpers.tokenize_minimax_h3_media_prompt(
            None,
            "prompt",
            [object(), object()],
            [Fraction(0)],
            "0.0s",
            encoder_helpers.MINIMAX_H3_MEDIA_STRUCTURE,
            default_single_visual=True,
        )


@pytest.mark.parametrize("video_fps", [0, 25, 2.5, True])
def test_minimax_h3_media_config_rejects_invalid_video_fps(video_fps):
    with pytest.raises(ValueError, match="video_fps"):
        encoder_helpers.build_minimax_h3_media_config(None, video_fps=video_fps)


def test_minimax_h3_media_config_rejects_timestamp_beyond_output_duration():
    config = {
        "schema_version": 3,
        "timestamps_seconds": (Fraction("1.01"),),
        "timestamp_format": "0.0s",
        "structure": encoder_helpers.MINIMAX_H3_MEDIA_STRUCTURE,
        "video_fps": 2,
    }
    with pytest.raises(ValueError, match="output duration"):
        encoder_helpers._validate_minimax_h3_media_config(config, 24)


def test_minimax_h3_audio_reference_matches_core_contract(monkeypatch):
    class AudioVAE:
        audio_sample_rate = 32000

        def encode(self, waveform):
            assert waveform.shape == (1, 8, 2)
            return torch.ones(1, 32, 2, 4)

    called = []
    monkeypatch.setattr(encoder_helpers.torchaudio.functional, "resample", lambda waveform, source, target: called.append((source, target)) or waveform)
    block = encoder_helpers._encode_minimax_h3_audio_reference(
        {"waveform": torch.ones(2, 2, 8), "sample_rate": 16000},
        AudioVAE(),
    )
    assert called == [(16000, 32000)]
    assert block["kind"] == "audio"
    assert block["ref_audio_t"] == 4


def test_minimax_h3_reference_video_matches_core_resize_trim_and_payload(monkeypatch):
    resized = []

    def upscale(samples, width, height, method, crop):
        resized.append((samples.shape, width, height, method, crop))
        return torch.zeros(samples.shape[0], samples.shape[1], height, width)

    class VideoVAE:
        def encode(self, frames):
            assert frames.shape == (22, 96, 192, 3)
            return torch.ones(1, 4, 3, 6, 12)

    monkeypatch.setattr(encoder_helpers.comfy.utils, "common_upscale", upscale)
    frames, reference = encoder_helpers.prepare_minimax_h3_reference_video(
        torch.ones(30, 100, 200, 3), VideoVAE(), 23
    )
    assert frames.shape == (22, 96, 192, 3)
    assert resized == [((22, 3, 100, 200), 192, 96, "lanczos", "disabled")]
    assert reference["kind"] == "video"
    assert reference["latent_t"] == 3
    assert reference["latent_h"] == 6
    assert reference["latent_w"] == 12
    assert reference["ref_audio_t"] == 0
    assert reference["latent"].shape == (1, 4, 3, 6, 12)
    assert reference["audio_latent"] is None


def test_minimax_h3_reference_video_none_mode_skips_vae(monkeypatch):
    monkeypatch.setattr(
        encoder_helpers.comfy.utils,
        "common_upscale",
        lambda samples, _width, _height, _method, _crop: samples,
    )
    frames, reference = encoder_helpers.prepare_minimax_h3_reference_video(
        torch.ones(22, 64, 64, 3), None, 22, encode_reference=False
    )
    assert frames.shape == (22, 64, 64, 3)
    assert reference is None


@pytest.mark.parametrize("connect_media_config", [False, True])
def test_advanced_minimax_h3_media_config_uses_default_two_fps_presentation(
    connect_media_config,
):
    class VideoVAE:
        def encode(self, frames):
            return torch.ones(1, 4, 2, frames.shape[1] // 16, frames.shape[2] // 16)

    clip = _MiniMaxH3TestClip()
    conditioning, _latent = encoder_helpers.execute_advanced_minimax_h3_image_to_video(
        clip,
        VideoVAE(),
        "prompt",
        64,
        64,
        22,
        video=torch.ones(22, 64, 64, 3),
        media_config=(
            encoder_helpers.build_minimax_h3_media_config(None)
            if connect_media_config else None
        ),
    )
    video_calls = [
        call for call in clip.tokenize_calls
        if call["minimax_ref_items"]
        and call["minimax_ref_items"][0]["type"] == "video"
    ]
    assert len(video_calls) == 1
    video_item = video_calls[0]["minimax_ref_items"][0]
    assert video_item["data"].shape[0] == 2
    assert video_item["timestamps"] == [Fraction(0), Fraction(1, 2)]
    assert conditioning[0][1]["minimax_refs"][0]["kind"] == "video"


def test_minimax_h3_video_sample_indices_support_non_divisor_rates():
    assert encoder_helpers.minimax_h3_video_sample_indices(24, 2) == [0, 12]
    assert encoder_helpers.minimax_h3_video_sample_indices(24, 5) == [0, 5, 10, 14, 19]
    assert encoder_helpers.minimax_h3_video_sample_indices(24, 24) == list(range(24))


def test_advanced_minimax_h3_video_fps_samples_source_and_keeps_latent():
    class VideoVAE:
        def encode(self, frames):
            return torch.ones(1, 4, 2, frames.shape[1] // 16, frames.shape[2] // 16)

    clip = _MiniMaxH3TestClip()
    video = (torch.arange(24, dtype=torch.float32) / 23).view(24, 1, 1, 1).expand(24, 64, 64, 3)
    conditioning, _latent = encoder_helpers.execute_advanced_minimax_h3_image_to_video(
        clip,
        VideoVAE(),
        "prompt",
        64,
        64,
        22,
        video=video,
        media_config=encoder_helpers.build_minimax_h3_media_config(None, video_fps=5),
    )
    video_call = next(
        call for call in clip.tokenize_calls
        if call["minimax_ref_items"]
        and call["minimax_ref_items"][0]["type"] == "video"
    )
    video_item = video_call["minimax_ref_items"][0]
    assert video_item["data"].shape[0] == 6
    assert video_item["timestamps"] == [
        Fraction(0), Fraction(5, 24), Fraction(5, 12),
        Fraction(7, 12), Fraction(19, 24), Fraction(19, 24),
    ]
    assert conditioning[0][1]["minimax_refs"][0]["kind"] == "video"


@pytest.mark.parametrize("connect_media_config", [False, True])
def test_advanced_minimax_h3_video_qwen_frames_use_vlm_resolution(
    monkeypatch, connect_media_config,
):
    class VideoVAE:
        def encode(self, frames):
            return torch.ones(
                1, 4, 2, max(1, frames.shape[1] // 16), max(1, frames.shape[2] // 16)
            )

    calls = []

    def prepare(image, resolution):
        calls.append((tuple(image.shape), resolution))
        return image

    monkeypatch.setattr(encoder_helpers, "prepare_vlm_image", prepare)
    video = torch.ones(22, 64, 96, 3)
    media_config = (
        encoder_helpers.build_minimax_h3_media_config(None)
        if connect_media_config else None
    )
    encoder_helpers.execute_advanced_minimax_h3_image_to_video(
        _MiniMaxH3TestClip(),
        VideoVAE(),
        "prompt",
        64,
        64,
        22,
        video=video,
        media_config=media_config,
        vlm_resolution=256,
        vlm_video_resolution=512,
    )
    assert calls == [((1, 64, 96, 3), 512), ((1, 64, 96, 3), 512)]


@pytest.mark.parametrize("connect_media_config", [False, True])
def test_advanced_minimax_h3_keeps_reference_pictures_and_video_together(
    connect_media_config,
):
    class VideoVAE:
        def encode(self, frames):
            return torch.ones(
                1, 4, 2, max(1, frames.shape[1] // 16), max(1, frames.shape[2] // 16)
            )

    clip = _MiniMaxH3TestClip()
    reference = torch.full((1, 64, 64, 3), 0.25)
    video = torch.full((22, 64, 64, 3), 0.75)
    media_config = (
        encoder_helpers.build_minimax_h3_media_config([Fraction(0)])
        if connect_media_config else None
    )
    conditioning, _latent = encoder_helpers.execute_advanced_minimax_h3_image_to_video(
        clip,
        VideoVAE(),
        "prompt",
        64,
        64,
        22,
        reference_images={"reference_image_1": reference},
        video=video,
        media_config=media_config,
    )
    entries = clip.encoded_tokens[-1]["qwen3vl_32b"][0]
    text = "".join(entry[0] for entry in entries if isinstance(entry[0], str))
    assert text.index("<Picture 1>") < text.index("<Video 1>") < text.index("prompt")
    kinds = [reference["kind"] for reference in conditioning[0][1]["minimax_refs"]]
    assert kinds == ["image", "video"]


def test_advanced_combined_minimax_h3_wraps_keyframe_with_native_video():
    class VideoVAE:
        def encode(self, frames):
            return torch.ones(
                1, 4, 2, max(1, frames.shape[1] // 16), max(1, frames.shape[2] // 16)
            )

    model = _MiniMaxH3TestPatcher()
    output_model, conditioning, _latent = (
        UC_AdvancedMiniMaxH3ImageToVideoCombined.execute(
            model=model,
            clip=_MiniMaxH3TestClip(),
            vae=VideoVAE(),
            prompt="prompt",
            width=64,
            height=64,
            length=5,
            first_frame=torch.ones(1, 64, 64, 3),
            video=torch.ones(5, 64, 64, 3),
        ).args
    )
    assert output_model is not model
    assert model.clone_calls == 1
    assert [item["kind"] for item in conditioning[0][1]["minimax_refs"]] == [
        "video"
    ]


def test_minimax_h3_media_tokenization_default_matches_core_picture_constructor():
    clip = _MiniMaxH3TestClip()
    tokens = encoder_helpers.tokenize_minimax_h3_media_prompt(
        clip,
        "prompt",
        [torch.ones(1, 2, 2, 3)],
        [Fraction(0)],
        "0.00s",
        encoder_helpers.MINIMAX_H3_MEDIA_STRUCTURE,
    )
    entries = tokens["qwen3vl_32b"][0]
    text = "".join(entry[0] for entry in entries if isinstance(entry[0], str))
    assert text == "<Picture 1>: \nprompt"


def test_minimax_h3_media_tokenization_builds_picture_anchors_before_prompt():
    clip = _MiniMaxH3TestClip()
    first = torch.ones(1, 2, 3, 3)
    second = torch.ones(1, 3, 2, 3)
    tokens = encoder_helpers.tokenize_minimax_h3_media_prompt(
        clip,
        "prompt",
        [first, second],
        [Fraction("1.21"), Fraction("2.46")],
        "0.00s",
        "At <<time>>, <<picture>>: <<visual>> (from <<shot>>) is fully anchored.",
    )
    entries = tokens["qwen3vl_32b"][0]
    text = "".join(entry[0] for entry in entries if isinstance(entry[0], str))
    visuals = [entry[0] for entry in entries if encoder_helpers.is_image_token(entry)]
    assert "At 1.21s, <Picture 1>: " in text
    assert "(from [Shot 1]) is fully anchored." in text
    assert "At 2.46s, <Picture 2>: " in text
    assert "(from [Shot 2]) is fully anchored." in text
    assert text.index("At 1.21s") < text.index("At 2.46s") < text.index("prompt")
    assert visuals[0]["data"].shape == (1, 2, 3, 3)
    assert visuals[1]["data"].shape == (1, 3, 2, 3)
    image_calls = [call for call in clip.tokenize_calls if call["images"] is not None]
    assert len(image_calls) == 1
    assert len(image_calls[0]["images"]) == 2
    conditioning = clip.encode_from_tokens_scheduled(tokens)
    tensor, metadata = conditioning[0]
    assert metadata["minimax_token_tags"].numel() == tensor.shape[1]


def test_minimax_h3_media_tokenization_leaves_unanchored_pictures_once():
    clip = _MiniMaxH3TestClip()
    base = torch.zeros(1, 2, 2, 3)
    shot = torch.ones(1, 2, 2, 3)
    tokens = encoder_helpers.tokenize_minimax_h3_media_prompt(
        clip,
        "prompt",
        [base, shot],
        [Fraction("61.25")],
        "MM:SS.mmm",
        "<<shot>> @ <<time>> uses <<picture>> = <<visual>>",
    )
    entries = tokens["qwen3vl_32b"][0]
    text = "".join(entry[0] for entry in entries if isinstance(entry[0], str))
    assert "[Shot 1] @ 01:01.250 uses <Picture 1> = " in text
    assert "<Picture 2>: " in text
    visuals = [entry[0] for entry in entries if encoder_helpers.is_image_token(entry)]
    assert len(visuals) == 2


def test_minimax_h3_media_tokenization_rejects_more_timestamps_than_pictures():
    with pytest.raises(ValueError, match="2 timestamps for 1 available Pictures"):
        encoder_helpers.tokenize_minimax_h3_media_prompt(
            _MiniMaxH3TestClip(),
            "prompt",
            [torch.ones(1, 2, 2, 3)],
            [Fraction(0), Fraction(1)],
            "0.0s",
            encoder_helpers.MINIMAX_H3_MEDIA_STRUCTURE,
        )


def test_minimax_h3_media_tokenization_combines_picture_and_restricted_video():
    clip = _MiniMaxH3TestClip()
    picture = torch.ones(1, 2, 2, 3)
    frames = [
        torch.full((1, 2, 2, 3), 2.0),
        torch.full((1, 2, 2, 3), 3.0),
    ]
    tokens = encoder_helpers.tokenize_minimax_h3_media_prompt(
        clip,
        "prompt",
        [picture],
        [Fraction(0)],
        "0.00s",
        "At <<time>>, <<picture>>: <<visual>> (from <<shot>>) is fully anchored.",
        video_frames=frames,
        video_timestamps=[Fraction(1), Fraction(2)],
    )
    entries = tokens["qwen3vl_32b"][0]
    text = "".join(entry[0] for entry in entries if isinstance(entry[0], str))
    visuals = [entry[0] for entry in entries if encoder_helpers.is_image_token(entry)]
    assert "At 0.00s, <Picture 1>: " in text
    assert text.count("<Video 1>: ") == 1
    assert "<1.50s>" in text
    assert text.index("<Picture 1>") < text.index("<Video 1>") < text.index("prompt")
    assert visuals[1]["minimax_video_block"] is True
    assert visuals[1]["data"].shape[0] == 2


def test_minimax_h3_media_tokenization_formats_derived_video_timestamps():
    clip = _MiniMaxH3TestClip()
    frames = torch.ones(3, 2, 2, 3)
    tokens = encoder_helpers.tokenize_minimax_h3_media_prompt(
        clip,
        "prompt",
        [],
        [Fraction(0)],
        "0.0s",
        encoder_helpers.MINIMAX_H3_MEDIA_STRUCTURE,
        default_single_visual=True,
        default_video_frames=frames,
        default_video_timestamps=[Fraction(0), Fraction(1, 2), Fraction(1)],
    )
    entries = tokens["qwen3vl_32b"][0]
    text = "".join(entry[0] for entry in entries if isinstance(entry[0], str))
    visuals = [
        entry[0] for entry in entries if encoder_helpers.is_image_token(entry)
    ]
    assert "<Video 1>: " in text
    assert text.index("<Video 1>") < text.index("prompt")
    assert "<0.3s>" in text
    assert "<1.0s>" in text
    assert len(visuals) == 2
    assert torch.equal(visuals[0]["data"], frames[:2])
    assert torch.equal(visuals[1]["data"], frames[-1:].expand(2, -1, -1, -1))
    assert all(visual["minimax_video_block"] is True for visual in visuals)


@pytest.mark.parametrize(
    "structure, message",
    [
        ("", "must not be empty"),
        ("<<picture>> <<shot>>", "missing <<visual>>"),
        ("<<visual>> <<shot>>", "missing <<picture>>"),
        ("<<time>> <<picture>> <<visual>> <<shot>> <<unknown>>", "Unknown"),
        ("<<time>> <<picture>> <<visual>> <<visual>> <<shot>>", "exactly one <<visual>>"),
    ],
)
def test_minimax_h3_media_structure_validation(structure, message):
    with pytest.raises(ValueError, match=message):
        encoder_helpers._validate_minimax_h3_media_structure(structure)


def test_minimax_h3_media_structure_allows_optional_time_and_shot_labels():
    structure = "<<picture>>: <<visual>>"
    assert encoder_helpers._validate_minimax_h3_media_structure(structure) == structure


def test_advanced_combined_minimax_h3_schema_is_additive():
    schema = UC_AdvancedMiniMaxH3ImageToVideoCombined.define_schema()
    inputs = {value.id: value for value in schema.inputs}

    assert schema.node_id == "UC_AdvancedMiniMaxH3ImageToVideoCombined"
    assert schema.display_name == "Advanced MiniMax H3 Image to Video (Combined)"
    assert [value.id for value in schema.inputs] == [
        "model",
        "clip",
        "vae",
        "first_frame",
        "last_frame",
        "prompt",
        "width",
        "height",
        "length",
        "visual_fusion_config",
        "multiplier",
        "ref_image_size",
        "vlm_resolution",
        "vlm_video_resolution",
        "reference_images",
        "fusion_images",
        "media_config",
        "video",
        "audio",
        "audio_vae",
    ]
    assert inputs["ref_image_size"].options == [
        "match",
        "max",
        "none",
        "first + match",
        "first + max",
        "first + last + match",
        "first + last + max",
    ]
    assert inputs["ref_image_size"].default == "match"
    assert inputs["reference_images"].template.names == [
        f"reference_image_{index}" for index in range(1, 33)
    ]
    assert [output.display_name for output in schema.outputs] == [
        "model",
        "positive",
        None,
    ]


def test_combined_minimax_h3_node_schema_is_additive_and_one_based():
    schema = UC_MiniMaxH3FirstFrameReferences.define_schema()
    inputs = {value.id: value for value in schema.inputs}

    assert schema.node_id == "UC_MiniMaxH3FirstFrameReferences"
    assert schema.display_name == "MiniMax H3 First/Last Frame + References"
    assert schema.category == "model/conditioning/minimax"
    assert [value.id for value in schema.inputs] == [
        "model",
        "clip",
        "vae",
        "first_frame",
        "last_frame",
        "prompt",
        "width",
        "height",
        "length",
        "ref_image_size",
        "vlm_resolution",
        "reference_images",
    ]
    assert inputs["reference_images"].template.names == [
        f"reference_image_{index}" for index in range(1, 17)
    ]
    assert inputs["reference_images"].optional is True
    assert inputs["reference_images"].template.min == 0
    assert inputs["ref_image_size"].options == ["match", "max"]
    assert inputs["ref_image_size"].default == "match"
    assert inputs["vlm_resolution"].default == 384
    assert "independent" in inputs["vlm_resolution"].tooltip.lower()
    assert inputs["length"].default == 124
    assert [output.display_name for output in schema.outputs] == [
        "model",
        "positive",
        None,
    ]


def test_combined_minimax_h3_native_order_metadata_and_model_patch():
    model = _MiniMaxH3TestPatcher()
    wrapper_type = comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL
    unrelated = object()
    model.wrappers = {wrapper_type: {"unrelated": [unrelated]}}
    clip = _MiniMaxH3TestClip()
    vae = _RecordingMiniMaxVAE()
    first = torch.full((1, 32, 64, 3), 0.125)
    second = torch.full((1, 32, 64, 3), 0.25)
    third = torch.full((1, 32, 64, 3), 0.5)
    fourth = torch.full((1, 64, 32, 3), 0.75)
    last = torch.full((1, 32, 64, 3), 0.875)

    patched, conditioning, latent = UC_MiniMaxH3FirstFrameReferences.execute(
        model,
        clip,
        vae,
        first,
        "subject",
        64,
        32,
        5,
        "match",
        {
            "reference_image_2": fourth,
            "reference_image_1": torch.cat([second, third], dim=0),
        },
        last,
        256,
    ).args

    native_call = clip.tokenize_calls[-1]
    presented = [item["data"] for item in native_call["minimax_ref_items"]]
    assert native_call["images"] is None
    assert native_call["text"] == "subject"
    assert [float(image.mean()) for image in presented] == pytest.approx(
        [0.125, 0.875, 0.25, 0.5, 0.75], abs=0.004
    )
    assert len(vae.images) == 5
    assert [image.shape[1:3] for image in presented] == [
        encoder_helpers.vlm_target_dimensions(image.shape[1], image.shape[2], 256)
        for image in [first, last, second, third, fourth]
    ]
    assert all(qwen is not encoded for qwen, encoded in zip(presented, vae.images))

    tensor, metadata = conditioning[0]
    assert torch.all(tensor == 1.0)
    assert metadata["minimax_frame_count"] == 5
    assert [item["resolved_frame_index"] for item in metadata["minimax_keyframes"]] == [0, 4]
    assert float(metadata["minimax_keyframes"][0]["latent"].mean()) == pytest.approx(
        float(vae.images[0].mean())
    )
    assert [float(item["latent"].mean()) for item in metadata["minimax_refs"]] == pytest.approx(
        [0.25, 0.5, 0.75], abs=0.004
    )
    assert metadata["minimax_token_tags"].numel() == tensor.shape[1]
    video, audio = latent["samples"].tensors
    assert video.shape == (1, 24, 2, 2, 4)
    assert audio.shape == (1, 32, 2, 8)

    assert patched is not model
    assert model.clone_calls == 1
    assert model.get_wrappers(wrapper_type, "uc_minimax_h3_combined_visual_latents") == []
    assert patched.get_wrappers(wrapper_type, "unrelated") == [unrelated]
    assert patched.get_wrappers(
        wrapper_type, "uc_minimax_h3_combined_visual_latents"
    ) == [encoder_helpers.minimax_h3_combined_payload_wrapper]


def test_combined_minimax_h3_without_references_uses_native_keyframes():
    model = _MiniMaxH3TestPatcher()
    clip = _MiniMaxH3TestClip()
    vae = _RecordingMiniMaxVAE()
    first = torch.full((1, 32, 64, 3), 0.125)
    last = torch.full((1, 32, 64, 3), 0.875)

    output_model, conditioning, _latent = UC_MiniMaxH3FirstFrameReferences.execute(
        model,
        clip,
        vae,
        first,
        "subject",
        64,
        32,
        5,
        "match",
        None,
        last,
    ).args

    native_call = clip.tokenize_calls[-1]
    assert native_call["minimax_ref_items"] is None
    assert [float(image.mean()) for image in native_call["images"]] == pytest.approx(
        [0.125, 0.875], abs=0.004
    )
    metadata = conditioning[0][1]
    assert [item["resolved_frame_index"] for item in metadata["minimax_keyframes"]] == [0, 4]
    assert [float(item["latent"].mean()) for item in metadata["minimax_keyframes"]] == pytest.approx(
        [0.125, 0.875], abs=0.004
    )
    assert "minimax_refs" not in metadata
    assert output_model is model
    assert model.clone_calls == 0


def test_combined_minimax_h3_wrapper_repairs_rows_without_mutation():
    keyframe_latent = torch.tensor([1.0])
    reference_latent = torch.tensor([2.0])
    audio_latent = torch.tensor([3.0])
    layout = object()
    payload = {
        "keyframes": [{"latent": keyframe_latent}],
        "refs": [
            {"kind": "image", "latent": reference_latent},
            {"kind": "audio", "audio_latent": audio_latent},
        ],
        "cond_video_latents": [reference_latent],
        "cond_audio_latents": [audio_latent],
        "layout": layout,
        "seed": 9,
    }
    original_kwargs = {"minimax_payload": payload, "marker": object()}
    observed = {}

    def executor(*args, **kwargs):
        observed["args"] = args
        observed["kwargs"] = kwargs
        return "done"

    assert encoder_helpers.minimax_h3_combined_payload_wrapper(
        executor, "x", **original_kwargs
    ) == "done"
    updated_kwargs = observed["kwargs"]
    updated_payload = updated_kwargs["minimax_payload"]
    assert observed["args"] == ("x",)
    assert updated_kwargs is not original_kwargs
    assert updated_payload is not payload
    assert payload["cond_video_latents"] == [reference_latent]
    assert updated_payload["cond_video_latents"][0] is keyframe_latent
    assert updated_payload["cond_video_latents"][1] is reference_latent
    assert updated_payload["layout"] is layout
    assert updated_payload["cond_audio_latents"] is payload["cond_audio_latents"]
    assert updated_kwargs["marker"] is original_kwargs["marker"]


@pytest.mark.parametrize(
    "payload",
    [
        None,
        {},
        {"keyframes": [{"latent": torch.tensor([1.0])}]},
        {"refs": [{"kind": "image", "latent": torch.tensor([2.0])}]},
    ],
)
def test_combined_minimax_h3_wrapper_passes_single_modes_through(payload):
    kwargs = {"minimax_payload": payload}
    observed = {}

    def executor(**received):
        observed.update(received)
        return "unchanged"

    assert encoder_helpers.minimax_h3_combined_payload_wrapper(
        executor, **kwargs
    ) == "unchanged"
    assert observed["minimax_payload"] is payload


@pytest.mark.parametrize(
    "payload",
    [
        {"keyframes": [{}], "refs": [{"kind": "image", "latent": torch.ones(1)}]},
        {"keyframes": [{"latent": torch.ones(1)}], "refs": [{"kind": "image"}]},
        {"keyframes": [{"latent": torch.ones(1)}], "refs": [None]},
    ],
)
def test_combined_minimax_h3_wrapper_rejects_malformed_combined_payload(payload):
    with pytest.raises(ValueError, match="Malformed combined MiniMax H3"):
        encoder_helpers.minimax_h3_combined_payload_wrapper(
            lambda **_kwargs: None, minimax_payload=payload
        )


def test_combined_minimax_h3_validates_before_expensive_work():
    class WrongPatcher:
        model = object()

        @staticmethod
        def clone():
            raise AssertionError("model cloned before validation")

    clip = _MiniMaxH3TestClip()
    vae = _RecordingMiniMaxVAE()
    image = torch.ones(1, 32, 32, 3)
    with pytest.raises(ValueError, match="requires a MiniMax H3 model"):
        UC_MiniMaxH3FirstFrameReferences.execute(
            WrongPatcher(), clip, vae, image, "prompt", 32, 32, 5,
            "match", {"reference_image_1": image},
        )
    assert clip.tokenize_calls == []
    assert vae.images == []


def test_combined_minimax_h3_rejects_invalid_clip_images_and_controls_before_clone():
    model = _MiniMaxH3TestPatcher()
    image = torch.ones(1, 32, 32, 3)
    reference = {"reference_image_1": image}

    class WrongClip:
        tokenizer = types.SimpleNamespace(clip_name="qwen3vl_8b")

    with pytest.raises(ValueError, match="qwen3vl_32b"):
        UC_MiniMaxH3FirstFrameReferences.execute(
            model, WrongClip(), _RecordingMiniMaxVAE(), image, "prompt",
            32, 32, 5, "match", reference,
        )
    with pytest.raises(ValueError, match="first frame"):
        UC_MiniMaxH3FirstFrameReferences.execute(
            model, _MiniMaxH3TestClip(), _RecordingMiniMaxVAE(), image.repeat(2, 1, 1, 1),
            "prompt", 32, 32, 5, "match", reference,
        )
    with pytest.raises(ValueError, match="last frame"):
        UC_MiniMaxH3FirstFrameReferences.execute(
            model, _MiniMaxH3TestClip(), _RecordingMiniMaxVAE(), image,
            "prompt", 32, 32, 5, "match", reference, image.repeat(2, 1, 1, 1),
        )
    with pytest.raises(ValueError, match="reference image 1"):
        UC_MiniMaxH3FirstFrameReferences.execute(
            model, _MiniMaxH3TestClip(), _RecordingMiniMaxVAE(), image, "prompt",
            32, 32, 5, "match", {"reference_image_1": torch.ones(32, 32)},
        )
    with pytest.raises(ValueError, match="reference image size"):
        UC_MiniMaxH3FirstFrameReferences.execute(
            model, _MiniMaxH3TestClip(), _RecordingMiniMaxVAE(), image, "prompt",
            32, 32, 5, "hidden", reference,
        )
    with pytest.raises(ValueError, match="multiples of 32"):
        UC_MiniMaxH3FirstFrameReferences.execute(
            model, _MiniMaxH3TestClip(), _RecordingMiniMaxVAE(), image, "prompt",
            48, 32, 5, "match", reference,
        )
    assert model.clone_calls == 0


def test_combined_minimax_h3_patch_is_idempotent_and_preserves_unrelated_wrappers():
    wrapper_type = comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL
    model = _MiniMaxH3TestPatcher()
    unrelated = object()
    model.wrappers = {wrapper_type: {"unrelated": [unrelated]}}

    first_patch = encoder_helpers.patch_minimax_h3_combined_model(model)
    second_patch = encoder_helpers.patch_minimax_h3_combined_model(first_patch)

    assert first_patch.get_wrappers(
        wrapper_type, "uc_minimax_h3_combined_visual_latents"
    ) == [encoder_helpers.minimax_h3_combined_payload_wrapper]
    assert second_patch.get_wrappers(
        wrapper_type, "uc_minimax_h3_combined_visual_latents"
    ) == [encoder_helpers.minimax_h3_combined_payload_wrapper]
    assert second_patch.get_wrappers(wrapper_type, "unrelated") == [unrelated]


def test_combined_minimax_h3_preserves_dtype_and_pooled_output():
    class PooledClip(_MiniMaxH3TestClip):
        def encode_from_tokens_scheduled(self, tokens):
            conditioning = super().encode_from_tokens_scheduled(tokens)
            conditioning[0][0] = conditioning[0][0].to(torch.float16)
            conditioning[0][1]["pooled_output"] = torch.ones(
                1, 4, dtype=torch.float16
            )
            return conditioning

    image = torch.ones(1, 32, 32, 3)
    _model, conditioning, _latent = UC_MiniMaxH3FirstFrameReferences.execute(
        _MiniMaxH3TestPatcher(),
        PooledClip(),
        _RecordingMiniMaxVAE(),
        image,
        "prompt",
        32,
        32,
        5,
        "match",
        {"reference_image_1": image},
    ).args

    tensor, metadata = conditioning[0]
    assert tensor.dtype == torch.float16
    assert metadata["pooled_output"].dtype == torch.float16
    assert torch.all(tensor == 1.0)
    assert torch.all(metadata["pooled_output"] == 1.0)


@pytest.mark.parametrize(
    ("mode", "expected_keyframes", "expected_references", "expected_size_mode"),
    [
        ("first + match", [0.125], [0.25, 0.5, 0.75], "match"),
        ("first + max", [0.125], [0.25, 0.5, 0.75], "max"),
        ("first + last + match", [0.125, 0.75], [0.25, 0.5], "match"),
        ("first + last + max", [0.125, 0.75], [0.25, 0.5], "max"),
    ],
)
def test_advanced_combined_minimax_h3_routes_flattened_reference_endpoints_once(
    monkeypatch,
    mode,
    expected_keyframes,
    expected_references,
    expected_size_mode,
):
    prepared_size_modes = []

    def record_reference_size(image, _width, _height, size_mode):
        prepared_size_modes.append(size_mode)
        return image

    monkeypatch.setattr(
        encoder_helpers,
        "prepare_minimax_h3_reference_image",
        record_reference_size,
    )
    images = [
        torch.full((1, 32, 64, 3), value)
        for value in (0.125, 0.25, 0.5, 0.75)
    ]
    references = {
        "reference_image_2": images[3],
        "reference_image_1": torch.cat(images[:3], dim=0),
    }
    model = _MiniMaxH3TestPatcher()
    clip = _MiniMaxH3TestClip()
    vae = _RecordingMiniMaxVAE()

    output_model, conditioning, _latent = (
        UC_AdvancedMiniMaxH3ImageToVideoCombined.execute(
            model=model,
            clip=clip,
            vae=vae,
            prompt="subject",
            width=64,
            height=32,
            length=5,
            reference_images=references,
            ref_image_size=mode,
            vlm_resolution=0,
        ).args
    )

    presented = [
        item["data"] for item in clip.tokenize_calls[-1]["minimax_ref_items"]
    ]
    assert clip.tokenize_calls[-1]["images"] is None
    assert [float(image.mean()) for image in presented] == pytest.approx(
        [0.125, 0.25, 0.5, 0.75]
    )
    metadata = conditioning[0][1]
    assert [
        item["resolved_frame_index"] for item in metadata["minimax_keyframes"]
    ] == ([0] if len(expected_keyframes) == 1 else [0, 4])
    assert [
        float(item["latent"].mean()) for item in metadata["minimax_keyframes"]
    ] == pytest.approx(expected_keyframes, abs=0.004)
    assert [
        float(item["latent"].mean()) for item in metadata["minimax_refs"]
    ] == pytest.approx(expected_references)
    assert prepared_size_modes == [expected_size_mode] * len(expected_references)
    assert len(vae.images) == len(images)
    assert sorted(float(image.mean()) for image in vae.images) == pytest.approx(
        [0.125, 0.25, 0.5, 0.75], abs=0.004
    )
    assert output_model is not model
    assert model.clone_calls == 1


@pytest.mark.parametrize(
    ("mode", "values", "keyframe_indices", "reference_count", "needs_patch"),
    [
        ("first + match", [0.25], [0], 0, False),
        ("first + match", [0.25, 0.5], [0], 1, True),
        ("first + last + match", [0.25], [0], 0, False),
        ("first + last + match", [0.25, 0.5], [0, 4], 0, False),
        ("first + last + match", [0.25, 0.5, 0.75], [0, 4], 1, True),
    ],
)
def test_advanced_combined_minimax_h3_patches_only_mixed_payloads(
    mode,
    values,
    keyframe_indices,
    reference_count,
    needs_patch,
):
    model = _MiniMaxH3TestPatcher()
    images = [torch.full((1, 32, 32, 3), value) for value in values]

    output_model, conditioning, _latent = (
        UC_AdvancedMiniMaxH3ImageToVideoCombined.execute(
            model=model,
            clip=_MiniMaxH3TestClip(),
            vae=_RecordingMiniMaxVAE(),
            prompt="subject",
            width=32,
            height=32,
            length=5,
            reference_images={"reference_image_1": torch.cat(images, dim=0)},
            ref_image_size=mode,
            vlm_resolution=0,
        ).args
    )

    metadata = conditioning[0][1]
    assert [
        item["resolved_frame_index"] for item in metadata["minimax_keyframes"]
    ] == keyframe_indices
    assert len(metadata.get("minimax_refs", [])) == reference_count
    assert (output_model is not model) is needs_patch
    assert model.clone_calls == int(needs_patch)


def test_advanced_combined_minimax_h3_requires_references_for_hybrid_modes():
    model = _MiniMaxH3TestPatcher()
    clip = _MiniMaxH3TestClip()
    vae = _RecordingMiniMaxVAE()

    with pytest.raises(ValueError, match="requires at least one reference image"):
        UC_AdvancedMiniMaxH3ImageToVideoCombined.execute(
            model=model,
            clip=clip,
            vae=vae,
            prompt="subject",
            width=32,
            height=32,
            length=5,
            ref_image_size="first + match",
        )

    assert clip.encoded_tokens == []
    assert vae.images == []
    assert model.clone_calls == 0


def test_advanced_combined_minimax_h3_validates_model_before_encoding():
    class WrongPatcher:
        model = object()

        @staticmethod
        def clone():
            raise AssertionError("model cloned before validation")

    clip = _MiniMaxH3TestClip()
    vae = _RecordingMiniMaxVAE()
    image = torch.ones(1, 32, 32, 3)

    with pytest.raises(
        ValueError,
        match=r"Advanced MiniMax H3 Image to Video \(Combined\) requires a MiniMax H3 model",
    ):
        UC_AdvancedMiniMaxH3ImageToVideoCombined.execute(
            model=WrongPatcher(),
            clip=clip,
            vae=vae,
            prompt="subject",
            width=32,
            height=32,
            length=5,
            reference_images={"reference_image_1": image},
            ref_image_size="first + match",
        )

    assert clip.tokenize_calls == []
    assert vae.images == []


@pytest.mark.parametrize(
    ("extra_inputs", "message"),
    [
        ({"first_frame": torch.ones(1, 32, 32, 3)}, "frame inputs cannot be combined"),
    ],
)
def test_advanced_combined_minimax_h3_rejects_conflicting_hybrid_inputs(
    extra_inputs,
    message,
):
    model = _MiniMaxH3TestPatcher()
    clip = _MiniMaxH3TestClip()
    vae = _RecordingMiniMaxVAE()
    image = torch.full((1, 32, 32, 3), 0.5)

    with pytest.raises(ValueError, match=message):
        UC_AdvancedMiniMaxH3ImageToVideoCombined.execute(
            model=model,
            clip=clip,
            vae=vae,
            prompt="subject",
            width=32,
            height=32,
            length=5,
            reference_images={"reference_image_1": image},
            ref_image_size="first + match",
            **extra_inputs,
        )

    assert clip.encoded_tokens == []
    assert vae.images == []
    assert model.clone_calls == 0


def test_advanced_combined_minimax_h3_uses_installed_core_payload_order(monkeypatch):
    monkeypatch.setattr(
        comfy.model_base.BaseModel,
        "extra_conds",
        lambda _self, **_kwargs: {},
    )
    core_model = object.__new__(comfy.model_base.MiniMaxH3)
    core_model.latent_shapes = None
    keyframe_latent = torch.tensor([[[1.0]]])
    reference_latent = torch.tensor([[[2.0]]])
    keyframes = [{"resolved_frame_index": 0, "latent": keyframe_latent}]
    references = [
        {
            "kind": "image",
            "latent_h": 2,
            "latent_w": 2,
            "latent": reference_latent,
        }
    ]

    core_conds = comfy.model_base.MiniMaxH3.extra_conds(
        core_model,
        minimax_keyframes=keyframes,
        minimax_frame_count=5,
        minimax_refs=references,
    )
    payload = core_conds["minimax_payload"].cond
    assert payload["cond_video_latents"] == [keyframe_latent, reference_latent]

    layout = comfy.ldm.minimax.model.PackedLayout(
        2,
        2,
        2,
        2,
        2,
        keyframes=keyframes,
        refs=references,
    )
    payload["layout"] = layout
    observed = {}

    def executor(**kwargs):
        observed.update(kwargs)

    encoder_helpers.minimax_h3_combined_payload_wrapper(
        executor, minimax_payload=payload
    )
    forwarded = observed["minimax_payload"]
    assert forwarded["cond_video_latents"] == [keyframe_latent, reference_latent]
    assert forwarded["layout"] is layout
    assert [kind for _start, _stop, kind in layout.segments] == [
        "text",
        "cond",
        "ref_img",
        "audio",
        "video",
    ]


@pytest.mark.parametrize(
    ("image_width", "image_height", "generation_width", "generation_height", "mode", "expected"),
    [
        (128, 64, 64, 32, "match", (64, 32)),
        (100, 50, 1024, 1024, "match", (96, 64)),
        (4096, 4096, 64, 64, "max", (2048, 2048)),
        (4096, 3072, 64, 64, "max", (2720, 2048)),
    ],
)
def test_minimax_h3_reference_size_matches_core(
    image_width,
    image_height,
    generation_width,
    generation_height,
    mode,
    expected,
):
    assert encoder_helpers.minimax_h3_reference_size(
        image_width,
        image_height,
        generation_width,
        generation_height,
        mode,
    ) == expected


def test_advanced_minimax_h3_reference_mode_preserves_flat_order_and_pixels():
    clip = _MiniMaxH3TestClip()
    vae = _RecordingMiniMaxVAE()
    first = torch.full((1, 32, 64, 3), 0.25)
    second = torch.full((1, 32, 64, 3), 0.5)
    third = torch.full((1, 64, 32, 3), 0.75)

    conditioning, latent = UC_AdvancedMiniMaxH3ImageToVideo.execute(
        clip,
        vae,
        prompt="subject",
        ref_image_size="match",
        vlm_resolution=0,
        width=64,
        height=32,
        length=5,
        reference_images={
            "reference_image_1": torch.cat([first, second], dim=0),
            "reference_image_2": third,
        },
        visual_fusion_config=None,
    ).args

    entries = clip.encoded_tokens[-1]["qwen3vl_32b"][0]
    qwen_images = [
        entry[0]["data"] for entry in entries if encoder_helpers.is_image_token(entry)
    ]
    assert [float(image.mean()) for image in qwen_images] == pytest.approx(
        [0.25, 0.5, 0.75], abs=0.004
    )
    native_call = clip.tokenize_calls[-1]
    assert native_call["images"] is None
    assert native_call["minimax_ref_items"] is not None
    assert native_call["text"] == "subject"
    assert len(vae.images) == 3
    assert [float(image.mean()) for image in vae.images] == pytest.approx(
        [0.25, 0.5, 0.75], abs=0.004
    )

    metadata = conditioning[0][1]
    references = metadata["minimax_refs"]
    assert [item["kind"] for item in references] == ["image", "image", "image"]
    assert [(item["latent_h"], item["latent_w"]) for item in references] == [
        (2, 4),
        (2, 4),
        (4, 2),
    ]
    assert [float(item["latent"].mean()) for item in references] == pytest.approx(
        [0.25, 0.5, 0.75], abs=0.004
    )
    assert "minimax_keyframes" not in metadata
    assert "minimax_frame_count" not in metadata
    video, audio = latent["samples"].tensors
    assert video.shape == (1, 24, 2, 2, 4)
    assert audio.shape == (1, 32, 2, 8)


def test_advanced_minimax_h3_reference_fusion_pairs_flattened_inputs():
    clip = _MiniMaxH3TestClip()
    vae = _RecordingMiniMaxVAE()
    references = [
        torch.full((1, 4, 6, 3), value)
        for value in (0.25, 0.5, 0.75)
    ]
    fusion = [
        torch.full((1, 4, 6, 3), value)
        for value in (1.0, 0.125, 0.375, 0.625)
    ]

    UC_AdvancedMiniMaxH3ImageToVideo.execute(
        clip,
        vae,
        prompt="subject",
        width=64,
        height=32,
        length=5,
        vlm_resolution=0,
        reference_images={
            "reference_image_1": torch.cat(references[:2], dim=0),
            "reference_image_2": references[2],
        },
        fusion_images={"fusion_image_1": torch.cat(fusion, dim=0)},
        visual_fusion_config={
            "visual_fusion_method": "linear",
            "visual_encoder_path": "grid-deepstack",
        },
    )

    encoded_images = [
        [float(entry[0]["data"].mean()) for entry in tokens["qwen3vl_32b"][0] if encoder_helpers.is_image_token(entry)]
        for tokens in clip.encoded_tokens
    ]
    assert np.allclose(
        encoded_images,
        [
            [0.25, 0.5, 0.75],
            [1.0, 0.5, 0.75],
            [0.25, 0.125, 0.75],
            [0.25, 0.5, 0.375],
        ],
        atol=0.004,
    )
    assert [float(image.mean()) for image in vae.images] == pytest.approx(
        [0.25, 0.5, 0.75], abs=0.004
    )


def test_advanced_minimax_h3_reference_fusion_singleton_broadcasts():
    clip = _MiniMaxH3TestClip()
    vae = _RecordingMiniMaxVAE()
    references = [
        torch.full((1, 4, 6, 3), value)
        for value in (0.25, 0.5, 0.75)
    ]
    fusion = torch.full((1, 4, 6, 3), 1.0)

    UC_AdvancedMiniMaxH3ImageToVideo.execute(
        clip,
        vae,
        prompt="subject",
        width=64,
        height=32,
        length=5,
        vlm_resolution=0,
        reference_images={"reference_image_1": torch.cat(references, dim=0)},
        fusion_images={"fusion_image_1": fusion},
        visual_fusion_config={
            "visual_fusion_method": "linear",
            "visual_encoder_path": "grid-deepstack",
        },
    )

    encoded_images = [
        [float(entry[0]["data"].mean()) for entry in tokens["qwen3vl_32b"][0] if encoder_helpers.is_image_token(entry)]
        for tokens in clip.encoded_tokens
    ]
    assert np.allclose(
        encoded_images,
        [[0.25, 0.5, 0.75], [1.0, 0.5, 0.75], [0.25, 1.0, 0.75], [0.25, 0.5, 1.0]],
        atol=0.004,
    )


def test_advanced_minimax_h3_reference_fusion_second_socket_disables_broadcast():
    clip = _MiniMaxH3TestClip()
    vae = _RecordingMiniMaxVAE()
    references = torch.stack(
        [
            torch.full((4, 6, 3), 0.25),
            torch.full((4, 6, 3), 0.5),
            torch.full((4, 6, 3), 0.75),
        ]
    )

    UC_AdvancedMiniMaxH3ImageToVideo.execute(
        clip,
        vae,
        prompt="subject",
        width=64,
        height=32,
        length=5,
        vlm_resolution=0,
        reference_images={"reference_image_1": references},
        fusion_images={
            "fusion_image_1": torch.full((1, 4, 6, 3), 1.0),
            "fusion_image_2": torch.full((1, 4, 6, 3), 0.125),
        },
        visual_fusion_config={
            "visual_fusion_method": "linear",
            "visual_encoder_path": "grid-deepstack",
        },
    )

    encoded_images = [
        [float(entry[0]["data"].mean()) for entry in tokens["qwen3vl_32b"][0] if encoder_helpers.is_image_token(entry)]
        for tokens in clip.encoded_tokens
    ]
    assert np.allclose(
        encoded_images,
        [[0.25, 0.5, 0.75], [1.0, 0.5, 0.75], [0.25, 0.125, 0.75]],
        atol=0.004,
    )


def test_advanced_minimax_h3_reference_fusion_off_ignores_fusion_inputs():
    clip = _MiniMaxH3TestClip()
    vae = _RecordingMiniMaxVAE()
    references = torch.stack(
        [
            torch.full((4, 6, 3), 0.25),
            torch.full((4, 6, 3), 0.5),
        ]
    )
    fusion = torch.full((1, 4, 6, 3), 1.0)

    UC_AdvancedMiniMaxH3ImageToVideo.execute(
        clip,
        vae,
        prompt="subject",
        width=64,
        height=32,
        length=5,
        vlm_resolution=0,
        reference_images={"reference_image_1": references},
        fusion_images={"fusion_image_1": fusion},
        visual_fusion_config=None,
    )

    entries = clip.encoded_tokens[-1]["qwen3vl_32b"][0]
    qwen_images = [entry[0]["data"] for entry in entries if encoder_helpers.is_image_token(entry)]
    assert [float(image.mean()) for image in qwen_images] == pytest.approx(
        [0.25, 0.5], abs=0.004
    )
    assert len(clip.encoded_tokens) == 1


def test_advanced_minimax_h3_reference_save_exports_each_visual_span(monkeypatch):
    clip = _MiniMaxH3TestClip()
    clip.cond_stage_model = types.SimpleNamespace(clip_name="qwen3vl_8b")
    clip.tokenizer = types.SimpleNamespace(clip_name="qwen3vl_32b")
    vae = _RecordingMiniMaxVAE()
    exported = []
    monkeypatch.setattr(
        encoder_helpers,
        "save_source_visual_embeddings",
        lambda *args: exported.append(args),
    )

    UC_AdvancedMiniMaxH3ImageToVideo.execute(
        clip,
        vae,
        prompt="subject",
        ref_image_size="match",
        vlm_resolution=0,
        width=64,
        height=32,
        length=5,
        reference_images={
            "reference_image_1": torch.full((1, 32, 64, 3), 0.25),
            "reference_image_2": torch.full((1, 32, 64, 3), 0.5),
        },
        visual_fusion_config={
            "visual_fusion_method": "linear",
            "save_blended_embeds": True,
        },
    )

    assert len(exported) == 1
    _, tokens, _config, key, _device, visual_indices = exported[0]
    assert key == "qwen3vl_8b"
    assert visual_indices == [0, 1]
    assert sum(encoder_helpers.is_image_token(entry) for entry in tokens["qwen3vl_32b"][0]) == 2


def test_advanced_minimax_h3_none_keeps_frame_pictures_without_vae_keyframes():
    clip = _MiniMaxH3TestClip()
    vae = _RecordingMiniMaxVAE()
    first = torch.full((1, 32, 64, 3), 0.25)
    last = torch.full((1, 32, 64, 3), 0.75)

    conditioning, _latent = UC_AdvancedMiniMaxH3ImageToVideo.execute(
        clip,
        vae,
        prompt="subject",
        first_frame=first,
        last_frame=last,
        ref_image_size="none",
        width=64,
        height=32,
        length=5,
    ).args

    qwen_images = [
        entry[0]["data"]
        for entry in clip.encoded_tokens[-1]["qwen3vl_32b"][0]
        if encoder_helpers.is_image_token(entry)
    ]
    assert [float(image.mean()) for image in qwen_images] == pytest.approx(
        [0.25, 0.75], abs=0.004
    )
    assert vae.images == []
    metadata = conditioning[0][1]
    assert "minimax_keyframes" not in metadata
    assert "minimax_frame_count" not in metadata


def test_advanced_minimax_h3_reference_none_is_ordered_vlm_only():
    clip = _MiniMaxH3TestClip()
    vae = _RecordingMiniMaxVAE()
    first = torch.full((1, 32, 64, 3), 0.25)
    second = torch.full((1, 32, 64, 3), 0.5)
    third = torch.full((1, 64, 32, 3), 0.75)

    conditioning, _latent = UC_AdvancedMiniMaxH3ImageToVideo.execute(
        clip,
        vae,
        prompt="subject",
        ref_image_size="none",
        vlm_resolution=256,
        width=64,
        height=32,
        length=5,
        reference_images={
            "reference_image_1": torch.cat([first, second], dim=0),
            "reference_image_2": third,
        },
    ).args

    native_call = clip.tokenize_calls[-1]
    assert native_call["images"] is None
    assert native_call["minimax_ref_items"] is not None
    assert len(native_call["minimax_ref_items"]) == 3
    qwen_images = [item["data"] for item in native_call["minimax_ref_items"]]
    assert [float(image.mean()) for image in qwen_images] == pytest.approx(
        [0.25, 0.5, 0.75], abs=0.004
    )
    assert [image.shape[1:3] for image in qwen_images] == [
        encoder_helpers.vlm_target_dimensions(32, 64, 256),
        encoder_helpers.vlm_target_dimensions(32, 64, 256),
        encoder_helpers.vlm_target_dimensions(64, 32, 256),
    ]
    assert vae.images == []
    metadata = conditioning[0][1]
    assert "minimax_refs" not in metadata
    assert "minimax_keyframes" not in metadata
    assert "minimax_frame_count" not in metadata


def test_advanced_minimax_h3_vlm_resolution_is_independent_for_every_role():
    first = torch.full((1, 16, 32, 3), 0.125)
    reference = torch.full((1, 32, 64, 3), 0.25)
    fusion = torch.full((1, 64, 32, 3), 0.5)

    keyframe_clip = _MiniMaxH3TestClip()
    keyframe_vae = _RecordingMiniMaxVAE()
    UC_AdvancedMiniMaxH3ImageToVideo.execute(
        keyframe_clip,
        keyframe_vae,
        prompt="subject",
        first_frame=first,
        vlm_resolution=256,
        width=64,
        height=32,
        length=5,
        fusion_images={"fusion_image_1": fusion},
    )
    keyframe_images = [
        entry[0]["data"]
        for entry in keyframe_clip.encoded_tokens[-1]["qwen3vl_32b"][0]
        if encoder_helpers.is_image_token(entry)
    ]
    assert [image.shape[1:3] for image in keyframe_images] == [
        encoder_helpers.vlm_target_dimensions(16, 32, 256),
        encoder_helpers.vlm_target_dimensions(64, 32, 256),
    ]
    assert [image.shape[1:3] for image in keyframe_vae.images] == [(32, 64)]

    reference_clip = _MiniMaxH3TestClip()
    reference_vae = _RecordingMiniMaxVAE()
    UC_AdvancedMiniMaxH3ImageToVideo.execute(
        reference_clip,
        reference_vae,
        prompt="subject",
        ref_image_size="match",
        vlm_resolution=256,
        width=64,
        height=32,
        length=5,
        reference_images={"reference_image_1": reference},
    )
    reference_qwen = next(
        entry[0]["data"]
        for entry in reference_clip.encoded_tokens[-1]["qwen3vl_32b"][0]
        if encoder_helpers.is_image_token(entry)
    )
    assert reference_qwen.shape[1:3] == encoder_helpers.vlm_target_dimensions(
        32, 64, 256
    )
    assert [image.shape[1:3] for image in reference_vae.images] == [(32, 64)]


def test_advanced_minimax_h3_rejects_simultaneous_native_modes_before_encoding():
    first = torch.full((1, 32, 64, 3), 0.125)
    reference = torch.full((1, 32, 64, 3), 0.25)

    for kwargs, message in [
        (
            {"first_frame": first, "reference_images": {"reference_image_1": reference}},
            "frame inputs cannot be combined",
        ),
    ]:
        clip = _MiniMaxH3TestClip()
        vae = _RecordingMiniMaxVAE()
        with pytest.raises(ValueError, match=message):
            UC_AdvancedMiniMaxH3ImageToVideo.execute(
                clip, vae, "subject", 64, 32, 5, **kwargs
            )
        assert clip.encoded_tokens == []
        assert vae.images == []


def test_advanced_minimax_h3_frame_fusion_targets_matching_picture_slots():
    clip = _MiniMaxH3TestClip()
    vae = _RecordingMiniMaxVAE()
    first = torch.full((1, 4, 6, 3), 0.25)
    second = torch.full((1, 4, 6, 3), 0.5)
    third = torch.full((1, 4, 6, 3), 0.75)
    fourth = torch.full((1, 4, 6, 3), 1.0)

    output = UC_AdvancedMiniMaxH3ImageToVideo.execute(
        clip,
        vae,
        prompt="subject",
        first_frame=first,
        last_frame=second,
        vlm_resolution=0,
        width=64,
        height=32,
        length=22,
        fusion_images={
            "fusion_image_1": torch.cat([third, fourth], dim=0),
        },
        visual_fusion_config={
            "visual_fusion_method": "linear",
            "visual_encoder_path": "grid-deepstack",
        },
    )

    conditioning, latent = output.args
    encoded_images = []
    for tokens in clip.encoded_tokens:
        entries = tokens["qwen3vl_32b"][0]
        images = [entry[0]["data"] for entry in entries if encoder_helpers.is_image_token(entry)]
        if images:
            encoded_images.append(images)
    assert len(encoded_images) == 3
    assert [len(images) for images in encoded_images] == [2, 2, 2]
    assert np.allclose(
        [[float(image.mean()) for image in images] for images in encoded_images],
        [[0.25, 0.5], [0.75, 0.5], [1.0, 0.5]],
        atol=0.004,
    )
    assert [float(image.mean()) for image in vae.images] == pytest.approx(
        [0.25, 0.5], abs=0.004
    )
    assert encoded_images[0][0].shape[1:3] == (4, 6)
    assert encoded_images[1][0].shape[1:3] == (4, 6)
    assert vae.images[0].shape[1:3] == (32, 64)
    assert vae.images[1].shape[1:3] == (32, 64)
    image_calls = [call for call in clip.tokenize_calls if call["images"]]
    assert image_calls
    assert all(call["minimax_ref_items"] is None for call in image_calls)
    assert all(call["text"] == "subject" for call in image_calls)

    metadata = conditioning[0][1]
    assert metadata["minimax_frame_count"] == 22
    assert [item["resolved_frame_index"] for item in metadata["minimax_keyframes"]] == [0, 21]
    assert metadata["minimax_token_tags"].numel() == conditioning[0][0].shape[1]
    video, audio = latent["samples"].tensors
    assert video.shape == (1, 24, 7, 2, 4)
    assert audio.shape == (1, 32, 2, 37)


def test_advanced_minimax_h3_first_frame_fusion_uses_only_picture_one():
    clip = _MiniMaxH3TestClip()
    vae = _RecordingMiniMaxVAE()
    first = torch.full((1, 4, 6, 3), 0.25)
    fusion = torch.full((1, 4, 6, 3), 0.75)

    conditioning, _latent = UC_AdvancedMiniMaxH3ImageToVideo.execute(
        clip,
        vae,
        prompt="subject",
        first_frame=first,
        width=64,
        height=32,
        length=5,
        fusion_images={"fusion_image_1": fusion},
        visual_fusion_config={"visual_fusion_method": "linear"},
    ).args

    encoded_images = [
        [entry[0]["data"] for entry in tokens["qwen3vl_32b"][0] if encoder_helpers.is_image_token(entry)]
        for tokens in clip.encoded_tokens
    ]
    assert [[float(image.mean()) for image in images] for images in encoded_images] == [
        [0.25],
        [0.75],
    ]
    assert [item["resolved_frame_index"] for item in conditioning[0][1]["minimax_keyframes"]] == [0]
    assert [float(image.mean()) for image in vae.images] == pytest.approx([0.25], abs=0.004)


def test_advanced_minimax_h3_fusion_batches_stay_on_their_socket_slots():
    clip = _MiniMaxH3TestClip()
    vae = _RecordingMiniMaxVAE()
    first = torch.full((1, 4, 6, 3), 0.25)
    last = torch.full((1, 4, 6, 3), 0.5)
    first_fusion = torch.cat(
        [
            torch.full((1, 4, 6, 3), 0.75),
            torch.full((1, 4, 6, 3), 1.0),
        ]
    )
    last_fusion = torch.cat(
        [
            torch.full((1, 4, 6, 3), 0.125),
            torch.full((1, 4, 6, 3), 0.375),
        ]
    )

    UC_AdvancedMiniMaxH3ImageToVideo.execute(
        clip,
        vae,
        prompt="subject",
        first_frame=first,
        last_frame=last,
        width=64,
        height=32,
        length=5,
        fusion_images={
            "fusion_image_1": first_fusion,
            "fusion_image_2": last_fusion,
        },
        visual_fusion_config={"visual_fusion_method": "linear"},
    )

    encoded_images = [
        [entry[0]["data"] for entry in tokens["qwen3vl_32b"][0] if encoder_helpers.is_image_token(entry)]
        for tokens in clip.encoded_tokens
    ]
    assert np.allclose(
        [[float(image.mean()) for image in images] for images in encoded_images],
        [
            [0.25, 0.5],
            [0.75, 0.5],
            [1.0, 0.5],
            [0.25, 0.125],
            [0.25, 0.375],
        ],
        atol=0.004,
    )
    assert [float(image.mean()) for image in vae.images] == pytest.approx([0.25, 0.5], abs=0.004)


def test_advanced_minimax_h3_rejects_unpaired_frame_fusion_input():
    clip = _MiniMaxH3TestClip()
    vae = _RecordingMiniMaxVAE()
    first = torch.full((1, 4, 6, 3), 0.25)
    fusion = torch.full((1, 4, 6, 3), 0.75)

    with pytest.raises(ValueError, match="without a matching picture slot"):
        UC_AdvancedMiniMaxH3ImageToVideo.execute(
            clip,
            vae,
            prompt="subject",
            first_frame=first,
            width=64,
            height=32,
            length=5,
            fusion_images={
                "fusion_image_1": fusion,
                "fusion_image_2": fusion,
            },
            visual_fusion_config={"visual_fusion_method": "linear"},
        )
    assert clip.encoded_tokens == []
    assert vae.images == []


def test_advanced_minimax_h3_fusion_off_keeps_all_images_as_separate_pictures():
    clip = _MiniMaxH3TestClip()
    vae = _RecordingMiniMaxVAE()
    images = torch.stack(
        [
            torch.full((3, 5, 3), 0.25),
            torch.full((3, 5, 3), 0.5),
            torch.full((3, 5, 3), 0.75),
        ]
    )

    first = torch.full((1, 3, 5, 3), 0.125)
    conditioning, _latent = UC_AdvancedMiniMaxH3ImageToVideo.execute(
        clip,
        vae,
        prompt="subject",
        first_frame=first,
        width=64,
        height=32,
        length=5,
        fusion_images={"fusion_image_1": images},
        visual_fusion_config=None,
    ).args

    entries = clip.encoded_tokens[-1]["qwen3vl_32b"][0]
    qwen_images = [entry[0]["data"] for entry in entries if encoder_helpers.is_image_token(entry)]
    assert [float(image.mean()) for image in qwen_images] == pytest.approx(
        [0.125, 0.25, 0.5, 0.75], abs=0.004
    )
    assert len(vae.images) == 1
    assert float(vae.images[0].mean()) == pytest.approx(0.125, abs=0.004)
    assert conditioning[0][1]["minimax_keyframes"][0]["resolved_frame_index"] == 0
    native_call = clip.tokenize_calls[-1]
    assert native_call["images"] is not None
    assert native_call["minimax_ref_items"] is None
    assert native_call["text"] == "subject"


def test_advanced_minimax_h3_keeps_placeholder_like_text_raw(monkeypatch):
    def reject_generic_placeholder_path(*_args, **_kwargs):
        raise AssertionError("Dedicated H3 execution used generic placeholder handling.")

    monkeypatch.setattr(
        encoder_helpers,
        "prepare_image_placeholder_prompt",
        reject_generic_placeholder_path,
    )
    clip = _MiniMaxH3TestClip()
    image = torch.ones(1, 4, 4, 3)

    UC_AdvancedMiniMaxH3ImageToVideo.execute(
        clip,
        _RecordingMiniMaxVAE(),
        prompt="image_input_fusion image_input_1 image_input_2",
        first_frame=image,
        width=64,
        height=32,
        length=5,
    )

    native_call = clip.tokenize_calls[-1]
    assert native_call["text"] == "image_input_fusion image_input_1 image_input_2"
    assert native_call["images"] is not None


def test_advanced_minimax_h3_validates_encoder():
    class WrongClip:
        tokenizer = types.SimpleNamespace(clip_name="qwen3vl_8b")

    with pytest.raises(ValueError, match="qwen3vl_32b"):
        UC_AdvancedMiniMaxH3ImageToVideo.execute(
            WrongClip(),
            _RecordingMiniMaxVAE(),
            "prompt",
            64,
            32,
            5,
        )


def test_advanced_minimax_h3_accepts_text_only_and_last_only():
    clip = _MiniMaxH3TestClip()
    vae = _RecordingMiniMaxVAE()

    conditioning, _latent = UC_AdvancedMiniMaxH3ImageToVideo.execute(
        clip, vae, "subject", 64, 32, 5, multiplier=2.0
    ).args

    assert vae.images == []
    assert torch.all(conditioning[0][0] == 2.0)
    assert "minimax_keyframes" not in conditioning[0][1]
    assert "minimax_refs" not in conditioning[0][1]
    assert clip.tokenize_calls[-1]["images"] == []

    last = torch.full((1, 32, 64, 3), 0.75)
    conditioning, _latent = UC_AdvancedMiniMaxH3ImageToVideo.execute(
        clip,
        vae,
        "subject",
        64,
        32,
        5,
        last_frame=last,
    ).args

    assert [item["resolved_frame_index"] for item in conditioning[0][1]["minimax_keyframes"]] == [4]
    assert [float(image.mean()) for image in clip.tokenize_calls[-1]["images"]] == pytest.approx([0.75], abs=0.004)


def test_embedding_output_cannot_escape_root(tmp_path):
    nested = encoder_helpers.resolve_embedding_output_path(str(tmp_path), "nested/item.safetensors")
    assert pathlib.Path(nested).is_relative_to(tmp_path)
    with pytest.raises(ValueError, match="escapes"):
        encoder_helpers.resolve_embedding_output_path(str(tmp_path), "../outside.safetensors")
    with pytest.raises(ValueError, match="relative"):
        encoder_helpers.resolve_embedding_output_path(str(tmp_path), str(tmp_path / "absolute.safetensors"))


def test_krea2_mapping_mirrors_core_prefix_strip():
    tokens = [
        (np.int64(151644), 1.0), (8948, 1.0), (198, 1.0), (42, 1.0), (151645, 1.0),
        (np.int64(151644), 1.0), (872, 1.0), (198, 1.0), (100, 1.0), (101, 1.0), (151645, 1.0),
    ]
    conditioning = torch.zeros(1, 3, 12 * 2560)
    mapping = encoder_helpers.build_token_to_conditioning_map(tokens, conditioning)
    assert mapping[:8] == [(-1, -1)] * 8
    assert mapping[8:] == [(0, 1), (1, 2), (2, 3)]


def test_token_mapping_expands_prompt_embedding_entries():
    embedding = torch.zeros(3, 5120)
    tokens = [(100, 1.0), (embedding, 1.0), (101, 1.0)]
    conditioning = torch.zeros(1, 5, 5120)

    mapping = encoder_helpers.build_token_to_conditioning_map(tokens, conditioning)

    assert encoder_helpers.is_image_token(tokens[1]) is False
    assert mapping == [(0, 1), (1, 4), (4, 5)]


def test_mage_flow_mapping_mirrors_core_prefix_strip():
    image = torch.zeros(1, 832, 1248, 3)
    tokens = [(151644, 1.0), (8948, 1.0), (198, 1.0)]
    tokens.extend((token_id, 1.0) for token_id in range(28))
    tokens.extend([
        (np.int64(151644), 1.0), (872, 1.0), (198, 1.0), (74785, 1.0),
        ({"type": "image", "data": image}, 1.0),
        (100, 1.0), (101, 1.0), (102, 1.0), (103, 1.0), (104, 1.0), (151645, 1.0),
    ])
    conditioning = torch.zeros(1, 1021, 2560)

    mapping = encoder_helpers.build_token_to_conditioning_map(tokens, conditioning)

    assert mapping[:34] == [(-1, -1)] * 34
    assert mapping[34] == (0, 1)
    assert mapping[35] == (1, 1015)
    assert mapping[-1] == (1020, 1021)


def test_krea2_mapping_mirrors_custom_system_prefix_strip():
    image = torch.zeros(1, 32, 32, 3)
    tokens = [
        (151644, 1.0), (872, 1.0), (198, 1.0), (151645, 1.0), (198, 1.0),
        (151644, 1.0), (8948, 1.0), ({"type": "image", "data": image}, 1.0),
        (200, 1.0), (151645, 1.0),
    ]
    conditioning = torch.zeros(1, 8, 12 * 2560)

    mapping = encoder_helpers.build_token_to_conditioning_map(tokens, conditioning)

    assert mapping[:5] == [(-1, -1)] * 5
    assert mapping[5:] == [(0, 1), (1, 2), (2, 6), (6, 7), (7, 8)]


def test_krea2_mapping_rejects_unexplained_length_mismatch():
    image = torch.zeros(1, 32, 32, 3)
    tokens = [
        (151644, 1.0), (872, 1.0), (198, 1.0), (151645, 1.0), (198, 1.0),
        (151644, 1.0), (8948, 1.0), ({"type": "image", "data": image}, 1.0),
        (200, 1.0), (151645, 1.0),
    ]
    conditioning = torch.zeros(1, 6, 12 * 2560)

    with pytest.raises(ValueError, match="refusing to guess a visual range"):
        encoder_helpers.build_token_to_conditioning_map(tokens, conditioning)


def test_minimax_visual_range_uses_core_modality_tags():
    image = torch.zeros(1, 800, 832, 3)
    tokens = {
        "qwen3vl_32b": [[
            (100, 1.0),
            (151652, 1.0),
            ({"type": "image", "data": image}, 1.0),
            (151653, 1.0),
            (101, 1.0),
        ]],
    }
    conditioning = torch.zeros(1, 664, 5120)
    tags = torch.ones(664, dtype=torch.long)
    tags[5:657] = 0

    assert encoder_helpers.find_visual_token_range(
        tokens, conditioning, minimax_token_tags=tags
    ) == (6, 656)


def test_minimax_visual_range_selects_one_numbered_block():
    first = torch.zeros(1, 32, 32, 3)
    second = torch.zeros(1, 32, 64, 3)
    tokens = {
        "qwen3vl_32b": [[
            (100, 1.0),
            (151652, 1.0),
            ({"type": "image", "data": first}, 1.0),
            (151653, 1.0),
            (101, 1.0),
            (151652, 1.0),
            ({"type": "image", "data": second}, 1.0),
            (151653, 1.0),
            (102, 1.0),
        ]],
    }
    conditioning = torch.zeros(1, 20, 5120)
    tags = torch.ones(20, dtype=torch.long)
    tags[2:6] = 0
    tags[10:15] = 0

    assert encoder_helpers.find_visual_token_range(
        tokens,
        conditioning,
        minimax_token_tags=tags,
        minimax_visual_index=1,
    ) == (11, 14)
    with pytest.raises(ValueError, match="requires a selected visual block"):
        encoder_helpers.find_visual_token_range(
            tokens, conditioning, minimax_token_tags=tags
        )


def test_minimax_visual_range_rejects_tag_run_count_mismatch():
    image = torch.zeros(1, 32, 32, 3)
    tokens = {
        "qwen3vl_32b": [[
            (151652, 1.0),
            ({"type": "image", "data": image}, 1.0),
            (151653, 1.0),
        ]],
    }
    conditioning = torch.zeros(1, 12, 5120)
    tags = torch.ones(12, dtype=torch.long)
    tags[1:4] = 0
    tags[7:10] = 0

    with pytest.raises(ValueError, match="numbered visual blocks"):
        encoder_helpers.find_visual_token_range(
            tokens, conditioning, minimax_token_tags=tags
        )


def test_legacy_flat_visual_range_preserves_pre_refactor_spatial_mapping():
    image = torch.zeros(1, 128, 128, 3)
    tokens = {
        "qwen3vl_4b": [[
            (151644, 1.0), (872, 1.0), (198, 1.0), (151645, 1.0), (198, 1.0),
            (151644, 1.0), (8948, 1.0), ({"type": "image", "data": image}, 1.0),
            (200, 1.0), (151645, 1.0),
        ]],
    }
    conditioning = torch.zeros(1, 20, 12 * 2560)

    assert encoder_helpers.find_visual_token_range(
        tokens,
        conditioning,
        legacy_krea_spatial=True,
    ) == (7, 18)


def test_legacy_flat_fusion_layout_uses_retained_visual_span():
    image = torch.zeros(1, 896, 1184, 3)
    assert encoder_helpers.qwen3vl_visual_grid(image) == (28, 37)
    assert encoder_helpers.visual_fusion_grid(image, 1002, legacy_flat=True) == (1, 1002)
    with pytest.raises(ValueError, match="does not match range length"):
        encoder_helpers.visual_fusion_grid(image, 1002)


def test_unknown_visual_expansion_is_rejected_when_length_has_no_solution():
    tokens = [({"type": "image"}, 1.0), (10, 1.0), ({"type": "image"}, 1.0)]
    conditioning = torch.zeros(1, 8, 16)
    with pytest.raises(ValueError, match="no usable Qwen3-VL tensor payload"):
        encoder_helpers.build_token_to_conditioning_map(tokens, conditioning)


def test_klein_visual_range_ignores_core_tail_padding():
    image = torch.zeros(1, 32, 32, 3)
    tokens = {
        "qwen3_4b": [[
            (151652, 1.0),
            ({"type": "image", "data": image}, 1.0),
            (151653, 1.0),
            (151652, 1.0),
            (151655, 1.0),
            (151653, 1.0),
            (10, 1.0),
        ]]
    }
    conditioning = torch.zeros(1, 512, 16)

    assert encoder_helpers.find_visual_token_range(tokens, conditioning) == (1, 5)


def test_klein_vl_detection_does_not_match_z_image_tokenizer():
    klein_type = type(
        "KleinVLTokenizer", (), {"__module__": "comfy.text_encoders.flux"}
    )
    z_image_type = type(
        "ZImageTokenizer", (), {"__module__": "comfy.text_encoders.z_image"}
    )

    assert encoder_helpers.is_klein_vl_text_encoder(
        types.SimpleNamespace(tokenizer=klein_type())
    )
    assert not encoder_helpers.is_klein_vl_text_encoder(
        types.SimpleNamespace(tokenizer=z_image_type())
    )


def test_consensus_off_returns_reference_and_fractional_weights_stay_finite():
    first = torch.tensor([[[1.0, 0.0]]])
    second = torch.tensor([[[-1.0, 0.0]]])
    off, _ = encoder_helpers.blend_text_vectors({"a": first, "b": second}, {"blend_preset": "off"})
    assert off is first
    blended, _ = encoder_helpers.blend_text_vectors(
        {"a": first, "b": second},
        {
            "blend_preset": "custom",
            "blend_method": "consensus",
            "consensus_type": "mean",
            "alignment_method": "index",
            "power_alpha": 1.5,
            "similarity_threshold": -1.0,
        },
    )
    assert torch.isfinite(blended).all()


def test_consensus_blend_restores_sequence_and_pooled_reference_dtype():
    sequences = {
        "a": torch.tensor([[[1.0, 0.0]]], dtype=torch.float64),
        "b": torch.tensor([[[0.0, 1.0]]]),
    }
    pooled = {
        "a": torch.tensor([[1.0, 0.0]], dtype=torch.float16),
        "b": torch.tensor([[0.0, 1.0]]),
    }

    blended, blended_pooled = encoder_helpers.blend_text_vectors(
        sequences,
        {"blend_preset": "baseline"},
        pooled_tensors=pooled,
        device=sequences["a"].device,
        compute_dtype=torch.float32,
    )

    assert blended.device == sequences["a"].device
    assert blended.dtype == sequences["a"].dtype
    assert blended_pooled.device == pooled["a"].device
    assert blended_pooled.dtype == pooled["a"].dtype


def test_common_prefix_is_preserved_before_power_blend():
    prefix = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    first = torch.cat([prefix, torch.tensor([[1.0, 0.0], [0.0, 1.0]])])[None]
    second = torch.cat([prefix, torch.tensor([[0.8, 0.2], [0.2, 0.8]])])[None]
    blended, _ = encoder_helpers.blend_text_vectors(
        {"a": first, "b": second},
        {"blend_preset": "power_blend", "preserve_common_prefix": True},
        device=first.device,
        compute_dtype=torch.float32,
    )
    assert torch.equal(blended[0, :2], prefix)
    assert not torch.equal(blended[0, 2:], first[0, 2:])


def test_common_prefix_is_not_scaled_in_linear_mode():
    first = torch.tensor([[[2.0, 3.0], [1.0, 0.0]]])
    second = torch.tensor([[[2.0, 3.0], [3.0, 2.0]]])
    blended, _ = encoder_helpers.blend_text_vectors(
        {"a": first, "b": second},
        {"blend_preset": "custom", "blend_method": "linear", "global_scale": 2.0,
         "preserve_common_prefix": True},
        device=first.device,
        compute_dtype=torch.float32,
    )
    assert torch.equal(blended[0, 0], first[0, 0])
    assert torch.equal(blended[0, 1], torch.tensor([4.0, 2.0]))


def test_position_bias_prefers_nearby_normalized_positions():
    similarities = torch.tensor([[0.8, 0.9], [0.9, 0.8]])
    unbiased = encoder_helpers._position_biased_similarity_scores(similarities, 0.0)
    biased = encoder_helpers._position_biased_similarity_scores(similarities, 1.0)
    assert torch.equal(unbiased, similarities)
    assert biased[0, 0] > biased[0, 1]
    assert biased[1, 1] > biased[1, 0]


def test_position_bias_does_not_bypass_cosine_alignment_threshold():
    first = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    second = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]])
    blended, _ = encoder_helpers.blend_text_vectors(
        {"a": first, "b": second},
        {"blend_preset": "custom", "blend_method": "consensus", "consensus_type": "mean",
         "alignment_method": "similarity", "alignment_threshold": 0.9,
         "similarity_threshold": -1.0, "power_alpha": 1.0, "position_weight": 1.0,
         "rescale_norm": False, "global_scale": 1.0},
        device=first.device,
        compute_dtype=torch.float32,
    )
    assert torch.equal(blended, first)


def test_zero_position_weight_matches_legacy_similarity_alignment():
    sequences = {
        "a": torch.tensor([[[1.0, 0.0], [0.2, 0.8]]]),
        "b": torch.tensor([[[0.9, 0.1], [0.0, 1.0]]]),
    }
    config = {"blend_preset": "baseline"}
    legacy, _ = encoder_helpers.blend_text_vectors(sequences, config)
    explicit_zero, _ = encoder_helpers.blend_text_vectors(sequences, {**config, "position_weight": 0.0})
    assert torch.equal(legacy, explicit_zero)


def test_position_weight_validation_rejects_out_of_range_values():
    sequences = {"a": torch.ones(1, 1, 2), "b": torch.ones(1, 1, 2)}
    with pytest.raises(ValueError, match="Position weight"):
        encoder_helpers.blend_text_vectors(sequences, {"blend_preset": "baseline", "position_weight": 1.1})


def test_consensus_node_passes_original_tensors_to_blender(monkeypatch):
    first = torch.ones(1, 2, 3, dtype=torch.float64)
    second = torch.zeros(1, 2, 3)
    first_pooled = torch.ones(1, 3, dtype=torch.float16)
    seen = {}

    def fake_blend(sequence_tensors, config, pooled_tensors, device, compute_dtype):
        seen["sequence"] = sequence_tensors["a"]
        seen["pooled"] = pooled_tensors["a"]
        return sequence_tensors["a"], pooled_tensors["a"]

    monkeypatch.setattr(encoder_helpers.comfy.model_management, "get_torch_device", lambda: first.device)
    monkeypatch.setattr(encoder_helpers.comfy.model_management, "intermediate_dtype", lambda: torch.float32)
    monkeypatch.setattr("utils_collection_encoder_test.encoder_nodes.blend_text_vectors", fake_blend)

    output = UC_ConditioningConsensusBlend.execute(
        {
            "conditioning_1": [[first, {"pooled_output": first_pooled}]],
            "conditioning_2": [[second, {"pooled_output": torch.zeros(1, 3)}]],
        },
        {"blend_preset": "baseline"},
    ).result[0]

    assert seen["sequence"] is first
    assert seen["pooled"] is first_pooled
    assert output[0][0] is first
    assert output[0][1]["pooled_output"] is first_pooled


@pytest.mark.skipif(
    encoder_helpers.comfy.model_management.is_device_cpu(
        encoder_helpers.comfy.model_management.get_torch_device()
    ),
    reason="No accelerator backend is selected",
)
def test_consensus_accelerator_compute_does_not_change_cpu_output_placement():
    sequences = {"a": torch.ones(1, 2, 3), "b": torch.zeros(1, 2, 3)}
    pooled = {"a": torch.ones(1, 3), "b": torch.zeros(1, 3)}
    compute_device = encoder_helpers.comfy.model_management.get_torch_device()
    compute_dtype = encoder_helpers.comfy.model_management.intermediate_dtype()

    blended, blended_pooled = encoder_helpers.blend_text_vectors(
        sequences,
        {"blend_preset": "baseline"},
        pooled_tensors=pooled,
        device=compute_device,
        compute_dtype=compute_dtype,
    )

    assert blended.device == sequences["a"].device
    assert blended_pooled.device == pooled["a"].device


def test_contextual_weighting_does_not_scale_pooled_output():
    class Clip:
        @staticmethod
        def tokenize(text, **kwargs):
            return {"fake": [[(ord(char), 1.0) for char in text]]}

        @staticmethod
        def encode_from_tokens_scheduled(tokens):
            length = len(tokens["fake"][0])
            sequence = torch.ones(1, length, 2)
            pooled = torch.full((1, 2), 7.0)
            return [[sequence, {"pooled_output": pooled}]]

    conditioning = encoder_helpers.encode_embedding_classical_scaled_bias(Clip(), "(ab:2)c")
    sequence, metadata = conditioning[0]
    assert torch.equal(sequence[0, :2], torch.full((2, 2), 2.0))
    assert torch.equal(sequence[0, 2:], torch.ones(1, 2))
    assert torch.equal(metadata["pooled_output"], torch.full((1, 2), 7.0))


def test_contextual_weight_syntax_clean_text_matches_encoder_input():
    assert encoder_helpers.strip_contextual_weight_syntax("a (painting:-1) and ((light:2):0.5)") == "a painting and light"


def test_contextual_weight_syntax_preserves_backslash_escaped_parentheses():
    assert encoder_helpers.strip_contextual_weight_syntax(r"pop culture \(Overwatch\) and \(banana\)") == (
        "pop culture (Overwatch) and (banana)"
    )


def test_advanced_visual_text_only_path_preserves_custom_system_prompt():
    class Clip:
        tokenized_text = None

        @classmethod
        def tokenize(cls, text, **kwargs):
            cls.tokenized_text = text
            return {"fake": [[(1, 1.0)]]}

        @staticmethod
        def encode_from_tokens_scheduled(tokens):
            return [[torch.ones(1, 1, 1), {}]]

    UC_AdvancedVisualConditioningEncode.execute(
        Clip(),
        prompt="subject",
        system_prompt="custom rules",
        vlm_resolution=384,
        image_inputs={},
    )

    assert Clip.tokenized_text.startswith("<|im_start|>user\n<|im_end|>\n<|im_start|>system\ncustom rules")
    assert "<|im_start|>user\nsubject<|im_end|>" in Clip.tokenized_text

    UC_AdvancedVisualConditioningEncode.execute(
        Clip(),
        prompt="subject",
        system_prompt="",
        vlm_resolution=384,
        image_inputs={},
    )

    assert Clip.tokenized_text.startswith("<|im_start|>system\nDescribe the image")
    assert not Clip.tokenized_text.startswith("<|im_start|>user\n<|im_end|>")


def test_advanced_visual_image_only_path_uses_anti_stripping_template(monkeypatch):
    class Clip:
        tokenized_text = None

        @classmethod
        def tokenize(cls, text, **kwargs):
            cls.tokenized_text = text
            return {"fake": [[(1, 1.0)]]}

        @staticmethod
        def encode_from_tokens_scheduled(tokens):
            return [[torch.ones(1, 1, 1), {}]]

    monkeypatch.setattr(
        encoder_nodes, "prepare_vlm_image", lambda image, _resolution: image
    )

    UC_AdvancedVisualConditioningEncode.execute(
        Clip(),
        prompt="",
        system_prompt="",
        vlm_resolution=384,
        image_inputs={"image_1": torch.ones(1, 2, 2, 3)},
    )

    assert Clip.tokenized_text.startswith(
        "<|im_start|>user\n<|im_end|>\n<|im_start|>system\n<|im_end|>"
    )
    assert "Describe the image by detailing" not in Clip.tokenized_text
    assert "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>" in Clip.tokenized_text


def test_advanced_visual_semantic_anchor_numbers_unfused_inputs(monkeypatch):
    encoded_prompts = []

    def encode(_clip, prompt, **_kwargs):
        encoded_prompts.append(prompt)
        return [[torch.ones(1, 1, 1), {}]]

    monkeypatch.setattr(encoder_nodes, "prepare_vlm_image", lambda image, _resolution: image)
    monkeypatch.setattr(encoder_nodes, "encode_embedding_classical_scaled_bias", encode)

    UC_AdvancedVisualConditioningEncode.execute(
        object(),
        prompt="describe",
        system_prompt="",
        vlm_resolution=384,
        image_inputs={
            "image_1": torch.zeros(1, 2, 2, 3),
            "image_2": torch.ones(1, 2, 2, 3),
        },
        semantic_anchor=True,
    )

    assert f"<Picture 1>: {encoder_nodes.VISION_BLOCK}" in encoded_prompts[0]
    assert f"<Picture 2>: {encoder_nodes.VISION_BLOCK}" in encoded_prompts[1]


def test_advanced_visual_semantic_anchor_is_disabled_by_default(monkeypatch):
    encoded_prompts = []

    def encode(_clip, prompt, **_kwargs):
        encoded_prompts.append(prompt)
        return [[torch.ones(1, 1, 1), {}]]

    monkeypatch.setattr(encoder_nodes, "prepare_vlm_image", lambda image, _resolution: image)
    monkeypatch.setattr(encoder_nodes, "encode_embedding_classical_scaled_bias", encode)

    UC_AdvancedVisualConditioningEncode.execute(
        object(),
        prompt="describe",
        system_prompt="",
        vlm_resolution=384,
        image_inputs={"image_1": torch.ones(1, 2, 2, 3)},
    )

    assert "<Picture 1>:" not in encoded_prompts[0]


def test_advanced_visual_semantic_anchor_preserves_inline_image_numbers(monkeypatch):
    class Clip:
        cond_stage_model = object()
        tokenized_text = None

        @classmethod
        def tokenize(cls, text, **_kwargs):
            cls.tokenized_text = text
            return {"fake": [[(1, 1.0)]]}

        @staticmethod
        def encode_from_tokens_scheduled(_tokens):
            return [[torch.ones(1, 1, 1), {}]]

    monkeypatch.setattr(encoder_nodes, "prepare_vlm_image", lambda image, _resolution: image)

    UC_AdvancedVisualConditioningEncode.execute(
        Clip(),
        prompt="image_input_2 then image_input_1",
        system_prompt="",
        vlm_resolution=384,
        image_inputs={
            "image_1": torch.zeros(1, 2, 2, 3),
            "image_2": torch.ones(1, 2, 2, 3),
        },
        semantic_anchor=True,
    )

    picture_two = f"<Picture 2>: {encoder_nodes.VISION_BLOCK}"
    picture_one = f"<Picture 1>: {encoder_nodes.VISION_BLOCK}"
    assert picture_two in Clip.tokenized_text, Clip.tokenized_text
    assert picture_one in Clip.tokenized_text, Clip.tokenized_text
    assert Clip.tokenized_text.index(picture_two) < Clip.tokenized_text.index(picture_one)


def test_numbered_image_placeholders_preserve_prompt_order_and_strip_invalid(caplog):
    prompt, numbers = encoder_helpers.prepare_image_placeholder_prompt(
        "first image_input_2 then IMAGE_INPUT_1 repeat image_input_2 missing image_input_3 image_input_fusion",
        image_count=2,
        fusion_active=False,
        context="test",
    )

    assert numbers == (2, 1, 2)
    assert prompt.count(encoder_helpers.VISION_BLOCK) == 3
    assert "image_input_" not in prompt.lower()
    assert "stripped unavailable or fusion-only" in caplog.text


def test_fusion_placeholder_uses_one_slot_and_strips_the_rest(caplog):
    prompt, numbers = encoder_helpers.prepare_image_placeholder_prompt(
        "ignored image_input_1 chosen image_input_fusion removed image_input_2",
        image_count=2,
        fusion_active=True,
        context="test",
    )

    assert numbers == ()
    assert prompt.count(encoder_helpers.VISION_BLOCK) == 1
    assert "image_input_" not in prompt.lower()
    assert "stripped 2 additional" in caplog.text


def test_fusion_placeholder_accepts_image_one_alias_and_logs_fallback(caplog):
    prompt, _ = encoder_helpers.prepare_image_placeholder_prompt(
        "near image_input_1 subject",
        image_count=3,
        fusion_active=True,
        context="test",
    )

    assert prompt == f"near {encoder_helpers.VISION_BLOCK} subject"
    assert "treating image_input_1 as image_input_fusion" in caplog.text


def test_canonical_and_compatibility_schema_flags():
    assert UC_AttentionBiasTextEncode.define_schema().is_experimental
    assert UC_AdvancedVisualConditioningEncode.define_schema().is_experimental
    assert not UC_AdvancedVisualConditioningEncode.define_schema().is_deprecated
    assert TextEncodeKrea2SystemEditScaledAdv.define_schema().is_deprecated
    assert UC_Krea2TokenAttentionWeight.define_schema().is_experimental
    assert TextEncodeKrea2SysEditScaledAdvAttn.define_schema().is_deprecated
    assert UC_Qwen3VLInputEmbeds.define_schema().is_deprecated
    assert not UC_VLMInputEmbeds.define_schema().is_deprecated


def test_visual_fusion_encoder_formula_defaults_are_blank():
    for node in (
        encoder_nodes.UC_AdvancedVisualConditioningEncode,
        encoder_nodes.UC_Krea2TokenAttentionWeight,
    ):
        inputs = {value.id: value for value in node.define_schema().inputs}
        assert inputs["formula"].default == ""
        assert inspect.signature(node.execute).parameters["formula"].default == ""

    schema = encoder_nodes.UC_AdvancedVisualConditioningEncode.define_schema()
    assert [value.id for value in schema.inputs][-2:] == [
        "semantic_anchor",
        "image_inputs",
    ]
    assert {value.id: value for value in schema.inputs}["semantic_anchor"].default is False
    assert list(inspect.signature(encoder_nodes.UC_AdvancedVisualConditioningEncode.execute).parameters)[-1] == "semantic_anchor"
