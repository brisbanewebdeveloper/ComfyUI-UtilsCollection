import pathlib
import sys
import types

import torch


CUSTOM_NODE_ROOT = pathlib.Path(__file__).parents[1]
PACKAGE_NAME = "utils_collection_parameter_test"
package = types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(CUSTOM_NODE_ROOT)]
sys.modules.setdefault(PACKAGE_NAME, package)

from comfy.cli_args import args as cli_args

prior_cpu = cli_args.cpu
cli_args.cpu = True
try:
    from utils_collection_parameter_test.nodes import parameter_nodes
    from utils_collection_parameter_test.helpers.helper_functions import AspectRatio
    from utils_collection_parameter_test.helpers.parameter_helpers import select_video_resolution
finally:
    cli_args.cpu = prior_cpu


def test_image_scale_picker_schema_uses_smooth_default_and_positive_scale():
    schema = parameter_nodes.UC_ImageScaleAndResolutionPicker.define_schema()
    inputs = {value.id: value for value in schema.inputs}
    outputs = {value.id: value for value in schema.outputs if value.id}

    assert inputs["upscale_method"].default == "lanczos"
    assert inputs["scale_by"].min > 0
    assert "scale_by" in outputs["upscaled_image"].tooltip
    assert "upscale_by" not in outputs["upscaled_image"].tooltip


def test_video_resolution_selector_uses_nominal_megapixel_target_with_ratio_tolerance():
    width, height = select_video_resolution(
        16,
        9,
        megapixels=0.4,
        multiple=32,
        minimum=256,
        maximum=4096,
    )

    assert (width, height) == (864, 480)
    assert width % 32 == height % 32 == 0


def test_video_resolution_selector_uses_middle_band_video_rungs():
    selected = {
        megapixels: select_video_resolution(
            16,
            9,
            megapixels=megapixels,
            multiple=32,
            minimum=256,
            maximum=4096,
        )
        for megapixels in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
    }

    assert selected == {
        0.3: (768, 416),
        0.4: (864, 480),
        0.5: (1024, 576),
        0.6: (1024, 576),
        0.7: (1152, 672),
        0.8: (1152, 672),
    }


def test_video_resolution_selector_schema_defaults_to_video_multiple():
    schema = parameter_nodes.UC_VideoResolutionSelector.define_schema()
    inputs = {value.id: value for value in schema.inputs}

    assert inputs["multiple"].default == 32
    assert "minimum" not in inputs


def test_video_resolution_selector_returns_resolution_preview():
    output = parameter_nodes.UC_VideoResolutionSelector.execute(
        aspect_ratio=AspectRatio.WIDESCREEN_H,
        megapixels=0.4,
        multiple=32,
        minimum=256,
        duration_seconds=9.3112024,
    )

    assert output.result == (864, 480, 226)
    assert output.ui == {"resolution": ("864×480 · 226 frames",)}


def test_regular_resolution_selector_returns_resolution_preview():
    output = parameter_nodes.UC_ResolutionSelectorExtended.execute(
        aspect_ratio=AspectRatio.WIDESCREEN_H,
        megapixels=0.4,
        multiple=32,
        minimum=256,
    )

    assert output.result == (864, 480)
    assert output.ui == {"resolution": ("864×480",)}


def test_resolution_preview_frontend_is_display_only_and_live():
    frontend = (CUSTOM_NODE_ROOT / "web" / "resolution_preview.js").read_text(
        encoding="utf-8"
    )

    assert "onDrawForeground" in frontend
    assert "onWidgetChanged" in frontend
    assert "prototype.computeSize" in frontend
    assert "prototype.onResize" in frontend
    assert "addWidget" not in frontend


def test_video_resolution_selector_keeps_an_aspect_ratio_axis_exact():
    width, height = select_video_resolution(
        21,
        9,
        megapixels=0.5,
        multiple=32,
        minimum=256,
        maximum=4096,
    )

    assert (width, height) == (1120, 480)
    assert width % 7 == 0


def test_image_scale_picker_keeps_megapixel_fit_and_upscale_factor_separate():
    image = torch.zeros(1, 100, 200, 3)

    output = parameter_nodes.UC_ImageScaleAndResolutionPicker.execute(
        image=image,
        upscale_method="bilinear",
        crop_method="disabled",
        aspect_ratio=AspectRatio.SQUARE,
        megapixels=0.01,
        resolution_steps=256,
        scale_by=2.0,
        multiple=16,
    )
    adjusted, upscaled, width, height, upscaled_width, upscaled_height = output.result

    assert (width, height) == (144, 80)
    assert (upscaled_width, upscaled_height) == (288, 160)
    assert adjusted.shape == (1, 80, 144, 3)
    assert upscaled.shape == (1, 160, 288, 3)


def test_image_scale_picker_center_crop_uses_adjusted_base_for_upscale():
    image = torch.zeros(1, 100, 200, 3)

    output = parameter_nodes.UC_ImageScaleAndResolutionPicker.execute(
        image=image,
        upscale_method="bilinear",
        crop_method="center",
        aspect_ratio=AspectRatio.SQUARE,
        megapixels=0.01,
        resolution_steps=1,
        scale_by=1.5,
        multiple=16,
    )

    assert output.result[2:] == (96, 96, 144, 144)
    assert output.result[0].shape == (1, 96, 96, 3)


def test_video_resolution_and_length_picker():
    schema = parameter_nodes.UC_VideoResolutionAndLengthPicker.define_schema()
    inputs = {value.id: value for value in schema.inputs}
    assert "video" in inputs and "image" in inputs and "use_video_duration" in inputs
    assert len(schema.outputs) == 6
    assert schema.outputs[4].display_name == "Duration (s)"
    assert schema.outputs[5].id == "audio"

    # Standalone execution
    output = parameter_nodes.UC_VideoResolutionAndLengthPicker.execute(
        aspect_ratio=AspectRatio.WIDESCREEN_H,
        megapixels=1.0,
        multiple=16,
        duration_seconds=5.16666,
    )
    image_out, width, height, length, duration, audio = output.result
    assert (width, height, length) == (1360, 768, 124)
    assert image_out.shape == (1, 768, 1360, 3)
    assert audio is None
    assert output.ui == {"resolution": ("1360×768 · 124 frames · 5.17 s",)}

    # Execution with input frames and duration
    mock_frames = torch.zeros(48, 100, 200, 3)
    out_video = parameter_nodes.UC_VideoResolutionAndLengthPicker.execute(
        video=mock_frames,
        use_video_duration=True,
        aspect_ratio=AspectRatio.SQUARE,
        megapixels=0.01,
        multiple=16,
    )
    frames_out, w, h, l, d, audio = out_video.result
    assert (w, h) == (144, 80)
    assert l == 56
    assert frames_out.shape == (56, 80, 144, 3)
    assert audio is None


def test_video_resolution_and_length_picker_resampling_low_and_high_fps():
    import types

    # 1. Low fps: 12 fps with 24 frames (2.0s duration) -> resample to 48 frames at 24 fps
    frames_12fps = torch.arange(24, dtype=torch.float32).view(24, 1, 1, 1).repeat(1, 64, 64, 3)
    comp_12fps = types.SimpleNamespace(images=frames_12fps, frame_rate=12.0, audio=None)
    vid_12fps = types.SimpleNamespace(get_components=lambda: comp_12fps, get_duration=lambda: 2.0)

    out_12 = parameter_nodes.UC_VideoResolutionAndLengthPicker.execute(
        video=vid_12fps,
        use_video_duration=True,
        match_video_length=False,
        aspect_ratio=AspectRatio.SQUARE,
        megapixels=0.01,
        multiple=16,
    )
    frames_out_12, _, _, _, duration_12, _ = out_12.result
    assert frames_out_12.shape[0] == 48
    assert duration_12 == 2.0
    # Clones neighboring frames: frame 0 and 1 are clone of source 0, etc.
    assert frames_out_12[0, 0, 0, 0].item() == 0.0
    assert frames_out_12[1, 0, 0, 0].item() in (0.0, 1.0)

    # 2. High fps: 60 fps with 120 frames (2.0s duration) -> resample to 48 frames at 24 fps
    frames_60fps = torch.arange(120, dtype=torch.float32).view(120, 1, 1, 1).repeat(1, 64, 64, 3)
    comp_60fps = types.SimpleNamespace(images=frames_60fps, frame_rate=60.0, audio=None)
    vid_60fps = types.SimpleNamespace(get_components=lambda: comp_60fps, get_duration=lambda: 2.0)

    out_60 = parameter_nodes.UC_VideoResolutionAndLengthPicker.execute(
        video=vid_60fps,
        use_video_duration=True,
        match_video_length=False,
        aspect_ratio=AspectRatio.SQUARE,
        megapixels=0.01,
        multiple=16,
    )
    frames_out_60, _, _, _, duration_60, _ = out_60.result
    assert frames_out_60.shape[0] == 48
    assert duration_60 == 2.0


def test_video_resolution_and_length_picker_outputs_h3_audio_untrimmed():
    import types

    # Audio with 44100 Hz mono waveform of 88200 samples (2.0 seconds)
    raw_waveform = torch.sin(torch.linspace(0, 100, 88200)).view(1, 1, 88200)
    raw_audio = {"waveform": raw_waveform, "sample_rate": 44100}
    comp = types.SimpleNamespace(
        images=torch.zeros(24, 64, 64, 3),
        frame_rate=24.0,
        audio=raw_audio,
    )
    vid = types.SimpleNamespace(get_components=lambda: comp, get_duration=lambda: 1.0)

    output = parameter_nodes.UC_VideoResolutionAndLengthPicker.execute(
        video=vid,
        use_video_duration=True,
        duration_seconds=1.0,
        aspect_ratio=AspectRatio.SQUARE,
        megapixels=0.01,
        multiple=16,
    )
    _, _, _, _, _, audio_out = output.result

    assert audio_out is not None
    assert isinstance(audio_out, dict)
    assert audio_out["sample_rate"] == 32000
    waveform_out = audio_out["waveform"]
    assert torch.is_tensor(waveform_out)
    # Valid for MiniMax H3: stereo (shape [1, 2, N])
    assert waveform_out.ndim == 3
    assert waveform_out.shape[1] == 2
    # Multiple of 800 samples for H3 audio VAE hop length
    assert waveform_out.shape[-1] % 800 == 0
    # Not trimmed to duration (duration was 1.0s, but 2.0s of audio is preserved ~64000 samples)
    assert waveform_out.shape[-1] >= 64000
