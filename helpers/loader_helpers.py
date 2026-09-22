import torch

import comfy.sd
import comfy.utils


_TEXT_PROJECTION_KEY = (
    "detector.backbone.language_backbone.encoder.text_projection"
)


def load_sam31_checkpoint(
    checkpoint_path,
    precision="fp32",
    output_vae=True,
    output_clip=True,
    output_model=True,
    embedding_directory=None,
    disable_dynamic=False,
):
    state_dict, metadata = comfy.utils.load_torch_file(
        checkpoint_path, return_metadata=True
    )
    state_dict.pop(_TEXT_PROJECTION_KEY, None)
    dtype_options = {"dtype": torch.float32} if precision == "fp32" else {}
    output = comfy.sd.load_state_dict_guess_config(
        state_dict,
        output_vae=output_vae,
        output_clip=output_clip,
        output_clipvision=False,
        embedding_directory=embedding_directory,
        output_model=output_model,
        model_options=dtype_options,
        te_model_options=dtype_options,
        metadata=metadata,
        disable_dynamic=disable_dynamic,
    )
    if output is None:
        raise RuntimeError(f"Could not detect a SAM 3.1 model in {checkpoint_path}.")
    if output[0] is not None:
        output[0].cached_patcher_init = (
            load_sam31_checkpoint,
            (
                checkpoint_path,
                precision,
                False,
                False,
                True,
                embedding_directory,
            ),
            0,
        )
    if output[1] is not None and getattr(output[1], "patcher", None) is not None:
        output[1].patcher.cached_patcher_init = (
            load_sam31_clip_patcher,
            (checkpoint_path, precision, embedding_directory),
        )
    return output


def load_sam31_clip_patcher(
    checkpoint_path,
    precision="fp32",
    embedding_directory=None,
    disable_dynamic=False,
):
    _, clip, _, _ = load_sam31_checkpoint(
        checkpoint_path,
        precision,
        output_vae=False,
        output_clip=True,
        output_model=False,
        embedding_directory=embedding_directory,
        disable_dynamic=disable_dynamic,
    )
    return clip.patcher
