import comfy
from folder_paths import get_filename_list, get_folder_paths, get_full_path_or_raise
from comfy.sd import load_lora_for_models
from comfy.utils import load_torch_file
from comfy_api.latest import io
from ..helpers.loader_helpers import load_sam31_checkpoint

_LORA_LOADER_CACHE = None


class UC_LoraLoaderCLIPOnly(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_LoraLoaderCLIPOnly",
            display_name="Load LoRA for CLIP Only",
            category="advanced/model",
            inputs=[
                io.Clip.Input("clip"),
                io.Combo.Input("lora_name", get_filename_list("loras"), tooltip="Choose a LoRA to apply to the text encoder. LoRAs without text-encoder data cannot be used here."),
                io.Float.Input("strength_clip", default=1.0, min=-10.0, max=10.0, step=0.05, tooltip="Strength of the LoRA effect on text encoding."),
            ],
            outputs=[
                io.Clip.Output(display_name="clip"),
            ],
        )

    @classmethod
    def execute(cls, clip, lora_name: str, strength_clip: float) -> io.NodeOutput:
        global _LORA_LOADER_CACHE
        if strength_clip == 0:
            return (io.NodeOutput(clip),)
        # Placeholder for actual LoRA loading logic
        lora_path = get_full_path_or_raise("loras", lora_name)
        lora = None

        if _LORA_LOADER_CACHE is not None:
            if _LORA_LOADER_CACHE[0] == lora_path:
                lora = _LORA_LOADER_CACHE[1]
            else:
                _LORA_LOADER_CACHE = None

        if lora is None:
            lora = load_torch_file(lora_path, safe_load=True)
            _LORA_LOADER_CACHE = (lora_path, lora)

        clip_lora = load_lora_for_models(None, clip, lora, 0, strength_clip)[1]
        return io.NodeOutput(clip_lora)


class UC_SAM31CheckpointLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_SAM31CheckpointLoader",
            display_name="Load SAM 3.1 Checkpoint",
            category="advanced/model",
            description="Loads a SAM 3.1 checkpoint with automatic or forced FP32 model and text-encoder precision.",
            inputs=[
                io.Combo.Input(
                    "ckpt_name",
                    options=get_filename_list("checkpoints"),
                    tooltip="SAM 3.1 checkpoint from the checkpoints folder.",
                ),
                io.Combo.Input(
                    "precision",
                    options=["auto", "fp32"],
                    default="fp32",
                    tooltip="fp32 prevents Core from downcasting model and text-encoder weights.",
                ),
            ],
            outputs=[
                io.Model.Output("model"),
                io.Clip.Output("clip"),
                io.Vae.Output("vae"),
            ],
        )

    @classmethod
    def execute(cls, ckpt_name: str, precision: str = "fp32") -> io.NodeOutput:
        checkpoint_path = get_full_path_or_raise("checkpoints", ckpt_name)
        model, clip, vae, _ = load_sam31_checkpoint(
            checkpoint_path,
            precision,
            embedding_directory=get_folder_paths("embeddings"),
        )
        return io.NodeOutput(model, clip, vae)

