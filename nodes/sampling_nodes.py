"""Node definitions for sampling workflows."""

from comfy_api.latest import io
from ..helpers.sampling_helpers import start_sampling_loop


class UC_H3LoopSampler(io.ComfyNode):
    """Looping video/audio sampler for MiniMax H3 joint clips."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UC_H3LoopSampler",
            display_name="H3 Loop Sampler",
            category="utils/sampling",
            description="Samples long-form MiniMax H3 AV latents by windowed chunks with carry preservation.",
            inputs=[
                io.Noise.Input("noise"),
                io.Guider.Input("guider"),
                io.Sampler.Input("sampler"),
                io.Sigmas.Input("sigmas"),
                io.Conditioning.Input(
                    "conditioning",
                    tooltip="Base positive conditioning. If a list of conditionings or cond_set is supplied, chunks consume them sequentially.",
                ),
                io.Latent.Input("latent", tooltip="Full target H3 joint AV latent (e.g. from empty AV latent)."),
                io.Int.Input(
                    "chunk_frames",
                    default=124,
                    min=0,
                    max=3600,
                    step=17,
                    tooltip="Frame count per chunk. 0 samples the whole clip in one chunk.",
                ),
                io.Int.Input(
                    "overlap_frames",
                    default=22,
                    min=0,
                    max=720,
                    step=17,
                    tooltip="Overlap frame count between consecutive chunks.",
                ),
                io.Combo.Input(
                    "carry_mode",
                    options=["mask", "none"],
                    default="mask",
                    tooltip="Preserve tail of previous chunk into next chunk using noise mask.",
                ),
                io.Float.Input(
                    "overlap_strength_video",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="1.0 keeps previous video frames unchanged; lower values allow re-denoising.",
                ),
                io.Float.Input(
                    "overlap_strength_audio",
                    default=0.9,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="1.0 keeps previous audio unchanged; 0.9 allows smooth transition blend.",
                ),
                io.Int.Input(
                    "sampling_start_step",
                    default=0,
                    min=0,
                    max=1000,
                    step=1,
                    optional=True,
                    tooltip="Absolute starting step index in sigma schedule.",
                ),
                io.Int.Input(
                    "sampling_end_step",
                    default=1000,
                    min=1,
                    max=1000,
                    step=1,
                    optional=True,
                    tooltip="Absolute ending step index in sigma schedule.",
                ),
                io.Int.Input(
                    "phase2_start_step",
                    default=0,
                    min=0,
                    max=1000,
                    step=1,
                    optional=True,
                    tooltip="Step index to switch to phase-2 sampler/guider.",
                ),
                io.Sampler.Input("phase2_sampler", optional=True),
                io.Guider.Input("phase2_guider", optional=True),
                io.Mask.Input("denoise_mask", optional=True, tooltip="Optional whole-clip video denoise mask."),
                io.Mask.Input("audio_denoise_mask", optional=True, tooltip="Optional whole-clip audio denoise mask."),
            ],
            outputs=[
                io.Latent.Output("latent"),
                io.Int.Output("chunks_rendered"),
                io.String.Output("report"),
            ],
        )

    @classmethod
    def execute(
        cls,
        noise,
        guider,
        sampler,
        sigmas,
        conditioning,
        latent,
        chunk_frames=124,
        overlap_frames=22,
        carry_mode="mask",
        overlap_strength_video=1.0,
        overlap_strength_audio=0.9,
        sampling_start_step=0,
        sampling_end_step=1000,
        phase2_start_step=0,
        phase2_sampler=None,
        phase2_guider=None,
        denoise_mask=None,
        audio_denoise_mask=None,
    ) -> io.NodeOutput:
        cond_list = conditioning.get("conds") if isinstance(conditioning, dict) and "conds" in conditioning else (
            conditioning if isinstance(conditioning, list) and conditioning and isinstance(conditioning[0], list) and not isinstance(conditioning[0][0], torch.Tensor)
            else [conditioning]
        )

        out_latent, num_chunks, report = start_sampling_loop(
            noise=noise,
            guider=guider,
            sampler=sampler,
            sigmas=sigmas,
            cond_list=cond_list,
            latent=latent,
            chunk_frames=chunk_frames,
            overlap_frames=overlap_frames,
            carry_mode=carry_mode,
            overlap_strength_video=overlap_strength_video,
            overlap_strength_audio=overlap_strength_audio,
            sampling_start_step=sampling_start_step,
            sampling_end_step=sampling_end_step,
            phase2_start_step=phase2_start_step,
            phase2_sampler=phase2_sampler,
            phase2_guider=phase2_guider,
            denoise_mask=denoise_mask,
            audio_denoise_mask=audio_denoise_mask,
        )
        return io.NodeOutput(out_latent, num_chunks, report)
