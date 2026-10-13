import hashlib
import os
from typing import Any

import torch
from comfy_api.latest import InputImpl, io, ui
from ..helpers.video_helpers import (
    is_video_file,
    list_video_files,
    normalize_video_path,
    save_keyframe_optimized_video,
)


class UC_LoadVideoPath(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_LoadVideoPath",
            display_name="Load Video (Path)",
            category="advanced/video",
            search_aliases=["import video path", "open video path", "video file path"],
            inputs=[
                io.String.Input(
                    "video_path",
                    multiline=False,
                    placeholder="path/to/video.mp4 or X:/path/to/video.mp4",
                    tooltip="Video file path. Supports absolute or relative paths and either slash style.",
                ),
            ],
            outputs=[io.Video.Output(display_name="VIDEO")],
        )

    @classmethod
    def execute(cls, video_path: str) -> io.NodeOutput:
        normalized_path = normalize_video_path(video_path)
        if not normalized_path or not os.path.isfile(normalized_path):
            raise ValueError(
                f"Invalid video path: {video_path} (resolved to: {normalized_path})"
            )
        if not is_video_file(normalized_path):
            raise ValueError(f"File does not contain readable video: {video_path}")
        return io.NodeOutput(InputImpl.VideoFromFile(normalized_path))

    @classmethod
    def fingerprint_inputs(cls, video_path: str):
        normalized_path = normalize_video_path(video_path)
        if not normalized_path or not os.path.isfile(normalized_path):
            return ""
        stat = os.stat(normalized_path)
        return f"{normalized_path}:{stat.st_mtime_ns}:{stat.st_size}"

    @classmethod
    def validate_inputs(cls, video_path: str):
        normalized_path = normalize_video_path(video_path)
        if not normalized_path:
            return "Video path cannot be empty"
        if not os.path.isfile(normalized_path):
            return f"Invalid video file: {video_path} (resolved to: {normalized_path})"
        if not is_video_file(normalized_path):
            return f"File does not contain readable video: {video_path}"
        return True


class UC_LoadVideoDirectory(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_LoadVideoDirectory",
            display_name="Load Videos (Directory)",
            category="advanced/video",
            search_aliases=["import videos directory", "open videos folder", "video folder"],
            inputs=[
                io.String.Input(
                    "directory_path",
                    multiline=False,
                    placeholder="path/to/video-directory",
                    tooltip="Directory containing video files. Files are sorted alphabetically.",
                ),
                io.Int.Input(
                    "start_index",
                    default=0,
                    min=0,
                    step=1,
                    tooltip="Index of the first video to load in sorted order.",
                ),
                io.Int.Input(
                    "load_count",
                    default=1,
                    min=1,
                    max=1024,
                    step=1,
                    tooltip="Number of videos to load.",
                ),
            ],
            outputs=[io.Video.Output(display_name="VIDEO", is_output_list=True)],
        )

    @classmethod
    def execute(cls, directory_path: str, start_index: int, load_count: int) -> io.NodeOutput:
        normalized_path = normalize_video_path(directory_path)
        if not normalized_path or not os.path.isdir(normalized_path):
            raise ValueError(f"Invalid video directory: {directory_path}")
        files = list_video_files(normalized_path)
        end_index = int(start_index) + int(load_count)
        selected = files[int(start_index):end_index]
        if not selected:
            raise ValueError(
                f"No videos found in range [{start_index}:{end_index}] in directory: {directory_path}"
            )
        return io.NodeOutput([InputImpl.VideoFromFile(path) for path in selected])

    @classmethod
    def IS_CHANGED(cls, directory_path: str, start_index: int, load_count: int):
        normalized_path = normalize_video_path(directory_path)
        if not normalized_path or not os.path.isdir(normalized_path):
            return ""
        selected = list_video_files(normalized_path)[int(start_index):int(start_index) + int(load_count)]
        digest = hashlib.sha256()
        for path in selected:
            stat = os.stat(path)
            digest.update(path.encode("utf-8"))
            digest.update(str(stat.st_mtime_ns).encode("ascii"))
            digest.update(str(stat.st_size).encode("ascii"))
        return digest.hexdigest()

    @classmethod
    def VALIDATE_INPUTS(cls, directory_path: str, start_index: int, load_count: int):
        normalized_path = normalize_video_path(directory_path)
        if not normalized_path or not os.path.isdir(normalized_path):
            return f"Invalid video directory: {directory_path}"
        if int(start_index) < 0 or int(load_count) < 1:
            return "Video directory start_index must be >= 0 and load_count must be >= 1"
        if not list_video_files(normalized_path)[int(start_index):int(start_index) + int(load_count)]:
            return f"No videos found in requested range: {directory_path}"
        return True


class UC_SaveVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_SaveVideo",
            display_name="Save Video (Keyframe Optimized)",
            category="advanced/video",
            search_aliases=["save video", "export video", "video combine", "nvenc video"],
            description="Saves video with proper codec keyframes (IDR) placed at scene cuts, matching Avidemux NVENC / x264 GOP settings.",
            inputs=[
                io.Image.Input("images", tooltip="Input image sequence to encode into video."),
                io.Float.Input("fps", default=24.0, min=0.01, max=240.0, step=0.01, tooltip="Frame rate of the output video."),
                io.Combo.Input(
                    "codec",
                    options=["auto", "h264_nvenc", "hevc_nvenc", "libx264", "libx265"],
                    default="auto",
                    tooltip="Video encoder. Auto selects hardware NVENC if available on NVIDIA GPUs, falling back to libx264.",
                ),
                io.Int.Input(
                    "gop_size",
                    default=100,
                    min=1,
                    max=1000,
                    step=1,
                    tooltip="Maximum distance between keyframes (GOP size). Default 100 matches Avidemux.",
                ),
                io.Int.Input(
                    "b_frames",
                    default=0,
                    min=0,
                    max=16,
                    step=1,
                    tooltip="Maximum consecutive B-frames. Default 0 matches Avidemux.",
                ),
                io.Float.Input(
                    "bitrate_mbps",
                    default=10.0,
                    min=0.1,
                    max=200.0,
                    step=0.5,
                    tooltip="Video bitrate in Mbps for NVENC. Default 10 Mbps matches Avidemux.",
                ),
                io.Float.Input(
                    "crf",
                    default=19.0,
                    min=0.0,
                    max=51.0,
                    step=1.0,
                    tooltip="Constant Rate Factor for software encoders (libx264/libx265). Lower is higher quality.",
                ),
                io.Audio.Input(
                    "audio",
                    optional=True,
                    tooltip="Optional audio track to mux into the video.",
                ),
                io.String.Input(
                    "filename_prefix",
                    default="video/%date:yyyy-MM-dd%/vid_%date:yyyy-MM-dd%_default",
                    tooltip="Prefix path in output folder. Supports date tokens like %date:yyyy-MM-dd%.",
                ),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=True,
            outputs=[
                io.Image.Output("images", display_name="IMAGES"),
                io.String.Output("filepath", display_name="Filepath"),
            ],
        )

    @classmethod
    def execute(
        cls,
        images: torch.Tensor,
        fps: float = 24.0,
        codec: str = "auto",
        gop_size: int = 100,
        b_frames: int = 0,
        bitrate_mbps: float = 10.0,
        crf: float = 19.0,
        audio: Any = None,
        filename_prefix: str = "video/%date:yyyy-MM-dd%/vid_%date:yyyy-MM-dd%_default",
    ) -> io.NodeOutput:
        prompt = cls.hidden.prompt if cls.hidden is not None else None
        extra_pnginfo = cls.hidden.extra_pnginfo if cls.hidden is not None else None
        output_path, file, subfolder = save_keyframe_optimized_video(
            images=images,
            fps=fps,
            codec=codec,
            gop_size=gop_size,
            b_frames=b_frames,
            bitrate_mbps=bitrate_mbps,
            crf=crf,
            audio=audio,
            filename_prefix=filename_prefix,
            prompt=prompt,
            extra_pnginfo=extra_pnginfo,
        )
        return io.NodeOutput(
            images,
            output_path,
            ui=ui.PreviewVideo([ui.SavedResult(file, subfolder, io.FolderType.output)]),
        )

