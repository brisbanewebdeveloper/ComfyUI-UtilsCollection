import hashlib
import os

from comfy_api.latest import InputImpl, io
from ..helpers.video_helpers import is_video_file, list_video_files, normalize_video_path


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
