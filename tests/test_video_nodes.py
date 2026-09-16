import pathlib
import sys
import types

import pytest


CUSTOM_NODE_ROOT = pathlib.Path(__file__).parents[1]
PACKAGE_NAME = "utils_collection_video_nodes_test"
package = types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(CUSTOM_NODE_ROOT)]
sys.modules.setdefault(PACKAGE_NAME, package)

from utils_collection_video_nodes_test.nodes import video_nodes


def test_load_video_path_schema_and_execution(monkeypatch, tmp_path):
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"video")
    monkeypatch.setattr(video_nodes.InputImpl, "VideoFromFile", lambda path: ("VIDEO", path))

    schema = video_nodes.UC_LoadVideoPath.define_schema()
    assert schema.node_id == "UC_LoadVideoPath"
    assert schema.outputs[0].display_name == "VIDEO"
    output = video_nodes.UC_LoadVideoPath.execute(str(source))
    assert output.args == (("VIDEO", str(source.resolve())),)
    assert video_nodes.UC_LoadVideoPath.validate_inputs(str(source)) is True


def test_load_video_path_rejects_non_video_file(tmp_path):
    source = tmp_path / "not-video.txt"
    source.write_text("not video")
    with pytest.raises(ValueError, match="Unsupported video file extension"):
        video_nodes.UC_LoadVideoPath.execute(str(source))


def test_load_video_directory_returns_sorted_video_list(monkeypatch, tmp_path):
    for name in ("10.mp4", "2.webm", "ignore.png"):
        (tmp_path / name).write_bytes(b"video")
    monkeypatch.setattr(video_nodes.InputImpl, "VideoFromFile", lambda path: path)

    schema = video_nodes.UC_LoadVideoDirectory.define_schema()
    assert schema.outputs[0].is_output_list is True
    output = video_nodes.UC_LoadVideoDirectory.execute(str(tmp_path), 0, 2)
    assert output.args == ([str((tmp_path / "10.mp4").resolve()), str((tmp_path / "2.webm").resolve())],)


def test_load_video_directory_validates_empty_selection(tmp_path):
    with pytest.raises(ValueError, match="No videos found"):
        video_nodes.UC_LoadVideoDirectory.execute(str(tmp_path), 0, 1)


def test_load_video_directory_uses_image_directory_compatibility_hooks():
    assert hasattr(video_nodes.UC_LoadVideoDirectory, "IS_CHANGED")
    assert hasattr(video_nodes.UC_LoadVideoDirectory, "VALIDATE_INPUTS")
