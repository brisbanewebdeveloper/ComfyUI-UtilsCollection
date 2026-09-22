import pathlib
import sys
import types

import av
import pytest


CUSTOM_NODE_ROOT = pathlib.Path(__file__).parents[1]
PACKAGE_NAME = "utils_collection_video_nodes_test"
package = types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(CUSTOM_NODE_ROOT)]
sys.modules.setdefault(PACKAGE_NAME, package)

from utils_collection_video_nodes_test.nodes import video_nodes


@pytest.fixture
def write_video():
    def write(path):
        with av.open(str(path), mode="w", format="mp4") as container:
            stream = container.add_stream("mpeg4", rate=24)
            stream.width = 16
            stream.height = 16
            stream.pix_fmt = "yuv420p"
            frame = av.VideoFrame(16, 16, "rgb24")
            frame.planes[0].update(bytes(frame.planes[0].buffer_size))
            for packet in stream.encode(frame):
                container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        return path
    return write


def test_load_video_path_schema_and_execution(monkeypatch, tmp_path, write_video):
    source = tmp_path / "clip.mp4"
    write_video(source)
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
    with pytest.raises(ValueError, match="does not contain readable video"):
        video_nodes.UC_LoadVideoPath.execute(str(source))


def test_load_video_directory_returns_sorted_video_list(monkeypatch, tmp_path, write_video):
    for name in ("10.mp4", "2.webm"):
        write_video(tmp_path / name)
    (tmp_path / "ignore.png").write_bytes(b"not video")
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


@pytest.mark.parametrize("name", ["clip", "clip.bin", "clip.ogv", "clip.3gp", "clip.nut"])
def test_load_video_accepts_decodable_content_regardless_of_suffix(tmp_path, write_video, name):
    source = write_video(tmp_path / name)
    assert video_nodes.UC_LoadVideoPath.validate_inputs(str(source)) is True
    video = video_nodes.UC_LoadVideoPath.execute(str(source)).args[0]
    assert video.get_dimensions() == (16, 16)
    output = video_nodes.UC_LoadVideoDirectory.execute(str(tmp_path), 0, 1).args[0]
    assert len(output) == 1
    assert output[0].get_dimensions() == (16, 16)


def test_video_extension_does_not_make_invalid_content_readable(tmp_path):
    source = tmp_path / "invalid.mp4"
    source.write_bytes(b"not video")
    assert video_nodes.UC_LoadVideoPath.validate_inputs(str(source)) is not True
    with pytest.raises(ValueError, match="does not contain readable video"):
        video_nodes.UC_LoadVideoPath.execute(str(source))
    assert video_nodes.list_video_files(str(tmp_path)) == []
