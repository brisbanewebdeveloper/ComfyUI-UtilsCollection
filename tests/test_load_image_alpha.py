import hashlib
import pathlib
import sys
import types


CUSTOM_NODE_ROOT = pathlib.Path(__file__).parents[1]
PACKAGE_NAME = "utils_collection_load_image_alpha_test"
package = types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(CUSTOM_NODE_ROOT)]
sys.modules.setdefault(PACKAGE_NAME, package)

from utils_collection_load_image_alpha_test.nodes import image_nodes


def test_load_image_alpha_v3_hooks_accept_annotated_input(monkeypatch, tmp_path):
    annotated_name = "clipspace-painted-masked-123.png [input]"
    image_path = tmp_path / "clipspace-painted-masked-123.png"
    image_path.write_bytes(b"edited-mask")

    monkeypatch.setattr(
        image_nodes.folder_paths,
        "exists_annotated_filepath",
        lambda value: value == annotated_name,
    )
    monkeypatch.setattr(
        image_nodes.folder_paths,
        "get_annotated_filepath",
        lambda value: str(image_path),
    )

    assert image_nodes.UC_LoadImageWithAlpha.validate_inputs(annotated_name) is True
    assert image_nodes.UC_LoadImageWithAlpha.fingerprint_inputs(
        annotated_name
    ) == hashlib.sha256(b"edited-mask").hexdigest()
