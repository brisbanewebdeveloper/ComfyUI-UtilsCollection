import pathlib
import sys
import types


CUSTOM_NODE_ROOT = pathlib.Path(__file__).parents[1]
PACKAGE_NAME = "utils_collection_composite_guide_test"
package = types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(CUSTOM_NODE_ROOT)]
sys.modules.setdefault(PACKAGE_NAME, package)

from comfy.cli_args import args as cli_args

prior_cpu = cli_args.cpu
cli_args.cpu = True
try:
    from utils_collection_composite_guide_test.nodes.utils_nodes import (
        UC_CompositeNodesGuide,
    )
finally:
    cli_args.cpu = prior_cpu


EXPECTED_TOPICS = [
    "node_catalog",
    "model_inputs",
    "mask_cleanup_and_resize",
    "background_removal_alpha",
    "unified_background_replace",
    "layered_composite",
    "staged_workflow",
    "staged_individual_workflow",
    "staged_face_workflow",
    "placement_editor",
    "paint_layer",
    "mediapipe_face_composite",
]


def test_composite_guide_topics_render_markdown():
    topic_input = {
        value.id: value for value in UC_CompositeNodesGuide.define_schema().inputs
    }["topic"]

    assert topic_input.options == EXPECTED_TOPICS
    assert topic_input.default == "node_catalog"
    for topic in EXPECTED_TOPICS:
        markdown = UC_CompositeNodesGuide.execute(topic).args[0]
        assert markdown.startswith("## ")
        assert "Unknown topic" not in markdown


def test_composite_guide_documents_optional_model_contract():
    markdown = UC_CompositeNodesGuide.execute("model_inputs").args[0]

    assert "`background_removal_model_opt`" in markdown
    assert "`face_detection_model_opt`" in markdown
    assert "`birefnet.safetensors`" in markdown
    assert "`mediapipe_face_fp32.safetensors`" in markdown
    assert "connected Lucida model" in markdown


def test_composite_guide_documents_automatic_staging_and_face_defaults():
    staged = UC_CompositeNodesGuide.execute("staged_workflow").args[0]
    individual = UC_CompositeNodesGuide.execute("staged_individual_workflow").args[0]
    face = UC_CompositeNodesGuide.execute("staged_face_workflow").args[0]

    assert "automatic stage fingerprint" in staged
    assert "without rerunning background removal" in staged
    assert "without loading either model" in face
    assert "`detection_threshold=0.55`" in face
    assert "`maximum_faces=16`" in face
    assert "never stacked" in individual


def test_composite_guide_documents_alpha_preserving_removal():
    markdown = UC_CompositeNodesGuide.execute("background_removal_alpha").args[0]

    assert "source-resolution RGBA" in markdown
    assert "soft model mask" in markdown
    assert "bypass model execution" in markdown


def test_composite_guide_documents_resolved_transform_preview_geometry():
    placement = UC_CompositeNodesGuide.execute("placement_editor").args[0]

    assert "expanded transformed raster" in placement
    assert "same off-canvas limits" in placement
    assert "crop metadata" in placement
    assert "three complete rows" in placement
