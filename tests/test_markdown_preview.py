import json
import pathlib
import sys
import types


CUSTOM_NODE_ROOT = pathlib.Path(__file__).parents[1]
PACKAGE_NAME = "utils_collection_markdown_preview_test"
package = types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(CUSTOM_NODE_ROOT)]
sys.modules.setdefault(PACKAGE_NAME, package)

from comfy.cli_args import args as cli_args

prior_cpu = cli_args.cpu
cli_args.cpu = True
try:
    from utils_collection_markdown_preview_test.nodes.utils_nodes import (
        UC_ImageToVideoPrompt,
        UC_MarkdownPreview,
    )
finally:
    cli_args.cpu = prior_cpu


def test_markdown_preview_schema_is_permanent_markdown_output_node():
    schema = UC_MarkdownPreview.define_schema()

    assert schema.node_id == "UC_MarkdownPreview"
    assert schema.display_name == "Preview as Markdown"
    assert schema.is_output_node is True
    assert [value.id for value in schema.inputs] == ["source"]
    assert [value.id for value in schema.outputs] == ["text"]


def test_markdown_preview_returns_ui_markdown_and_text_output():
    markdown = "## Heading\n\n- one\n- two"
    output = UC_MarkdownPreview.execute(markdown)

    assert output.args == (markdown,)
    assert output.ui == {"markdown": (markdown,)}


def test_markdown_preview_serializes_non_string_values():
    source = {"enabled": True, "items": [1, 2]}
    expected = json.dumps(source, indent=2)
    output = UC_MarkdownPreview.execute(source)

    assert output.args == (expected,)
    assert output.ui == {"markdown": (expected,)}


def test_markdown_preview_frontend_uses_core_markdown_widget():
    frontend = (CUSTOM_NODE_ROOT / "web" / "markdown_preview.js").read_text(
        encoding="utf-8"
    )

    assert 'nodeData.name !== "UC_MarkdownPreview"' in frontend
    assert "ComfyWidgets.MARKDOWN(" in frontend
    assert "preview_mode" not in frontend


def test_image_to_video_prompt_node_exposes_role_aware_schema():
    schema = UC_ImageToVideoPrompt.define_schema()

    assert schema.node_id == "UC_ImageToVideoPrompt"
    assert schema.display_name == "Image Prompt to Video Prompt"
    assert [value.id for value in schema.inputs] == ["text", "prompt_role"]
    assert schema.inputs[1].options == [
        "general",
        "system",
        "instruction",
        "bonus",
    ]
    assert schema.inputs[1].default == "general"
    assert [value.id for value in schema.outputs] == ["text"]
    assert UC_ImageToVideoPrompt.execute("").args == ("",)
