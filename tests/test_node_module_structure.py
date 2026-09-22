import ast
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
LEGACY_MODULE_FUNCTIONS = {
    "encoder_nodes.py": {
        "apply_parallel_ref_latents",
        "multiply_conditioning",
        "system_prompt_template",
        "_mark_deprecated_node",
    },
    "image_nodes.py": {
        "_blend_luminosity",
        "_blend_saturation",
        "_clip_blend_color",
        "_set_blend_luminosity",
        "_set_blend_saturation",
        "_blend_rgb",
    },
    "logic_math_nodes.py": {
        "_math_template",
        "_math_autogrow",
        "_boolean_autogrow",
    },
    "preset_nodes.py": {"escape_prompt_parentheses"},
    "textgen_nodes.py": {
        "evaluate_formula",
        "process_vlm_image",
        "_aligned_image_values",
        "_qwen_clip_model",
        "_token_rows",
        "_visual_grid_metadata",
        "_load_qwen_generation_model",
        "_encode_qwen_visual_sources",
        "_fuse_qwen_primary_sources",
        "generate_fused_qwen3vl",
        "generate_fused_qwen35",
        "detect_textgen_template",
    },
}


def test_node_modules_add_no_standalone_helpers():
    violations = {}
    for path in sorted((REPOSITORY_ROOT / "nodes").glob("*_nodes.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        actual = {
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        unexpected = actual - LEGACY_MODULE_FUNCTIONS.get(path.name, set())
        if unexpected:
            violations[path.name] = sorted(unexpected)

    assert not violations, (
        "Move standalone node-module helpers into focused *_helpers.py modules; "
        f"do not expand the legacy allowlist: {violations}"
    )
