import argparse
import ast
import hashlib
import json
from pathlib import Path
import re


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = (
    REPOSITORY_ROOT / "plans" / "minimax_h3_vlm_reference_expansion_audit.md"
)
VARS_PATH = REPOSITORY_ROOT / "minimax_h3_vlm_presets_vars.py"
DICTIONARY_NAME = "minimax_h3_system_instructions_vlm"
PRESETS = (
    ("minimax_h3_last_frame", "MINIMAX_H3_LAST_FRAME"),
    ("minimax_h3_full_reference", "MINIMAX_H3_FULL_REFERENCE"),
    (
        "minimax_h3_minimalist_product_ad_reference",
        "MINIMAX_H3_MINIMALIST_PRODUCT_AD_REFERENCE",
    ),
    ("minimax_h3_brand_promo_reference", "MINIMAX_H3_BRAND_PROMO_REFERENCE"),
    (
        "minimax_h3_stylized_3d_animation_reference",
        "MINIMAX_H3_STYLIZED_3D_ANIMATION_REFERENCE",
    ),
    (
        "minimax_h3_papercraft_stop_motion_reference",
        "MINIMAX_H3_PAPERCRAFT_STOP_MOTION_REFERENCE",
    ),
    ("minimax_h3_paper_collage_reference", "MINIMAX_H3_PAPER_COLLAGE_REFERENCE"),
    ("minimax_h3_music_video_reference", "MINIMAX_H3_MUSIC_VIDEO_REFERENCE"),
    (
        "minimax_h3_coop_game_intro_reference",
        "MINIMAX_H3_COOP_GAME_INTRO_REFERENCE",
    ),
    (
        "minimax_h3_handdrawn_live_action_reference",
        "MINIMAX_H3_HANDDRAWN_LIVE_ACTION_REFERENCE",
    ),
)


def audit_presets() -> dict[str, str]:
    source = AUDIT_PATH.read_text(encoding="utf-8")
    fence = "`" * 3
    pattern = re.compile(
        rf"^### `(minimax_h3_[^`]+)`\n\n{fence}text\n(.*?)\n{fence}$",
        re.MULTILINE | re.DOTALL,
    )
    values = dict(pattern.findall(source))
    expected = [key for key, _ in PRESETS]
    if list(values) != expected:
        raise RuntimeError(
            "Audit preset names or ordering differ from the synchronization contract."
        )
    if any("'''" in value for value in values.values()):
        raise RuntimeError("Audit preset contains the authority literal delimiter.")
    return values


def authority_block(values: dict[str, str], newline: str) -> str:
    delimiter = "'''"
    return "".join(
        f"{variable} = _crlf({delimiter}"
        f"{values[key].replace(chr(13) + chr(10), chr(10)).replace(chr(10), newline)}"
        f"{delimiter}){newline}{newline}"
        for key, variable in PRESETS
    )


def dictionary_entries(newline: str) -> str:
    return "".join(
        f'        "{key}": {variable},{newline}' for key, variable in PRESETS
    )


def updated_authority(source: str, values: dict[str, str], newline: str) -> str:
    variable_states = [f"{variable} = _crlf(" in source for _, variable in PRESETS]
    entry_states = [f'        "{key}": {variable},' in source for key, variable in PRESETS]
    if any(variable_states) and not all(variable_states):
        raise RuntimeError("Authority contains only part of the expansion variables.")
    if any(entry_states) and not all(entry_states):
        raise RuntimeError("Authority contains only part of the expansion dictionary entries.")

    updated = source
    if not all(variable_states):
        marker = "MINIMAX_H3_TIMELINE_FL2VA = _crlf("
        if updated.count(marker) != 1:
            raise RuntimeError("Could not locate the timeline literal insertion marker once.")
        updated = updated.replace(marker, authority_block(values, newline) + marker, 1)
    else:
        delimiter = "'''"
        for key, variable in PRESETS:
            pattern = re.compile(
                rf"^{variable} = _crlf\({delimiter}.*?{delimiter}\)\r\n\r\n",
                re.MULTILINE | re.DOTALL,
            )
            replacement = (
                f"{variable} = _crlf({delimiter}"
                f"{values[key].replace(chr(13) + chr(10), chr(10)).replace(chr(10), newline)}"
                f"{delimiter}){newline}{newline}"
            )
            updated, count = pattern.subn(lambda _: replacement, updated, count=1)
            if count != 1:
                raise RuntimeError(f"Could not locate one authority block for {key}.")

    if not all(entry_states):
        marker = (
            '        "minimax_h3_timeline_fl2va": '
            f"MINIMAX_H3_TIMELINE_FL2VA,{newline}"
        )
        if updated.count(marker) != 1:
            raise RuntimeError("Could not locate the dictionary insertion marker once.")
        updated = updated.replace(marker, dictionary_entries(newline) + marker, 1)
    return updated


def validate_rendered(source: str, values: dict[str, str]) -> None:
    tree = ast.parse(source, filename=str(VARS_PATH))
    assignments = {
        node.targets[0].id: node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    }
    dictionaries = assignments["RUNTIME_DICTIONARIES"]
    if not isinstance(dictionaries, ast.Dict):
        raise RuntimeError("RUNTIME_DICTIONARIES is not a dictionary literal.")
    dictionary_index = [
        ast.literal_eval(key) for key in dictionaries.keys
    ].index(DICTIONARY_NAME)
    mapping = dictionaries.values[dictionary_index]
    if not isinstance(mapping, ast.Dict):
        raise RuntimeError("MiniMax H3 runtime mapping is not a dictionary literal.")
    mapping_keys = [ast.literal_eval(key) for key in mapping.keys]
    expected_order = [
        "minimax_h3_base",
        "minimax_h3_first_last_frame",
        "minimax_h3_reference",
        *[key for key, _ in PRESETS],
        "minimax_h3_timeline_fl2va",
    ]
    if mapping_keys[: len(expected_order)] != expected_order:
        raise RuntimeError("Rendered dictionary order differs from the approved contract.")
    for key, variable in PRESETS:
        expected = values[key].replace("\r\n", "\n").replace("\n", "\r\n")
        assignment = assignments[variable]
        if (
            not isinstance(assignment, ast.Call)
            or not isinstance(assignment.func, ast.Name)
            or assignment.func.id != "_crlf"
            or len(assignment.args) != 1
            or not isinstance(assignment.args[0], ast.Constant)
            or assignment.args[0].value.replace("\n", "\r\n") != expected
        ):
            raise RuntimeError(f"Rendered authority differs from audit for {key}.")
        value_index = mapping_keys.index(key)
        mapping_value = mapping.values[value_index]
        if not isinstance(mapping_value, ast.Name) or mapping_value.id != variable:
            raise RuntimeError(f"Rendered dictionary binding differs for {key}.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Check the MiniMax H3 reference-expansion presets and prepare updates "
            "to vlm_presets_vars.py."
        )
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help=(
            "Update vlm_presets_vars.py. Without this option, show whether it needs "
            "changes without editing it."
        ),
    )
    args = parser.parse_args()

    values = audit_presets()
    original_bytes = VARS_PATH.read_bytes()
    if original_bytes.count(b"\n") != original_bytes.count(b"\r\n"):
        raise RuntimeError("Readable authority is not consistently CRLF encoded.")
    newline = "\r\n"
    original = original_bytes.decode("utf-8")
    updated = updated_authority(original, values, newline)
    validate_rendered(updated, values)
    updated_bytes = updated.encode("utf-8")
    changed = original_bytes != updated_bytes
    report = {
        "status": "validated",
        "changed": changed,
        "preset_count": len(PRESETS),
        "authority": str(VARS_PATH.relative_to(REPOSITORY_ROOT)),
        "before_sha256": hashlib.sha256(original_bytes).hexdigest(),
        "after_sha256": hashlib.sha256(updated_bytes).hexdigest(),
    }
    print(json.dumps(report, indent=2))  # noqa: T201
    if args.apply and changed:
        VARS_PATH.write_bytes(updated_bytes)
        print("Applied MiniMax H3 reference expansion to readable authority.")  # noqa: T201


if __name__ == "__main__":
    main()
