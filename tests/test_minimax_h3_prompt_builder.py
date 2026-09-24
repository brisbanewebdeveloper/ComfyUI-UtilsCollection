import json
import pathlib
import sys
import types

import pytest


PACKAGE_NAME = "utils_collection_h3_prompt_test"
package = types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(pathlib.Path(__file__).resolve().parents[1])]
sys.modules.setdefault(PACKAGE_NAME, package)

from utils_collection_h3_prompt_test.helpers.text_helpers import compile_h3_prompt, default_h3_prompt_state
from utils_collection_h3_prompt_test.nodes.utils_nodes import UC_MiniMaxH3DynamicPromptBuilder


def test_node_exposes_socketless_editor_state_and_string_output():
    schema = UC_MiniMaxH3DynamicPromptBuilder.define_schema()
    assert schema.node_id == "UC_MiniMaxH3DynamicPromptBuilder"
    assert schema.inputs[0].id == "prompt_state"
    assert schema.inputs[0].extra_dict["socketless"] is True
    assert schema.outputs[0].id == "prompt"
    assert UC_MiniMaxH3DynamicPromptBuilder.execute(json.dumps(default_h3_prompt_state()))[0] == ""


def test_headers_and_chronological_channels_compile_without_empty_fields():
    state = default_h3_prompt_state()
    state["headers"] = [
        {"name": "subject_definitions", "text": "- <Subject 1>: A dancer."},
        {"name": "summary", "text": "A stage performance."},
        {"name": "overall_soundscape", "text": "  "},
    ]
    state["segments"] = [{
        "start": "0", "end": "01:02.5",
        "channels": {
            "visual": {"enabled": True, "text": "Camera follows <Subject 1>."},
            "speech": {"enabled": True, "text": "<Subject 1>: \"Go.\""},
            "sounds": {"enabled": True, "text": "Footsteps."},
            "music": {"enabled": False, "text": "Saved draft."},
        },
    }]
    assert compile_h3_prompt(json.dumps(state)) == (
        "subject_definitions:\n- <Subject 1>: A dancer.\n\n"
        "summary:\nA stage performance.\n\n"
        "Timeline:\n[00:00.00-01:02.50]:\n"
        "[VISUAL]: Camera follows <Subject 1>.\n"
        "[SPEECH]: <Subject 1>: \"Go.\"\n"
        "[SOUNDS]: Footsteps."
    )


def test_three_decimal_timeline_allows_gaps_and_omits_empty_segment():
    state = default_h3_prompt_state()
    state["precision"] = 3
    empty = {name: {"enabled": name == "visual", "text": ""} for name in ("visual", "speech", "sounds", "music")}
    state["segments"] = [
        {"start": "0", "end": "5", "channels": empty},
        {"start": "7.125", "end": "8.500", "channels": {
            **empty, "music": {"enabled": True, "text": "Low strings."},
        }},
    ]
    assert compile_h3_prompt(json.dumps(state)) == (
        "Timeline:\n[00:07.125-00:08.500]:\n[MUSIC]: Low strings."
    )


@pytest.mark.parametrize("start,end,message", [
    ("00:60", "61", "seconds must be below 60"),
    ("2", "2", "end must be later"),
    ("0.123", "1", "select 3-decimal precision"),
])
def test_invalid_active_segment_reports_specific_time(start, end, message):
    state = default_h3_prompt_state()
    state["segments"] = [{
        "start": start, "end": end,
        "channels": {name: {"enabled": name == "visual", "text": "Move" if name == "visual" else ""}
                     for name in ("visual", "speech", "sounds", "music")},
    }]
    with pytest.raises(ValueError, match=message):
        compile_h3_prompt(json.dumps(state))
