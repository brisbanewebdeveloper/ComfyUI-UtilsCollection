import json
import pathlib
import sys
import types

import pytest


PACKAGE_NAME = "utils_collection_h3_prompt_test"
package = types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(pathlib.Path(__file__).resolve().parents[1])]
sys.modules.setdefault(PACKAGE_NAME, package)

from utils_collection_h3_prompt_test.helpers.text_helpers import (
    compile_h3_prompt,
    default_h3_prompt_state,
    compile_h3_base_prompt,
    default_h3_base_prompt_state,
)
from utils_collection_h3_prompt_test.nodes.utils_nodes import (
    UC_MiniMaxH3DynamicPromptBuilder,
    UC_MiniMaxH3BasePromptBuilder,
)


def test_node_exposes_socketless_editor_state_and_string_output():
    schema = UC_MiniMaxH3DynamicPromptBuilder.define_schema()
    assert schema.node_id == "UC_MiniMaxH3DynamicPromptBuilder"
    assert schema.inputs[0].id == "prompt_state"
    assert schema.inputs[0].extra_dict["socketless"] is True
    assert schema.inputs[0].extra_dict["state_version"] == 2
    assert schema.outputs[0].id == "prompt"
    assert UC_MiniMaxH3DynamicPromptBuilder.execute(json.dumps(default_h3_prompt_state()))[0] == ""


def test_standalone_subjects_without_references_and_five_field_t2v_envelope():
    """Verify standalone subjects without image references produce compliant 5-field T2VA envelope."""
    state = default_h3_prompt_state()
    state["definitions"] = [
        {"id": 1, "kind": "subject", "has_ref": False, "text": "a lone astronaut in a reflective white spacesuit."},
        {"id": 2, "kind": "subject", "has_ref": False, "text": "a four-legged metallic robotic rover."},
    ]
    state["summary"] = {
        "task_types": [],
        "text": "Cinematic sci-fi drama. The astronaut leads the rover across a desolate crater.",
    }
    state["retention"] = []  # No retention for standalone T2VA per Principle 4 line 3023
    state["detailed"]["has_timeline"] = True
    state["segments"] = [
        {
            "start": 0.0, "end": 2.333, "has_shot": True, "shot": 1,
            "visual": "[Shot 1] Extreme wide shot. <Subject 1> walks forward followed by <Subject 2>.",
            "sounds": {"enabled": True, "text": "Subtle mechanical servo whirs."},
        },
    ]
    state["overall_soundscape"] = "Desolate Martian wind howling against the helmet visor."
    state["non_diegetic_music"] = "Deep atmospheric analog synth drone."

    compiled = compile_h3_prompt(json.dumps(state))

    assert "retention_analysis:" not in compiled
    assert (
        "subject_definitions:\n"
        "<Subject 1> is a lone astronaut in a reflective white spacesuit.\n"
        "<Subject 2> is a four-legged metallic robotic rover."
    ) in compiled
    assert "summary:\nCinematic sci-fi drama. The astronaut leads the rover across a desolate crater." in compiled
    assert "[00.00s-02.33s]:\n[VISUAL]: [Shot 1] Extreme wide shot. <Subject 1> walks forward followed by <Subject 2>." in compiled


def test_all_reference_types_picture_video_audio_definitions():
    """Verify full-reference mode supporting <Subject N>, <Picture N>, <Video N>, and <Audio N>."""
    state = default_h3_prompt_state()
    state["definitions"] = [
        {"id": 1, "kind": "subject", "has_ref": True, "ref_type": "picture", "ref_index": 1, "text": "A young woman wearing a teal coat."},
        {"id": 1, "kind": "picture", "role": "first_frame"},
        {"id": 2, "kind": "picture", "role": "final_frame"},
        {"id": 1, "kind": "video", "role": "edit"},
        {"id": 1, "kind": "audio", "role": "timbre", "target_subject": 1},
    ]
    state["summary"] = {
        "task_types": ["video editing", "reference generation", "audio reference"],
        "text": "The target video is an edited version of <Video 1> featuring <Subject 1>.",
    }
    state["retention"] = [
        {"label": "<Subject 1>", "marker": "attribute_transfer", "text": "transfers identity from <Picture 1> while retaining motion from <Video 1>."},
        {"label": "<Picture 1>", "marker": "fully_preserved", "text": "first frame anchor at 00.00s."},
        {"label": "<Picture 2>", "marker": "fully_preserved", "text": "final frame anchor at video endpoint."},
        {"label": "<Video 1>", "marker": "partially_preserved", "text": "source video motion and timing are preserved."},
        {"label": "<Audio 1>", "marker": "reference", "text": "vocal timbre of <Subject 1> is referenced."},
    ]
    state["detailed"]["has_timeline"] = True
    state["segments"] = [
        {
            "start": 0.0, "end": 2.333, "has_shot": True, "shot": 1,
            "visual": "Camera cuts to <Subject 1> speaking toward the camera.",
            "speech": {"enabled": True, "speaker": "S1", "language": "English", "text": "Look what we found."},
        },
    ]
    state["overall_soundscape"] = "Subtle outdoor breeze."
    state["non_diegetic_music"] = "Soft orchestral strings."

    compiled = compile_h3_prompt(json.dumps(state))

    expected_defs = (
        "subject_definitions:\n"
        "<Subject 1> is fully referenced in <Picture 1>: A young woman wearing a teal coat.\n"
        "<Picture 1> is the fixed first frame anchor at 00.00s.\n"
        "<Picture 2> is the fixed final frame anchor at video endpoint.\n"
        "<Video 1> is the source video for the target video edit.\n"
        "<Audio 1> is the voice-timbre reference for <Subject 1> (S1)."
    )
    assert expected_defs in compiled
    assert "[video editing + reference generation + audio reference]" in compiled
    assert "<Audio 1>: reference - vocal timbre of <Subject 1> is referenced." in compiled


def test_dynamic_segment_durations_step_snapping():
    """Verify dynamic segment durations are preserved and format cleanly."""
    state = default_h3_prompt_state()
    state["precision"] = 2
    state["detailed"]["has_timeline"] = True
    state["segments"] = [
        {"start": 0.0, "end": 2.333, "visual": "Segment 1 (56 frames = 2.33s)"},
        {"start": 2.333, "end": 5.375, "visual": "Segment 2 (73 frames = 3.04s)"},
        {"start": 5.375, "end": 7.000, "visual": "Segment 3 (39 frames = 1.63s)"},
    ]
    compiled = compile_h3_prompt(json.dumps(state))

    assert "[00.00s-02.33s]:" in compiled
    assert "[02.33s-05.38s]:" in compiled
    assert "[05.38s-07.00s]:" in compiled


def test_three_decimal_precision_formatting():
    state = default_h3_prompt_state()
    state["precision"] = 3
    state["detailed"]["has_timeline"] = True
    state["segments"] = [
        {
            "start": 0.0, "end": 2.333, "has_shot": True, "shot": 1,
            "visual": "<Subject 1> raises their hand.",
        },
    ]
    compiled = compile_h3_prompt(json.dumps(state))
    assert "[00.000s-02.333s]:" in compiled


def test_continuous_detailed_description_without_timeline():
    state = default_h3_prompt_state()
    state["detailed"]["has_timeline"] = False
    state["detailed"]["continuous_text"] = "A steady medium tracking shot of a dancer moving across an empty stage under a single warm spotlight."

    compiled = compile_h3_prompt(json.dumps(state))
    assert "detailed_description:\nA steady medium tracking shot of a dancer moving across an empty stage under a single warm spotlight." in compiled
    assert "Timeline:" not in compiled


def test_base_prompt_builder_node_schema_and_execution():
    schema = UC_MiniMaxH3BasePromptBuilder.define_schema()
    assert schema.node_id == "UC_MiniMaxH3BasePromptBuilder"
    assert schema.inputs[0].id == "prompt_state"
    assert schema.inputs[0].extra_dict["socketless"] is True
    assert schema.inputs[0].extra_dict["state_version"] == 1
    assert schema.outputs[0].id == "prompt"
    assert UC_MiniMaxH3BasePromptBuilder.execute(json.dumps(default_h3_base_prompt_state()))[0] == ""


def test_base_t2va_compilation_timeline_and_continuous():
    # 1. Timeline mode:
    state = default_h3_base_prompt_state()
    state["task"] = "T2VA"
    state["description"]["mode"] = "timeline"
    state["segments"] = [
        {
            "start": 0.0, "end": 2.333, "has_shot": True, "shot": 1,
            "visual": "A baker opens the wooden shutters before dawn.",
            "speech": {"enabled": True, "speaker": "S1", "language": "English", "text": "First batch of the morning."},
            "sounds": {"enabled": True, "text": "Wooden shutters scrape open."},
        },
    ]
    state["overall_soundscape"] = "Wooden shutters scrape open over a quiet street."
    state["non_diegetic_music"] = "A soft acoustic-guitar pattern at a moderate tempo."

    compiled = compile_h3_base_prompt(json.dumps(state))

    assert "How the reference pictures align" not in compiled
    assert "For the target video" not in compiled
    assert "subject_definitions:" not in compiled
    assert "retention_analysis:" not in compiled
    assert "integrated_multimodal_description:\nTimeline:\n[00.00s-02.33s]:\n[VISUAL]: [Shot 1] A baker opens the wooden shutters before dawn.\n[SPEECH]: (S1) <d>[English] First batch of the morning.</d>\n[SOUNDS]: Wooden shutters scrape open." in compiled
    assert "overall_soundscape:\nWooden shutters scrape open over a quiet street." in compiled
    assert "non_diegetic_music:\nA soft acoustic-guitar pattern at a moderate tempo." in compiled

    # 2. Continuous mode:
    state["description"]["mode"] = "continuous"
    state["description"]["continuous_text"] = "[Shot 1] Live-action, cinematic, a baker opens shutters. [Shot 2] At 00:03.500, steam rises."
    compiled_cont = compile_h3_base_prompt(json.dumps(state))
    assert "integrated_multimodal_description:\n[Shot 1] Live-action, cinematic, a baker opens shutters. [Shot 2] At 00:03.500, steam rises." in compiled_cont
    assert "Timeline:" not in compiled_cont


def test_base_i2va_first_frame_instruction():
    state = default_h3_base_prompt_state()
    state["task"] = "I2VA"
    state["description"]["mode"] = "continuous"
    state["description"]["continuous_text"] = "[Shot 1] Live-action, cinematic, the woman shown in <Picture 1> turns toward the window."
    state["overall_soundscape"] = "Train wheels roll steadily."
    state["non_diegetic_music"] = "N/A"

    compiled = compile_h3_base_prompt(json.dumps(state))
    expected_header = "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.\n\n"
    assert compiled.startswith(expected_header)
    assert "integrated_multimodal_description:\n[Shot 1] Live-action, cinematic, the woman shown in <Picture 1> turns toward the window." in compiled
    assert "non_diegetic_music:\nN/A" in compiled


def test_base_fl2va_first_and_last_frame_instruction():
    state = default_h3_base_prompt_state()
    state["task"] = "FL2VA"
    state["duration"] = 8.0
    state["final_shot"] = 2
    state["auto_final_shot"] = False
    state["description"]["mode"] = "continuous"
    state["description"]["continuous_text"] = "[Shot 1] The cyclist begins in Picture 1. [Shot 2] At 00:04.000, lands on Picture 2."
    state["overall_soundscape"] = "Rain falls steadily."
    state["non_diegetic_music"] = "N/A"

    compiled = compile_h3_base_prompt(json.dumps(state))
    expected_header = "How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; Picture 2 (from Shot 2) aligns with the 8.00-second mark of the target video.\n\n"
    assert compiled.startswith(expected_header)


def test_base_l2va_last_frame_instruction():
    state = default_h3_base_prompt_state()
    state["task"] = "L2VA"
    state["duration"] = 6.0
    state["final_shot"] = 1
    state["description"]["mode"] = "continuous"
    state["description"]["continuous_text"] = "[Shot 1] A close shot begins with an intact drinking glass, then lands on <Picture 1>."
    state["overall_soundscape"] = "Glass scrapes and falls."
    state["non_diegetic_music"] = "A low electronic pulse."

    compiled = compile_h3_base_prompt(json.dumps(state))
    expected_header = "How the reference pictures align with the target video — <Picture 1> (from [Shot 1]) aligns with the 6.00-second mark of the target video.\n\n"
    assert compiled.startswith(expected_header)

