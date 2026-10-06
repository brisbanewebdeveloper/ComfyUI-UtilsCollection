import json
import re
from typing import Any, Union


H3_PROMPT_CHANNELS = ("visual", "speech", "sounds", "music")
_H3_TIME = re.compile(r"(?:(\d+):)?(\d+)(?:\.(\d{1,3}))?")


def default_h3_prompt_state() -> dict:
    return {
        "version": 2,
        "precision": 2,
        "active_tab": "subjects",
        "definitions": [],
        "summary": {
            "task_types": ["reference generation"],
            "text": "",
        },
        "retention": [],
        "detailed": {
            "has_timeline": True,
            "continuous_text": "",
            "segment_duration": 2.333,
        },
        "segments": [],
        "overall_soundscape": "",
        "non_diegetic_music": "N/A",
    }


def _format_seconds(val: Any, precision: int = 2) -> str:
    """Format duration in seconds to padded 00.00s or 00.000s per Principle 4."""
    width = precision + 3
    if isinstance(val, (int, float)):
        sec = max(0.0, float(val))
        return f"{sec:0{width}.{precision}f}s"
    s = str(val).strip()
    if s.endswith("s"):
        s = s[:-1]
    if ":" in s:
        parts = s.split(":")
        try:
            sec = max(0.0, int(parts[0]) * 60 + float(parts[1]))
            return f"{sec:0{width}.{precision}f}s"
        except Exception:
            return f"{val}"
    try:
        sec = max(0.0, float(s))
        return f"{sec:0{width}.{precision}f}s"
    except Exception:
        return f"{val}s"


def compile_h3_prompt(serialized: str) -> str:
    """Compile the interactive H3 prompt builder state into MiniMax H3 text adhering to Principle 4."""
    try:
        state = json.loads(serialized)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("prompt_state: invalid JSON.") from error

    if not isinstance(state, dict):
        raise ValueError("prompt_state: expected dictionary.")

    precision = state.get("precision", 2)
    if not isinstance(precision, int) or precision not in (2, 3):
        raise ValueError("prompt_state.precision: choose 2 or 3 decimals.")

    blocks = []

    # 1. subject_definitions:
    raw_defs = state.get("definitions") or state.get("subjects") or []
    subject_lines = []
    for item in raw_defs:
        if isinstance(item, str) and item.strip():
            subject_lines.append(item.strip())
        elif isinstance(item, dict):
            raw = item.get("raw", "").strip()
            if raw:
                subject_lines.append(raw)
                continue

            kind = item.get("kind", item.get("type", "subject"))
            d_id = item.get("id", 1)
            details = item.get("text", "").strip()
            role = item.get("role", "custom")

            if kind == "subject":
                has_ref = item.get("has_ref")
                if has_ref is None:
                    has_ref = item.get("ref_type") not in (None, "none")

                if has_ref:
                    ref_type = item.get("ref_type", "picture")
                    ref_index = item.get("ref_index", 1)
                    ref_tag = f"<{'Video' if ref_type == 'video' else 'Picture'} {ref_index}>"
                    prefix = f"<Subject {d_id}> is fully referenced in {ref_tag}:"
                    subject_lines.append(f"{prefix} {details}".strip())
                else:
                    if details.startswith("is ") or details.startswith("are "):
                        subject_lines.append(f"<Subject {d_id}> {details}")
                    else:
                        subject_lines.append(f"<Subject {d_id}> is {details}" if details else f"<Subject {d_id}> is a primary visual subject.")

            elif kind == "picture":
                if role == "first_frame":
                    subject_lines.append(f"<Picture {d_id}> is the fixed first frame anchor at 00.00s.")
                elif role == "final_frame":
                    subject_lines.append(f"<Picture {d_id}> is the fixed final frame anchor at video endpoint.")
                elif role == "storyboard":
                    subject_lines.append(f"<Picture {d_id}> is a storyboard reference for [Shot 1] and [Shot 2], defining viewpoint and subject placement.")
                else:
                    subject_lines.append(f"<Picture {d_id}>: {details}" if details else f"<Picture {d_id}>:")

            elif kind == "video":
                if role == "edit":
                    subject_lines.append(f"<Video {d_id}> is the source video for the target video edit.")
                elif role == "continue":
                    subject_lines.append(f"<Video {d_id}> is the source video for continuation.")
                elif role == "structure":
                    subject_lines.append(f"<Video {d_id}> provides camera movement, cuts, and temporal structure.")
                else:
                    subject_lines.append(f"<Video {d_id}>: {details}" if details else f"<Video {d_id}>:")

            elif kind == "audio":
                if role == "timbre":
                    spk = item.get("target_subject", 1)
                    subject_lines.append(f"<Audio {d_id}> is the voice-timbre reference for <Subject {spk}> (S{spk}).")
                elif role == "full":
                    subject_lines.append(f"<Audio {d_id}> is reused as the target video's complete final audio track.")
                elif role == "music":
                    subject_lines.append(f"<Audio {d_id}> is the music-style and rhythm reference.")
                else:
                    subject_lines.append(f"<Audio {d_id}>: {details}" if details else f"<Audio {d_id}>:")

    if subject_lines:
        blocks.append("subject_definitions:\n" + "\n".join(subject_lines))

    # 2. summary:
    summary_data = state.get("summary", {})
    if isinstance(summary_data, str):
        summary_text = summary_data.strip()
        task_types = []
    elif isinstance(summary_data, dict):
        summary_text = summary_data.get("text", "").strip()
        task_types = summary_data.get("task_types", [])
    else:
        summary_text = ""
        task_types = []

    if summary_text:
        if task_types:
            ordered_types = []
            if "video editing" in task_types:
                ordered_types.append("video editing")
            for t in task_types:
                if t != "video editing" and t not in ordered_types:
                    ordered_types.append(t)
            prefix = f"[{' + '.join(ordered_types)}]"
            summary_body = f"{prefix} {summary_text}".strip()
        else:
            summary_body = summary_text
        blocks.append(f"summary:\n{summary_body}")

    # 3. retention_analysis:
    retention = state.get("retention", [])
    retention_lines = []
    for item in retention:
        if isinstance(item, str) and item.strip():
            retention_lines.append(item.strip())
        elif isinstance(item, dict):
            label = item.get("label", "").strip()
            marker = item.get("marker", "").strip()
            desc = item.get("text", "") or item.get("descriptor", "")
            desc = desc.strip()
            if label and marker:
                if desc:
                    retention_lines.append(f"{label}: {marker} - {desc}")
                else:
                    retention_lines.append(f"{label}: {marker}")

    if retention_lines:
        blocks.append("retention_analysis:\n" + "\n".join(retention_lines))

    # 4. detailed_description:
    detailed_data = state.get("detailed", {})
    has_timeline = detailed_data.get("has_timeline", True) if isinstance(detailed_data, dict) else True
    continuous_text = detailed_data.get("continuous_text", "").strip() if isinstance(detailed_data, dict) else ""
    segments = state.get("segments", [])

    if has_timeline and segments:
        timeline_segments = []
        for index, seg in enumerate(segments, 1):
            if not isinstance(seg, dict):
                continue
            start_str = _format_seconds(seg.get("start", 0), precision)
            end_str = _format_seconds(seg.get("end", 0), precision)

            lines = []
            visual = seg.get("visual", "").strip()
            has_shot = seg.get("has_shot", False)
            shot_num = seg.get("shot", index)
            if visual or has_shot:
                if has_shot and not visual.startswith(f"[Shot {shot_num}]"):
                    shot_prefix = f"[Shot {shot_num}] "
                else:
                    shot_prefix = ""
                lines.append(f"[VISUAL]: {shot_prefix}{visual}".strip())

            speech = seg.get("speech", {})
            if isinstance(speech, dict) and speech.get("enabled"):
                speaker = speech.get("speaker", "S1").strip()
                lang = speech.get("language", "English").strip()
                s_text = speech.get("text", "").strip()
                subj = speech.get("subject", "").strip()
                subj_prefix = f"{subj} " if subj else ""
                speaker_tag = f"({speaker}) " if speaker else ""
                d_body = f"<d>[{lang}] {s_text}</d>" if s_text else ""
                lines.append(f"[SPEECH]: {subj_prefix}{speaker_tag}{d_body}".strip())
            elif isinstance(speech, str) and speech.strip():
                lines.append(f"[SPEECH]: {speech.strip()}")

            sounds = seg.get("sounds", {})
            if isinstance(sounds, dict) and sounds.get("enabled"):
                snd_text = sounds.get("text", "").strip()
                if snd_text:
                    lines.append(f"[SOUNDS]: {snd_text}")
            elif isinstance(sounds, str) and sounds.strip():
                lines.append(f"[SOUNDS]: {sounds.strip()}")

            music = seg.get("music", {})
            if isinstance(music, dict) and music.get("enabled"):
                mus_text = music.get("text", "").strip()
                if mus_text:
                    lines.append(f"[MUSIC]: {mus_text}")
            elif isinstance(music, str) and music.strip():
                lines.append(f"[MUSIC]: {music.strip()}")

            channels = seg.get("channels", {})
            if isinstance(channels, dict):
                for ch in ("visual", "speech", "sounds", "music"):
                    ch_entry = channels.get(ch, {})
                    if isinstance(ch_entry, dict) and ch_entry.get("enabled") and ch_entry.get("text", "").strip():
                        tag = ch.upper()
                        if not any(l.startswith(f"[{tag}]:") for l in lines):
                            lines.append(f"[{tag}]: {ch_entry['text'].strip()}")

            if lines:
                timeline_segments.append(f"[{start_str}-{end_str}]:\n" + "\n".join(lines))

        if timeline_segments:
            blocks.append("detailed_description:\nTimeline:\n" + "\n\n".join(timeline_segments))
    elif continuous_text:
        blocks.append(f"detailed_description:\n{continuous_text}")

    # 5. overall_soundscape:
    soundscape = state.get("overall_soundscape", "").strip()
    if soundscape:
        blocks.append(f"overall_soundscape:\n{soundscape}")

    # 6. non_diegetic_music:
    music = state.get("non_diegetic_music", "").strip()
    if music and blocks:
        blocks.append(f"non_diegetic_music:\n{music}")
    elif music and music != "N/A":
        blocks.append(f"non_diegetic_music:\n{music}")

    return "\n\n".join(blocks)


def default_h3_base_prompt_state() -> dict[str, Any]:
    return {
        "version": 1,
        "task": "T2VA",
        "duration": 5.0,
        "final_shot": 1,
        "auto_final_shot": True,
        "precision": 2,
        "active_tab": "task",
        "description": {
            "mode": "timeline",
            "continuous_text": "",
            "segment_duration": 2.333,
        },
        "segments": [],
        "overall_soundscape": "",
        "non_diegetic_music": "N/A",
    }


def compile_h3_base_prompt(prompt_state: Union[str, dict[str, Any]]) -> str:
    if isinstance(prompt_state, str):
        try:
            state = json.loads(prompt_state)
        except Exception:
            return prompt_state
    elif isinstance(prompt_state, dict):
        state = prompt_state
    else:
        return ""

    if not isinstance(state, dict):
        return ""

    task = str(state.get("task", "T2VA")).upper().strip()
    precision = 3 if state.get("precision") == 3 else 2
    duration = float(state.get("duration", 5.0) or 5.0)

    desc_data = state.get("description", {})
    if not isinstance(desc_data, dict):
        desc_data = {}
    mode = desc_data.get("mode", "timeline")
    continuous_text = str(desc_data.get("continuous_text", "")).strip()
    segments = state.get("segments", [])
    if not isinstance(segments, list):
        segments = []

    # Determine final shot index
    final_shot = int(state.get("final_shot", 1) or 1)
    if state.get("auto_final_shot", True) and segments:
        max_shot = 1
        for idx, seg in enumerate(segments, 1):
            if isinstance(seg, dict):
                s = seg.get("shot")
                if s is not None:
                    try:
                        max_shot = max(max_shot, int(s))
                    except (ValueError, TypeError):
                        pass
                elif seg.get("has_shot", False):
                    max_shot = max(max_shot, idx)
        final_shot = max_shot

    # Part One: Keyframe alignment instruction (if any)
    instruction = ""
    dur_str = f"{duration:.2f}"
    if task == "I2VA":
        instruction = "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced."
    elif task == "FL2VA":
        instruction = (
            f"How the reference pictures align with the target video — "
            f"Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; "
            f"Picture 2 (from Shot {final_shot}) aligns with the {dur_str}-second mark of the target video."
        )
    elif task == "L2VA":
        instruction = (
            f"How the reference pictures align with the target video — "
            f"<Picture 1> (from [Shot {final_shot}]) aligns with the {dur_str}-second mark of the target video."
        )

    blocks: list[str] = []

    # Field 1: integrated_multimodal_description:
    if mode == "timeline" and segments:
        timeline_segments = []
        for index, seg in enumerate(segments, 1):
            if not isinstance(seg, dict):
                continue
            start_str = _format_seconds(seg.get("start", 0), precision)
            end_str = _format_seconds(seg.get("end", 0), precision)

            lines = []
            visual = str(seg.get("visual", "")).strip()
            has_shot = seg.get("has_shot", False)
            shot_num = seg.get("shot", index)
            if visual or has_shot:
                if has_shot and not visual.startswith(f"[Shot {shot_num}]"):
                    shot_prefix = f"[Shot {shot_num}] "
                else:
                    shot_prefix = ""
                lines.append(f"[VISUAL]: {shot_prefix}{visual}".strip())

            speech = seg.get("speech", {})
            if isinstance(speech, dict) and speech.get("enabled", False):
                spk = str(speech.get("speaker", "S1")).strip() or "S1"
                lang = str(speech.get("language", "English")).strip() or "English"
                txt = str(speech.get("text", "")).strip()
                lines.append(f"[SPEECH]: ({spk}) <d>[{lang}] {txt}</d>".strip())

            sounds = seg.get("sounds", {})
            if isinstance(sounds, dict) and sounds.get("enabled", False):
                s_txt = str(sounds.get("text", "")).strip()
                if s_txt:
                    lines.append(f"[SOUNDS]: {s_txt}")

            music_chan = seg.get("music", {})
            if isinstance(music_chan, dict) and music_chan.get("enabled", False):
                m_txt = str(music_chan.get("text", "")).strip()
                if m_txt:
                    lines.append(f"[MUSIC]: {m_txt}")

            if lines:
                timeline_segments.append(f"[{start_str}-{end_str}]:\n" + "\n".join(lines))

        if timeline_segments:
            blocks.append("integrated_multimodal_description:\nTimeline:\n" + "\n\n".join(timeline_segments))
    elif continuous_text:
        blocks.append(f"integrated_multimodal_description:\n{continuous_text}")

    # Field 2: overall_soundscape:
    soundscape = str(state.get("overall_soundscape", "")).strip()
    if soundscape:
        blocks.append(f"overall_soundscape:\n{soundscape}")

    # Field 3: non_diegetic_music:
    music = str(state.get("non_diegetic_music", "")).strip()
    if music and blocks:
        blocks.append(f"non_diegetic_music:\n{music}")
    elif music and music != "N/A":
        blocks.append(f"non_diegetic_music:\n{music}")

    if not blocks and not instruction:
        return ""

    parts = []
    if instruction:
        parts.append(instruction)
    if blocks:
        parts.append("\n\n".join(blocks))

    return "\n\n".join(parts)


def concatenate_aligned_text_inputs(
    text_inputs: dict[str, list[Any]] | None,
    delimiter_values: list[Any] | None,
) -> list[str]:
    ordered_inputs = [
        values
        for _, values in sorted(
            (text_inputs or {}).items(),
            key=lambda item: int(item[0].removeprefix("text_")),
        )
    ]
    output_count = max((len(values) for values in ordered_inputs), default=1)
    delimiters = delimiter_values or [""]

    outputs = []
    for index in range(output_count):
        delimiter = delimiters[min(index, len(delimiters) - 1)]
        parts = [
            values[min(index, len(values) - 1)] if values else ""
            for values in ordered_inputs
        ]
        outputs.append(str(delimiter).join(str(value) for value in parts))
    return outputs
