import json
import re
from typing import Any


H3_PROMPT_CHANNELS = ("visual", "speech", "sounds", "music")
_H3_TIME = re.compile(r"(?:(\d+):)?(\d+)(?:\.(\d{1,3}))?")


def default_h3_prompt_state() -> dict:
    return {"version": 1, "precision": 2, "headers": [], "segments": []}


def _h3_time_milliseconds(value: str, precision: int, label: str) -> int:
    if not isinstance(value, str):
        raise ValueError(f"{label}: enter seconds or MM:SS.")
    match = _H3_TIME.fullmatch(value.strip())
    if match is None:
        raise ValueError(f"{label}: enter seconds or MM:SS.")
    minutes, seconds, fraction = match.groups()
    if minutes is not None and int(seconds) >= 60:
        raise ValueError(f"{label}: seconds must be below 60 in MM:SS.")
    if fraction and len(fraction) > precision:
        raise ValueError(f"{label}: select {len(fraction)}-decimal precision or shorten the time.")
    return (int(minutes or 0) * 60 + int(seconds)) * 1000 + int((fraction or "").ljust(3, "0"))


def _h3_format_time(milliseconds: int, precision: int) -> str:
    minutes, remainder = divmod(milliseconds, 60_000)
    seconds, fraction = divmod(remainder, 1000)
    return f"{minutes:02d}:{seconds:02d}.{fraction:03d}"[:6 + precision]


def compile_h3_prompt(serialized: str) -> str:
    """Compile the prompt editor's saved state into MiniMax H3 text."""
    try:
        state = json.loads(serialized)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("prompt_state: invalid JSON.") from error
    if not isinstance(state, dict) or type(state.get("version")) is not int or state["version"] != 1:
        raise ValueError("prompt_state.version: expected version 1.")
    precision = state.get("precision")
    if type(precision) is not int or precision not in (2, 3):
        raise ValueError("prompt_state.precision: choose 2 or 3 decimals.")
    headers, segments = state.get("headers"), state.get("segments")
    if not isinstance(headers, list) or not isinstance(segments, list):
        raise ValueError("prompt_state: headers and segments must be lists.")
    blocks = []
    for index, header in enumerate(headers, 1):
        if not isinstance(header, dict) or not isinstance(header.get("name"), str) or not isinstance(header.get("text"), str):
            raise ValueError(f"Header {index}: invalid name or text.")
        text = header["text"].strip()
        if not text:
            continue
        name = header["name"].strip().removesuffix(":")
        if not name or "\n" in name or "\r" in name or ":" in name:
            raise ValueError(f"Header {index}: enter one label without a colon.")
        blocks.append(f"{name}:\n{text}")
    timeline = []
    for index, segment in enumerate(segments, 1):
        if not isinstance(segment, dict) or not isinstance(segment.get("channels"), dict):
            raise ValueError(f"Segment {index}: invalid channels.")
        lines = []
        for channel in H3_PROMPT_CHANNELS:
            entry = segment["channels"].get(channel)
            if not isinstance(entry, dict) or type(entry.get("enabled")) is not bool or not isinstance(entry.get("text"), str):
                raise ValueError(f"Segment {index}: invalid {channel} field.")
            if entry["enabled"] and entry["text"].strip():
                lines.append(f"[{channel.upper()}]: {entry['text'].strip()}")
        if not lines:
            continue
        start = _h3_time_milliseconds(segment.get("start"), precision, f"Segment {index} start")
        end = _h3_time_milliseconds(segment.get("end"), precision, f"Segment {index} end")
        if end <= start:
            raise ValueError(f"Segment {index}: end must be later than start.")
        timeline.append(f"[{_h3_format_time(start, precision)}-{_h3_format_time(end, precision)}]:\n" + "\n".join(lines))
    if timeline:
        blocks.append("Timeline:\n" + "\n\n".join(timeline))
    return "\n\n".join(blocks)


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
