from typing import Any


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
