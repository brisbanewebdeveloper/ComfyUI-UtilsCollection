import argparse
import json
from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    (
        "vlm_presets.py",
        "vlm_presets_vars.py",
        (
            "system_instructions_vlm",
            "system_query_additional_vlm",
            "system_query_raw_vlm",
            "additional_instructions_vlm",
        ),
        (),
    ),
    (
        "vlm_experimental_presets.py",
        "vlm_experimental_presets_vars.py",
        ("system_instructions_vlm_experimental",),
        (),
    ),
    (
        "minimax_h3_vlm_presets.py",
        "minimax_h3_vlm_presets_vars.py",
        ("minimax_h3_system_instructions_vlm",),
        ("minimax_h3_vlm_jailbreak_prefix", "minimax_h3_vlm_jailbreak_suffix"),
    ),
    (
        "minimax_h3_vlm_experimental_presets.py",
        "minimax_h3_vlm_experimental_presets_vars.py",
        ("minimax_h3_system_instructions_vlm_experimental",),
        (),
    ),
)


def to_lf(value: str) -> str:
    return value.replace("\r\n", "\n").replace("\r", "\n")


def to_crlf(value: str) -> str:
    return to_lf(value).replace("\n", "\r\n")


def variable_name(name: str, used: set[str], namespace: str = "") -> str:
    candidate = name.upper()
    if candidate in used:
        candidate = f"{namespace.upper()}__{candidate}"
    if not candidate.isidentifier() or candidate in used:
        raise RuntimeError(f"Cannot derive unique variable name for {name!r}.")
    used.add(candidate)
    return candidate


def readable_literal(value: str) -> str:
    content = to_lf(value).replace("\\", "\\\\")
    delimiter = "'''" if content.count("'''") <= content.count('"""') else '"""'
    content = content.replace(delimiter, "\\" + delimiter)
    return f"_crlf({delimiter}{content}{delimiter})"


def render_authority(runtime: dict, dictionaries: tuple[str, ...], scalars: tuple[str, ...]) -> str:
    lines = [
        '"""Readable authority for generated VLM preset runtime data."""',
        "",
        "",
        "def _crlf(value: str) -> str:",
        '    return value.replace("\\r\\n", "\\n").replace("\\r", "\\n").replace("\\n", "\\r\\n")',
        "",
        "",
    ]
    used: set[str] = set()
    dictionary_variables: dict[str, list[tuple[str, str]]] = {}
    scalar_variables: dict[str, str] = {}

    for dictionary_name in dictionaries:
        mapping = runtime[dictionary_name]
        if not isinstance(mapping, dict):
            raise RuntimeError(f"{dictionary_name} is not a dictionary.")
        pairs = []
        for key, value in mapping.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise RuntimeError(f"{dictionary_name} must contain only string pairs.")
            name = variable_name(key, used, dictionary_name)
            lines.extend((f"{name} = {readable_literal(value)}", ""))
            pairs.append((key, name))
        dictionary_variables[dictionary_name] = pairs

    for scalar_name in scalars:
        value = runtime[scalar_name]
        if not isinstance(value, str):
            raise RuntimeError(f"{scalar_name} is not a string.")
        name = variable_name(scalar_name, used)
        lines.extend((f"{name} = {readable_literal(value)}", ""))
        scalar_variables[scalar_name] = name

    lines.extend(("RUNTIME_DICTIONARIES = {",))
    for dictionary_name, pairs in dictionary_variables.items():
        lines.append(f'    "{dictionary_name}": {{')
        for key, name in pairs:
            lines.append(f"        {json.dumps(key, ensure_ascii=False)}: {name},")
        lines.append("    },")
    lines.extend(("}", "", "RUNTIME_VALUES = {"))
    for scalar_name, name in scalar_variables.items():
        lines.append(f"    {json.dumps(scalar_name, ensure_ascii=False)}: {name},")
    lines.extend(("}", ""))
    return "\n".join(lines)


def render_runtime(authority: dict, dictionaries: tuple[str, ...], scalars: tuple[str, ...]) -> str:
    expected_dictionaries = authority["RUNTIME_DICTIONARIES"]
    expected_values = authority["RUNTIME_VALUES"]
    if tuple(expected_dictionaries) != dictionaries or tuple(expected_values) != scalars:
        raise RuntimeError("Authority names or ordering do not match configuration.")

    lines = ['"""Generated from the matching *_vars.py authority. Do not hand-edit."""', ""]
    for scalar_name in scalars:
        value = expected_values[scalar_name]
        lines.extend((f"{scalar_name} = {json.dumps(value, ensure_ascii=False)}", ""))
    for dictionary_name in dictionaries:
        lines.append(f"{dictionary_name} = {{")
        for key, value in expected_dictionaries[dictionary_name].items():
            lines.append(
                f"    {json.dumps(key, ensure_ascii=False)}: "
                f"{json.dumps(value, ensure_ascii=False)},"
            )
        lines.extend(("}", ""))
    return "\n".join(lines)


def validate_preserved(runtime: dict, authority: dict, dictionaries: tuple[str, ...], scalars: tuple[str, ...]) -> None:
    for dictionary_name in dictionaries:
        old = runtime[dictionary_name]
        new = authority.get("RUNTIME_DICTIONARIES", authority)[dictionary_name]
        if tuple(old) != tuple(new):
            raise RuntimeError(f"Key ordering changed for {dictionary_name}.")
        for key in old:
            if to_lf(old[key]) != to_lf(new[key]):
                raise RuntimeError(f"Non-newline content changed for {dictionary_name}.{key}.")
    for scalar_name in scalars:
        authority_values = authority.get("RUNTIME_VALUES", authority)
        if to_lf(runtime[scalar_name]) != to_lf(authority_values[scalar_name]):
            raise RuntimeError(f"Non-newline content changed for {scalar_name}.")


def validate_exact(runtime: dict, authority: dict, dictionaries: tuple[str, ...], scalars: tuple[str, ...]) -> None:
    for dictionary_name in dictionaries:
        if runtime[dictionary_name] != authority["RUNTIME_DICTIONARIES"][dictionary_name]:
            raise RuntimeError(f"Generated runtime differs from authority for {dictionary_name}.")
    for scalar_name in scalars:
        if runtime[scalar_name] != authority["RUNTIME_VALUES"][scalar_name]:
            raise RuntimeError(f"Generated runtime differs from authority for {scalar_name}.")


def crlf_bytes(source: str) -> bytes:
    return source.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\r\n").encode("utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Synchronize readable *_vars.py files and runtime VLM presets."
    )
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help=(
            "Reverse the normal direction: restore the readable *_vars.py files "
            "from the runtime presets."
        ),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help=(
            "Update runtime presets from the readable *_vars.py files. With "
            "--bootstrap, restore the readable files instead. Without this option, "
            "only show what would change."
        ),
    )
    args = parser.parse_args(argv)

    reports = []
    writes: list[tuple[Path, bytes]] = []
    for runtime_name, authority_name, dictionaries, scalars in CONFIG:
        runtime_path = ROOT / runtime_name
        authority_path = ROOT / authority_name
        runtime = runpy.run_path(str(runtime_path))

        if args.bootstrap:
            authority_source = render_authority(runtime, dictionaries, scalars)
            authority: dict = {}
            exec(compile(authority_source, authority_name, "exec"), authority)
            validate_preserved(runtime, authority, dictionaries, scalars)
            writes.append((authority_path, crlf_bytes(authority_source)))
        else:
            authority = runpy.run_path(str(authority_path))

        runtime_source = render_runtime(authority, dictionaries, scalars)
        generated_runtime: dict = {}
        exec(compile(runtime_source, runtime_name, "exec"), generated_runtime)
        if args.bootstrap:
            validate_preserved(runtime, generated_runtime, dictionaries, scalars)
        validate_exact(generated_runtime, authority, dictionaries, scalars)
        writes.append((runtime_path, crlf_bytes(runtime_source)))
        reports.append(
            {
                "runtime": runtime_name,
                "authority": authority_name,
                "dictionaries": sum(len(authority["RUNTIME_DICTIONARIES"][name]) for name in dictionaries),
                "scalars": len(scalars),
            }
        )

    changed = [str(path.relative_to(ROOT)) for path, data in writes if not path.exists() or path.read_bytes() != data]
    print(json.dumps({"status": "validated", "changed": changed, "files": reports}, indent=2))
    if args.apply:
        for path, data in writes:
            if not path.exists() or path.read_bytes() != data:
                path.write_bytes(data)
        print("Applied authoritative preset generation.")


if __name__ == "__main__":
    main()
