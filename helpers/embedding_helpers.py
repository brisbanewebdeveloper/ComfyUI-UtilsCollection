from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional

from comfy.sd1_clip import load_embed


_COMPARISON_CHUNK_SIZE = 4096


@dataclass(frozen=True)
class TokenizerBranch:
    path: tuple[str, ...]
    tokenizer: Any
    embedding_key: str
    embedding_size: int
    inverse_vocabulary: dict[int, str]

    @property
    def label(self) -> str:
        return ".".join(self.path) or "<root>"


@dataclass(frozen=True)
class ModelBranch:
    path: tuple[str, ...]
    embedding_module: Any
    vocabulary_size: int
    embedding_size: int

    @property
    def label(self) -> str:
        return ".".join(self.path) or "<root>"


@dataclass(frozen=True)
class TokenCandidate:
    token_id: int
    value: float
    norm: float


def _object_children(value: Any) -> list[tuple[str, Any]]:
    children = []
    if isinstance(value, torch.nn.Module):
        children.extend(value.named_children())
    else:
        for name, child in vars(value).items() if hasattr(value, "__dict__") else ():
            if name.startswith("_") or isinstance(
                child,
                (str, bytes, int, float, bool, type(None), torch.Tensor),
            ):
                continue
            children.append((name, child))
    return children


def _inverse_vocabulary(tokenizer: Any) -> dict[int, str]:
    existing = getattr(tokenizer, "inv_vocab", None)
    if isinstance(existing, dict) and existing:
        return {int(token_id): str(token) for token_id, token in existing.items()}

    inner = getattr(tokenizer, "tokenizer", None)
    get_vocab = getattr(inner, "get_vocab", None)
    if not callable(get_vocab):
        return {}
    return {int(token_id): str(token) for token, token_id in get_vocab().items()}


def discover_tokenizer_branches(tokenizer_root: Any) -> list[TokenizerBranch]:
    branches = []
    visited = set()

    def visit(value: Any, path: tuple[str, ...], depth: int) -> None:
        identity = id(value)
        if identity in visited or depth > 5:
            return
        visited.add(identity)

        embedding_key = getattr(value, "embedding_key", None)
        embedding_size = getattr(value, "embedding_size", None)
        inverse_vocabulary = _inverse_vocabulary(value)
        if (
            isinstance(embedding_key, str)
            and isinstance(embedding_size, int)
            and embedding_size > 0
            and inverse_vocabulary
        ):
            branches.append(
                TokenizerBranch(
                    path=path,
                    tokenizer=value,
                    embedding_key=embedding_key,
                    embedding_size=embedding_size,
                    inverse_vocabulary=inverse_vocabulary,
                )
            )
            return

        for name, child in _object_children(value):
            visit(child, (*path, name), depth + 1)

    visit(tokenizer_root, (), 0)
    return sorted(branches, key=lambda branch: (branch.label, branch.embedding_key))


def _input_embedding_module(value: Any) -> Any | None:
    transformer = getattr(value, "transformer", None)
    getter = getattr(transformer, "get_input_embeddings", None)
    if callable(getter):
        return getter()

    getter = getattr(value, "get_input_embeddings", None)
    if callable(getter):
        module = getter()
        if hasattr(module, "weight"):
            return module
    return None


def discover_model_branches(model_root: Any) -> list[ModelBranch]:
    branches = []
    visited = set()
    seen_modules = set()

    def visit(value: Any, path: tuple[str, ...], depth: int) -> None:
        identity = id(value)
        if identity in visited or depth > 5:
            return
        visited.add(identity)

        module = _input_embedding_module(value)
        weight = getattr(module, "weight", None)
        if module is not None and torch.is_tensor(weight) and weight.ndim == 2:
            module_identity = id(module)
            if module_identity not in seen_modules:
                seen_modules.add(module_identity)
                branches.append(
                    ModelBranch(
                        path=path,
                        embedding_module=module,
                        vocabulary_size=int(weight.shape[0]),
                        embedding_size=int(weight.shape[1]),
                    )
                )
            return

        for name, child in _object_children(value):
            visit(child, (*path, name), depth + 1)

    visit(model_root, (), 0)
    return sorted(branches, key=lambda branch: branch.label)


def _decode(tokenizer: Any, token_ids: list[int]) -> str:
    decode = getattr(tokenizer, "decode", None)
    if callable(decode):
        try:
            return str(decode(token_ids, skip_special_tokens=False))
        except TypeError:
            return str(decode(token_ids, False))

    inner = getattr(tokenizer, "tokenizer", None)
    decode = getattr(inner, "decode", None)
    if callable(decode):
        return str(decode(token_ids, skip_special_tokens=False))
    return ""


def _lookup_vectors(module: Any, token_ids: torch.Tensor) -> torch.Tensor:
    try:
        vectors = module(token_ids.unsqueeze(0), out_dtype=torch.float32)
    except TypeError:
        vectors = module(token_ids.unsqueeze(0))
    if not torch.is_tensor(vectors):
        raise TypeError("input embedding module did not return a tensor")
    return vectors.reshape(-1, vectors.shape[-1]).to(dtype=torch.float32)


def nearest_vocabulary_tokens(
    query_rows: torch.Tensor,
    model_branch: ModelBranch,
    token_ids: list[int],
    top_k: int,
    metric: str,
    chunk_size: int = _COMPARISON_CHUNK_SIZE,
) -> list[list[TokenCandidate]]:
    if metric not in {"cosine", "euclidean", "dot_product"}:
        raise ValueError(f"unsupported similarity metric: {metric}")
    if not token_ids:
        raise ValueError("tokenizer vocabulary contains no compatible token IDs")
    if query_rows.ndim != 2 or query_rows.shape[1] != model_branch.embedding_size:
        raise ValueError(
            f"query shape {tuple(query_rows.shape)} does not match model embedding "
            f"dimension {model_branch.embedding_size}"
        )

    weight = model_branch.embedding_module.weight
    if weight.device.type == "meta":
        raise RuntimeError("input embedding weights are not materialized")
    device = weight.device
    queries = query_rows.detach().to(device=device, dtype=torch.float32)
    query_count = queries.shape[0]
    result_count = min(int(top_k), len(token_ids))
    best_rank_values = torch.full(
        (query_count, result_count),
        -torch.inf,
        device=device,
        dtype=torch.float32,
    )
    best_token_ids = torch.full(
        (query_count, result_count),
        -1,
        device=device,
        dtype=torch.long,
    )
    best_norms = torch.zeros(
        (query_count, result_count),
        device=device,
        dtype=torch.float32,
    )
    normalized_queries = functional.normalize(queries, dim=-1, eps=1e-12)

    with torch.inference_mode():
        for start in range(0, len(token_ids), chunk_size):
            chunk = token_ids[start : start + chunk_size]
            chunk_ids = torch.tensor(chunk, device=device, dtype=torch.long)
            candidates = _lookup_vectors(model_branch.embedding_module, chunk_ids)
            if candidates.shape[1] != model_branch.embedding_size:
                raise ValueError(
                    f"model returned embedding dimension {candidates.shape[1]}, "
                    f"expected {model_branch.embedding_size}"
                )
            candidate_norms = torch.linalg.vector_norm(candidates, dim=-1)

            if metric == "cosine":
                rank_values = normalized_queries @ functional.normalize(
                    candidates,
                    dim=-1,
                    eps=1e-12,
                ).T
            elif metric == "dot_product":
                rank_values = queries @ candidates.T
            else:
                rank_values = -torch.cdist(queries, candidates)

            local_count = min(result_count, len(chunk))
            local_values, local_indices = torch.topk(
                rank_values,
                k=local_count,
                dim=-1,
            )
            local_ids = chunk_ids[local_indices]
            local_norms = candidate_norms[local_indices]

            combined_values = torch.cat((best_rank_values, local_values), dim=-1)
            combined_ids = torch.cat((best_token_ids, local_ids), dim=-1)
            combined_norms = torch.cat((best_norms, local_norms), dim=-1)
            best_rank_values, selection = torch.topk(
                combined_values,
                k=result_count,
                dim=-1,
            )
            best_token_ids = torch.gather(combined_ids, 1, selection)
            best_norms = torch.gather(combined_norms, 1, selection)

    output = []
    for row in range(query_count):
        row_candidates = []
        for rank in range(result_count):
            rank_value = float(best_rank_values[row, rank].cpu())
            row_candidates.append(
                TokenCandidate(
                    token_id=int(best_token_ids[row, rank].cpu()),
                    value=-rank_value if metric == "euclidean" else rank_value,
                    norm=float(best_norms[row, rank].cpu()),
                )
            )
        output.append(row_candidates)
    return output


def _compatible_model_branches(
    tokenizer_branch: TokenizerBranch,
    model_branches: list[ModelBranch],
) -> tuple[list[ModelBranch], str]:
    dimension_matches = [
        branch
        for branch in model_branches
        if branch.embedding_size == tokenizer_branch.embedding_size
    ]
    exact = [
        branch
        for branch in dimension_matches
        if branch.path == tokenizer_branch.path
        or (
            branch.path
            and branch.path[-1] == tokenizer_branch.embedding_key
        )
    ]
    if exact:
        return exact, "path/key"
    return dimension_matches, "dimension"


def _special_token_ids(tokenizer: Any) -> set[int]:
    identifiers = set()
    for name in ("start_token", "end_token", "pad_token"):
        value = getattr(tokenizer, name, None)
        if isinstance(value, int):
            identifiers.add(value)
    inner = getattr(tokenizer, "tokenizer", None)
    for value in getattr(inner, "all_special_ids", ()):
        if isinstance(value, int):
            identifiers.add(value)
    return identifiers


def _latin_decoded_token(value: str) -> bool:
    if not value:
        return False
    has_latin = any("a" <= char.lower() <= "z" for char in value)
    if has_latin:
        return not any(char.isalpha() and not char.isascii() for char in value)
    return all(char.isspace() or char.isascii() and not char.isalnum() for char in value)


def cwb_greedy_token_alignment(
    candidates_by_row: list[list[TokenCandidate]],
    special_ids: set[int],
) -> list[TokenCandidate | None]:
    """Globally select non-special candidate IDs with CWB's greedy one-to-one rule."""
    edges = sorted(
        (
            (-candidate.value, row_index, candidate.token_id, candidate)
            for row_index, candidates in enumerate(candidates_by_row)
            for candidate in candidates
            if candidate.token_id not in special_ids
        ),
        key=lambda edge: edge[:3],
    )
    selected: list[TokenCandidate | None] = [None] * len(candidates_by_row)
    used_token_ids = set()
    for _, row_index, token_id, candidate in edges:
        if selected[row_index] is None and token_id not in used_token_ids:
            selected[row_index] = candidate
            used_token_ids.add(token_id)
    return selected


def analyze_embedding_file(
    clip: Any,
    embedding_path: str,
    top_k: int,
    metric: str,
    alignment_candidates: int = 64,
    latin_only: bool = False,
) -> tuple[str, str, str]:
    path = Path(embedding_path)
    tokenizer_branches = discover_tokenizer_branches(clip.tokenizer)
    model_branches = discover_model_branches(clip.cond_stage_model)
    report = [
        f"Embedding: {path.name}",
        f"Path: {path}",
        f"Metric: {metric}",
        f"Tokenizer branches: {len(tokenizer_branches)}",
        f"Model branches: {len(model_branches)}",
    ]
    approximations = []
    cwb_approximations = []
    analyzed = 0
    loaded_cache: dict[tuple[str, int], Any] = {}

    if not tokenizer_branches:
        report.append("No tokenizer branches expose embedding metadata.")
    if not model_branches:
        report.append("No model branches expose input embedding matrices.")

    for tokenizer_branch in tokenizer_branches:
        report.extend(("", f"Tokenizer branch: {tokenizer_branch.label}"))
        report.append(f"Embedding key: {tokenizer_branch.embedding_key}")
        report.append(f"Embedding size: {tokenizer_branch.embedding_size}")
        cache_key = (
            tokenizer_branch.embedding_key,
            tokenizer_branch.embedding_size,
        )
        if cache_key not in loaded_cache:
            loaded_cache[cache_key] = load_embed(
                path.name,
                [str(path.parent)],
                tokenizer_branch.embedding_size,
                tokenizer_branch.embedding_key,
            )
        embedding = loaded_cache[cache_key]
        if not torch.is_tensor(embedding):
            report.append("Status: no compatible tensor loaded for this branch")
            continue
        if embedding.ndim == 0 or embedding.shape[-1] != tokenizer_branch.embedding_size:
            report.append(f"Status: incompatible tensor shape {tuple(embedding.shape)}")
            continue

        rows = embedding.detach().reshape(-1, embedding.shape[-1])
        report.append(f"Tensor shape: {tuple(embedding.shape)}")
        report.append(f"Rows: {rows.shape[0]}")
        compatible, pairing_method = _compatible_model_branches(
            tokenizer_branch,
            model_branches,
        )
        if not compatible:
            report.append("Status: no model branch with matching embedding dimension")
            continue

        special_ids = _special_token_ids(tokenizer_branch.tokenizer)
        for model_branch in compatible:
            valid_token_ids = sorted(
                token_id
                for token_id in tokenizer_branch.inverse_vocabulary
                if 0 <= token_id < model_branch.vocabulary_size
            )
            analysis_token_ids = valid_token_ids
            if latin_only:
                analysis_token_ids = [
                    token_id
                    for token_id in valid_token_ids
                    if _latin_decoded_token(
                        _decode(tokenizer_branch.tokenizer, [token_id])
                    )
                ]
            branch_name = f"{tokenizer_branch.label} -> {model_branch.label}"
            report.extend(
                (
                    f"Model branch: {model_branch.label}",
                    f"Pairing: {pairing_method}",
                    f"Vocabulary rows: {model_branch.vocabulary_size}",
                    f"Tokenizer IDs analyzed: {len(analysis_token_ids)}",
                )
            )
            try:
                nearest = nearest_vocabulary_tokens(
                    rows,
                    model_branch,
                    analysis_token_ids,
                    top_k,
                    metric,
                )
            except Exception as error:
                report.append(f"Status: analysis failed: {error}")
                continue

            analyzed += 1
            top_ids = []
            query_norms = torch.linalg.vector_norm(
                rows.to(dtype=torch.float32),
                dim=-1,
            ).tolist()
            value_label = "distance" if metric == "euclidean" else "score"
            for row_index, candidates in enumerate(nearest):
                report.append(f"Row {row_index}: norm={query_norms[row_index]:.8g}")
                top_ids.append(candidates[0].token_id)
                for rank, candidate in enumerate(candidates, start=1):
                    raw = tokenizer_branch.inverse_vocabulary.get(
                        candidate.token_id,
                        "",
                    )
                    decoded = _decode(
                        tokenizer_branch.tokenizer,
                        [candidate.token_id],
                    )
                    special = "yes" if candidate.token_id in special_ids else "no"
                    report.append(
                        f"  {rank}. id={candidate.token_id} raw={raw!r} "
                        f"decoded={decoded!r} {value_label}={candidate.value:.8g} "
                        f"norm={candidate.norm:.8g} special={special}"
                    )

            decoded_sequence = _decode(tokenizer_branch.tokenizer, top_ids)
            report.append(f"Top-1 IDs: {top_ids}")
            report.append(f"Top-1 decoded: {decoded_sequence!r}")
            approximations.append(f"[{branch_name}] {decoded_sequence}")

            if not analysis_token_ids:
                report.append("CWB alignment: no eligible vocabulary tokens.")
                cwb_approximations.append(f"[{branch_name}] ")
                continue

            cwb_nearest = nearest_vocabulary_tokens(
                rows,
                model_branch,
                analysis_token_ids,
                alignment_candidates,
                "cosine",
            )
            aligned = cwb_greedy_token_alignment(cwb_nearest, special_ids)
            aligned_ids = [candidate.token_id for candidate in aligned if candidate]
            aligned_scores = [candidate.value for candidate in aligned if candidate]
            cwb_decoded = _decode(tokenizer_branch.tokenizer, aligned_ids)
            report.extend(
                (
                    f"CWB alignment: candidates={alignment_candidates} latin_only={latin_only}",
                    f"CWB matches: {len(aligned_ids)}/{len(aligned)}",
                    f"CWB IDs: {aligned_ids}",
                    f"CWB scores: {[round(score, 8) for score in aligned_scores]}",
                    f"CWB decoded: {cwb_decoded!r}",
                )
            )
            cwb_approximations.append(f"[{branch_name}] {cwb_decoded}")

    if analyzed == 0:
        report.append("")
        report.append("No compatible tokenizer/model branch was analyzed.")
    return "\n".join(report), "\n".join(approximations), "\n".join(cwb_approximations)
