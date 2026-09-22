import pathlib
import sys
import types

import pytest
import torch


CUSTOM_NODE_ROOT = pathlib.Path(__file__).parents[1]
PACKAGE_NAME = "utils_collection_embedding_analysis_test"
package = types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(CUSTOM_NODE_ROOT)]
sys.modules.setdefault(PACKAGE_NAME, package)

from utils_collection_embedding_analysis_test.helpers import embedding_helpers
from utils_collection_embedding_analysis_test.nodes import embedding_nodes


class FakeTokenizerLeaf:
    embedding_key = "clip_l"
    embedding_size = 3
    inv_vocab = {0: "zero", 1: "one", 2: "special", 3: "opposite"}
    start_token = 2
    end_token = None
    pad_token = None

    @staticmethod
    def decode(token_ids, skip_special_tokens=True):
        assert skip_special_tokens is False
        return " ".join(FakeTokenizerLeaf.inv_vocab[token_id] for token_id in token_ids)


class FakeTokenizerRoot:
    def __init__(self):
        self.clip_l = FakeTokenizerLeaf()


class GenericTokenizerLeaf:
    def __init__(self, embedding_key, embedding_size, vocabulary):
        self.embedding_key = embedding_key
        self.embedding_size = embedding_size
        self.inv_vocab = vocabulary
        self.start_token = None
        self.end_token = None
        self.pad_token = None

    def decode(self, token_ids, skip_special_tokens=True):
        assert skip_special_tokens is False
        return " ".join(self.inv_vocab[token_id] for token_id in token_ids)


class FakeTransformer:
    def __init__(self, vectors):
        self.embedding = torch.nn.Embedding.from_pretrained(
            torch.tensor(vectors, dtype=torch.float32),
        )

    def get_input_embeddings(self):
        return self.embedding


class FakeModelLeaf(torch.nn.Module):
    def __init__(self, vectors):
        super().__init__()
        self.transformer = FakeTransformer(vectors)


class FakeModelRoot(torch.nn.Module):
    def __init__(self, vectors):
        super().__init__()
        self.clip_l = FakeModelLeaf(vectors)


class FakeClip:
    def __init__(self, vectors):
        self.tokenizer = FakeTokenizerRoot()
        self.cond_stage_model = FakeModelRoot(vectors)


VOCABULARY = [
    [1.0, 0.0, 0.0],
    [0.1, 0.5, 0.0],
    [0.0, 0.0, 1.0],
    [-1.0, 0.0, 0.0],
]


def _model_branch():
    return embedding_helpers.discover_model_branches(FakeModelRoot(VOCABULARY))[0]


def test_schema_lists_embedding_analysis_controls(monkeypatch):
    monkeypatch.setattr(
        embedding_nodes,
        "get_filename_list",
        lambda category: ["sample.pt"] if category == "embeddings" else [],
    )
    schema = embedding_nodes.UC_EmbeddingDetokenizerAnalysis.define_schema()
    inputs = {value.id: value for value in schema.inputs}

    assert schema.node_id == "UC_EmbeddingDetokenizerAnalysis"
    assert schema.display_name == "Embedding Detokenizer Analysis"
    assert [value.id for value in schema.inputs] == [
        "clip",
        "embedding_name",
        "top_k",
        "similarity_metric",
        "alignment_candidates",
        "latin_only",
    ]
    assert inputs["embedding_name"].options == ["sample.pt"]
    assert inputs["top_k"].default == 5
    assert inputs["similarity_metric"].default == "cosine"
    assert inputs["alignment_candidates"].default == 64
    assert inputs["latin_only"].default is False
    assert [value.display_name for value in schema.outputs] == [
        "analysis",
        "approximation",
        "cwb_approximation",
    ]


def test_node_resolves_selected_embedding_and_returns_analysis(monkeypatch):
    monkeypatch.setattr(
        embedding_nodes,
        "get_full_path_or_raise",
        lambda category, name: f"C:/resolved/{category}/{name}",
    )
    captured = {}

    def analyze(clip, path, top_k, metric, alignment_candidates, latin_only):
        captured.update(clip=clip, path=path, top_k=top_k, metric=metric, alignment_candidates=alignment_candidates, latin_only=latin_only)
        return "report", "tokens", "cwb tokens"

    monkeypatch.setattr(embedding_nodes, "analyze_embedding_file", analyze)
    clip = object()
    result = embedding_nodes.UC_EmbeddingDetokenizerAnalysis.execute(
        clip,
        "nested/sample.safetensors",
        7,
        "euclidean",
        80,
        True,
    )

    assert result.args == ("report", "tokens", "cwb tokens")
    assert captured == {
        "clip": clip,
        "path": "C:/resolved/embeddings/nested/sample.safetensors",
        "top_k": 7,
        "metric": "euclidean",
        "alignment_candidates": 80,
        "latin_only": True,
    }


def test_discovers_matching_tokenizer_and_model_branches():
    clip = FakeClip(VOCABULARY)
    tokenizers = embedding_helpers.discover_tokenizer_branches(clip.tokenizer)
    models = embedding_helpers.discover_model_branches(clip.cond_stage_model)

    assert [(branch.label, branch.embedding_key, branch.embedding_size) for branch in tokenizers] == [
        ("clip_l", "clip_l", 3)
    ]
    assert [(branch.label, branch.vocabulary_size, branch.embedding_size) for branch in models] == [
        ("clip_l", 4, 3)
    ]


@pytest.mark.parametrize(
    ("metric", "expected_ids", "expected_values"),
    [
        ("cosine", [0, 1], [1.0, 0.196116]),
        ("dot_product", [0, 1], [1.0, 0.1]),
        ("euclidean", [0, 1], [0.0, 1.06**0.5]),
    ],
)
def test_nearest_tokens_support_all_metrics(metric, expected_ids, expected_values):
    result = embedding_helpers.nearest_vocabulary_tokens(
        torch.tensor([[1.0, 0.0, 0.0]]),
        _model_branch(),
        [0, 1, 2, 3],
        2,
        metric,
        chunk_size=2,
    )[0]

    assert [candidate.token_id for candidate in result] == expected_ids
    assert [candidate.value for candidate in result] == pytest.approx(expected_values)


def test_chunked_search_matches_single_chunk():
    queries = torch.tensor([[0.8, 0.2, 0.0], [0.0, 0.1, 0.9]])
    branch = _model_branch()
    chunked = embedding_helpers.nearest_vocabulary_tokens(
        queries, branch, [0, 1, 2, 3], 3, "cosine", chunk_size=1
    )
    single = embedding_helpers.nearest_vocabulary_tokens(
        queries, branch, [0, 1, 2, 3], 3, "cosine", chunk_size=16
    )

    assert chunked == single


def test_cwb_alignment_uses_global_non_position_locked_matches():
    candidate = embedding_helpers.TokenCandidate
    selected = embedding_helpers.cwb_greedy_token_alignment(
        [
            [candidate(0, 0.90, 1.0), candidate(1, 0.80, 1.0)],
            [candidate(0, 0.95, 1.0), candidate(2, 0.70, 1.0)],
        ],
        special_ids=set(),
    )

    assert [entry.token_id if entry else None for entry in selected] == [1, 0]


def test_latin_candidate_filter_excludes_other_scripts():
    assert embedding_helpers._latin_decoded_token(" costume")
    assert embedding_helpers._latin_decoded_token(",")
    assert not embedding_helpers._latin_decoded_token("本周")
    assert not embedding_helpers._latin_decoded_token("ظهور")


def test_latin_only_filters_regular_and_cwb_approximations(monkeypatch):
    tokenizer = GenericTokenizerLeaf(
        "clip_l",
        2,
        {0: "本周", 1: " english"},
    )
    model_root = torch.nn.Module()
    model_root.clip_l = FakeModelLeaf([[1.0, 0.0], [0.9, 0.1]])
    clip = types.SimpleNamespace(
        tokenizer=types.SimpleNamespace(clip_l=tokenizer),
        cond_stage_model=model_root,
    )
    monkeypatch.setattr(
        embedding_helpers,
        "load_embed",
        lambda *args: torch.tensor([[1.0, 0.0]]),
    )

    analysis, approximation, cwb_approximation = embedding_helpers.analyze_embedding_file(
        clip, "C:/embeddings/sample.pt", 2, "cosine", latin_only=True
    )

    assert "Tokenizer IDs analyzed: 1" in analysis
    assert approximation == "[clip_l -> clip_l]  english"
    assert cwb_approximation == "[clip_l -> clip_l]  english"


def test_analysis_reports_every_row_special_tokens_and_approximation(monkeypatch):
    monkeypatch.setattr(
        embedding_helpers,
        "load_embed",
        lambda name, directories, size, key: torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        ),
    )
    analysis, approximation, cwb_approximation = embedding_helpers.analyze_embedding_file(
        FakeClip(VOCABULARY),
        "C:/embeddings/sample.pt",
        2,
        "cosine",
    )

    assert "Tokenizer branch: clip_l" in analysis
    assert "Model branch: clip_l" in analysis
    assert "Row 0:" in analysis
    assert "Row 1:" in analysis
    assert "id=2" in analysis
    assert "special=yes" in analysis
    assert "Top-1 IDs: [0, 2]" in analysis
    assert approximation == "[clip_l -> clip_l] zero special"
    assert "CWB matches: 2/2" in analysis
    assert cwb_approximation == "[clip_l -> clip_l] zero one"


def test_analysis_handles_multiple_non_clip_l_encoder_branches(monkeypatch):
    tokenizer_root = types.SimpleNamespace(
        clip_g=GenericTokenizerLeaf("clip_g", 2, {0: "g0", 1: "g1"}),
        t5xxl=GenericTokenizerLeaf("t5xxl", 4, {0: "t0", 1: "t1"}),
    )
    model_root = torch.nn.Module()
    model_root.clip_g = FakeModelLeaf([[1.0, 0.0], [0.0, 1.0]])
    model_root.t5xxl = FakeModelLeaf(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
    )
    clip = types.SimpleNamespace(
        tokenizer=tokenizer_root,
        cond_stage_model=model_root,
    )

    def load(name, directories, size, key):
        return {
            "clip_g": torch.tensor([[1.0, 0.0]]),
            "t5xxl": torch.tensor([[0.0, 1.0, 0.0, 0.0]]),
        }[key]

    monkeypatch.setattr(embedding_helpers, "load_embed", load)
    analysis, approximation, cwb_approximation = embedding_helpers.analyze_embedding_file(
        clip,
        "C:/embeddings/multi.safetensors",
        1,
        "cosine",
    )

    assert "Tokenizer branch: clip_g" in analysis
    assert "Tokenizer branch: t5xxl" in analysis
    assert "[clip_g -> clip_g] g0" in approximation
    assert "[t5xxl -> t5xxl] t1" in approximation
    assert "[clip_g -> clip_g] g0" in cwb_approximation
    assert "[t5xxl -> t5xxl] t1" in cwb_approximation


def test_incompatible_embedding_returns_diagnostic_report(monkeypatch):
    monkeypatch.setattr(
        embedding_helpers,
        "load_embed",
        lambda name, directories, size, key: torch.zeros((2, 4)),
    )
    analysis, approximation, cwb_approximation = embedding_helpers.analyze_embedding_file(
        FakeClip(VOCABULARY),
        "C:/embeddings/incompatible.pt",
        5,
        "cosine",
    )

    assert "incompatible tensor shape (2, 4)" in analysis
    assert "No compatible tokenizer/model branch was analyzed." in analysis
    assert approximation == ""
    assert cwb_approximation == ""


def test_zero_query_vector_remains_finite():
    result = embedding_helpers.nearest_vocabulary_tokens(
        torch.zeros((1, 3)),
        _model_branch(),
        [0, 1, 2, 3],
        4,
        "cosine",
        chunk_size=2,
    )[0]

    assert all(torch.isfinite(torch.tensor(candidate.value)) for candidate in result)
