"""Whisper regressions for batch boundaries, window seeking, and cache cleanup."""
import json
import builtins
import importlib.util
from pathlib import Path
import sys
import types

import pytest
import torch


ROOT = Path(__file__).parents[1]
PACKAGE = "utils_collection_whisper_test"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT)]
sys.modules.setdefault(PACKAGE, package)
from utils_collection_whisper_test.helpers import model_helpers as helpers
from utils_collection_whisper_test.models.whisper import ModelDimensions, Whisper
from utils_collection_whisper_test.helpers import whisper_timing_helpers as timing
from utils_collection_whisper_test.models.whisper import MultiHeadAttention


def test_alignment_scores_do_not_change_native_attention_output():
    attention = MultiHeadAttention(8, 2, torch.nn)
    x = torch.arange(24, dtype=torch.float32).reshape(1, 3, 8) / 24
    audio = torch.arange(40, dtype=torch.float32).reshape(1, 5, 8) / 40
    expected = attention(x, audio)
    scores = []
    actual = attention(x, audio, alignment=([1], scores))
    torch.testing.assert_close(actual, expected)
    q = attention.query(x).reshape(1, 3, 2, 4).transpose(1, 2)[:, 1:2]
    k = attention.key(audio).reshape(1, 5, 2, 4).transpose(1, 2)[:, 1:2]
    torch.testing.assert_close(scores[0], ((q * 4 ** -0.25) @ (k * 4 ** -0.25).transpose(-1, -2))[0])
    assert not scores[0].requires_grad
    assert not attention._forward_hooks


def test_large_v3_alignment_heads_and_unicode_word_splitting():
    heads = timing.alignment_heads("large-v3", types.SimpleNamespace(n_text_layer=32, n_text_head=20))
    assert heads and all(0 <= layer < 32 and all(0 <= head < 20 for head in row) for layer, row in heads.items())
    for language, text in [("en", " Hello, world!"), ("zh", "你好世界。")]:
        tokenizer = helpers.whisper_get_tokenizer(True, language=language, task="transcribe")
        tokens = tokenizer.encode(text)
        words, grouped = tokenizer.split_to_word_tokens(tokens)
        assert "".join(words) == text
        assert [token for group in grouped for token in group] == tokens
        assert len(words) > 1


def test_word_alignment_uses_attention_boundaries_not_even_sentence_splitting():
    tokenizer = helpers.whisper_get_tokenizer(True, language="en", task="transcribe")
    text_tokens = tokenizer.encode(" hello world")
    assert len(text_tokens) == 2

    class Model:
        device = torch.device("cpu")

        def __call__(self, mel, tokens, alignment_scores):
            n = tokens.shape[1]
            offset = len(tokenizer.sot_sequence)
            scores = torch.full((1, n, 10), -10.0)
            scores[0, offset, :2] = 10
            scores[0, offset + 1, 2:8] = 10
            scores[0, offset + 2, 8:] = 10
            alignment_scores.append(scores)
            return torch.zeros(1, n, tokenizer.encoding.n_vocab)

    words = timing.find_alignment(Model(), tokenizer, text_tokens, torch.zeros(80, 20), 20, medfilt_width=1)
    assert [word.word for word in words] == [" hello", " world"]
    assert [(word.start, word.end) for word in words] == [(0.0, 0.04), (0.04, 0.16)]


def test_word_timestamps_keep_window_offsets_and_clip_padding(monkeypatch):
    tokenizer = helpers.whisper_get_tokenizer(True, language="en", task="transcribe")
    tokens = tokenizer.encode(" hello world")
    calls = []

    def decode(*args):
        seconds = 30 if not calls else 5
        return types.SimpleNamespace(tokens=[tokenizer.timestamp_begin, *tokens, tokenizer.timestamp_begin + seconds * 50],
                                     no_speech_prob=0, avg_logprob=0, temperature=0)

    def align(*, segments, time_offset, num_frames, **kwargs):
        calls.append((time_offset, num_frames))
        segments[0]["words"] = [
            {"word": " hello", "start": time_offset, "end": time_offset + 0.4, "probability": 0.9},
            {"word": " world", "start": time_offset + 0.4, "end": time_offset + num_frames / 100 + 0.5, "probability": 0.9},
        ]

    monkeypatch.setattr(helpers, "whisper_decode_with_fallback", decode)
    monkeypatch.setattr(helpers, "add_word_timestamps", align)
    model = types.SimpleNamespace(dims=types.SimpleNamespace(n_mels=80, n_audio_ctx=1500), device=torch.device("cpu"),
                                  compute_dtype=torch.float32, is_multilingual=True, num_languages=99)
    result = helpers.transcribe_whisper(model, torch.zeros(35 * 16000), "transcribe", "en", word_timestamps=True)
    # A completed timestamp boundary advances by the full 30-second window.
    assert calls == [(0, 3000), (30, 500)]
    assert result["segments"][-1]["words"][-1]["end"] == 35


def test_word_timestamps_merge_punctuation_and_emit_serializable_records(monkeypatch):
    aligned = [timing.WordTiming(" Hello", [1], 0.2, 0.5, 0.9),
               timing.WordTiming(",", [2], 0.5, 0.5, 0.9),
               timing.WordTiming(" world", [3], 0.5, 0.9, 0.8),
               timing.WordTiming("!", [4], 0.9, 0.9, 0.8)]
    monkeypatch.setattr(timing, "find_alignment", lambda *args, **kwargs: aligned)
    segments = [{"start": 30.2, "end": 30.9, "text": " Hello, world!", "tokens": [1, 2, 3, 4]}]
    timing.add_word_timestamps(segments=segments, model=None, tokenizer=types.SimpleNamespace(eot=100),
                               mel=None, num_frames=100, last_speech_timestamp=30, time_offset=30)
    words = segments[0]["words"]
    assert [word["word"] for word in words] == [" Hello,", " world!"]
    assert [(word["start"], word["end"]) for word in words] == [(30.2, 30.5), (30.5, 30.9)]
    assert json.loads(json.dumps(segments))[0]["words"] == words


def test_missing_tiktoken_does_not_prevent_model_helpers_import(monkeypatch):
    original_import = builtins.__import__

    def without_tiktoken(name, *args, **kwargs):
        if name == "tiktoken":
            raise ModuleNotFoundError("No module named 'tiktoken'", name="tiktoken")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_tiktoken)
    module_name = f"{PACKAGE}.helpers.model_helpers_without_tiktoken"
    spec = importlib.util.spec_from_file_location(module_name, ROOT / "helpers" / "model_helpers.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    assert callable(module.load_openpose_model)
    with pytest.raises(RuntimeError, match="Install tiktoken"):
        module.whisper_get_encoding()
    # Must fail before reading audio or loading the model onto the device.
    with pytest.raises(RuntimeError, match="Install tiktoken"):
        module.run_whisper(None, None, "transcribe", "auto")


def test_stereo_resampling_and_batch_output_alignment(monkeypatch):
    samples = torch.arange(8000, dtype=torch.float32) / 8000
    waveform = torch.stack([torch.stack([samples, -samples]), torch.stack([samples, samples])])
    audio = {"waveform": waveform, "sample_rate": 8000}
    prepared = helpers.prepare_whisper_audio(audio)
    assert prepared.shape == (2, 16000)
    assert torch.count_nonzero(prepared[0]) == 0
    assert prepared[1, 1000:15000].mean() > 0.4
    seen = []
    def transcribe(model, waveform, task, language):
        seen.append((waveform.clone(), task, language))
        return {"text": str(len(seen)), "segments": [{"start": 0, "end": 1, "text": str(len(seen))}], "language": "sv"}
    monkeypatch.setattr(helpers, "transcribe_whisper", transcribe)
    monkeypatch.setattr(helpers.comfy.model_management, "load_models_gpu", lambda models: None)
    output = helpers.run_whisper(types.SimpleNamespace(model=types.SimpleNamespace(num_languages=99)), audio, "translate", "sv")
    assert output[0] == ["1", "2"]
    assert [json.loads(value)[0]["text"] for value in output[1]] == output[0]
    assert output[2] == ["sv", "sv"]
    assert [(task, language) for _, task, language in seen] == [("translate", "sv"), ("translate", "sv")]
    torch.testing.assert_close(seen[0][0], prepared[0])
    torch.testing.assert_close(seen[1][0], prepared[1])


def test_multiwindow_timestamp_offsets_and_prompt_continuity(monkeypatch):
    tokenizer = helpers.whisper_get_tokenizer(True, language="en")
    text_tokens = tokenizer.encode(" hello")
    prompts = []
    def decode(model, mel, language, task, prompt):
        prompts.append(list(prompt))
        seconds = 30 if len(prompts) == 1 else 5
        return types.SimpleNamespace(tokens=[tokenizer.timestamp_begin, *text_tokens, tokenizer.timestamp_begin + seconds * 50], no_speech_prob=0, avg_logprob=0, temperature=0)
    monkeypatch.setattr(helpers, "whisper_decode_with_fallback", decode)
    model = types.SimpleNamespace(dims=types.SimpleNamespace(n_mels=80, n_audio_ctx=1500), device=torch.device("cpu"), compute_dtype=torch.float32, is_multilingual=True, num_languages=99)
    result = helpers.transcribe_whisper(model, torch.zeros(35 * 16000), "transcribe", "en")
    assert [(segment["start"], segment["end"]) for segment in result["segments"]] == [(0, 30), (30, 35)]
    assert result["text"] == " hello hello"
    assert prompts[0] == [] and text_tokens[0] in prompts[1]


def test_interruption_removes_decoder_hooks(monkeypatch):
    dims = ModelDimensions(4, 6, 8, 2, 1, 51865, 8, 8, 2, 1)
    model = Whisper(dims)
    model.device = torch.device("cpu")
    task = helpers.WhisperDecodingTask(model, helpers.WhisperDecodingOptions(language="en"))
    def interrupted(tokens, features):
        task.inference.kv_cache, task.inference.hooks = model.install_kv_cache_hooks()
        raise RuntimeError("test interruption")
    monkeypatch.setattr(task.inference, "logits", interrupted)
    with pytest.raises(RuntimeError, match="test interruption"):
        task._main_loop(torch.zeros(1, 6, 8), torch.tensor([[50258, 50259, 50359]]))
    assert task.inference.hooks == [] and task.inference.kv_cache == {}
    assert all(not layer._forward_hooks for layer in model.decoder.modules())
