from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pytest

from dual_memory import Embedder, HashingEmbedder, default_embedder


def test_hashing_is_deterministic() -> None:
    a = HashingEmbedder(dim=64).embed(["I live in Oslo"])
    b = HashingEmbedder(dim=64).embed(["I live in Oslo"])
    assert np.array_equal(a, b)


def test_hashing_vectors_are_unit_length() -> None:
    m = HashingEmbedder().embed(["one", "two words here", "Lisbon is lovely"])
    assert np.allclose(np.linalg.norm(m, axis=1), 1.0, atol=1e-5)


def test_hashing_empty_text_gives_zero_vector() -> None:
    m = HashingEmbedder(dim=32).embed([""])
    assert m.shape == (1, 32) and not m.any()


def test_hashing_similar_texts_score_higher() -> None:
    e = HashingEmbedder()
    q, near, far = e.embed(["where does Mira live", "Mira lives in Lisbon", "octopus documentary"])
    assert q @ near > q @ far


def test_hashing_rejects_tiny_dim() -> None:
    with pytest.raises(ValueError):
        HashingEmbedder(dim=4)


def test_hashing_name_encodes_settings() -> None:
    assert HashingEmbedder(dim=64).name != HashingEmbedder(dim=128).name
    assert HashingEmbedder(char_ngrams=False).name != HashingEmbedder().name


def test_hashing_satisfies_protocol() -> None:
    assert isinstance(HashingEmbedder(), Embedder)


def test_default_embedder_hashing_choice() -> None:
    assert isinstance(default_embedder("hashing"), HashingEmbedder)


def test_default_embedder_rejects_unknown() -> None:
    with pytest.raises(ValueError):
        default_embedder("word2vec")


def test_default_embedder_falls_back_with_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import dual_memory.embedders as mod

    def boom(*a: object, **k: object) -> None:
        raise ImportError("no onnxruntime here")

    monkeypatch.setattr(mod, "OnnxEmbedder", boom)
    with caplog.at_level(logging.WARNING):
        emb = mod.default_embedder("onnx")
    assert isinstance(emb, HashingEmbedder)
    assert "falling back" in caplog.text.lower() or "using hashingembedder" in caplog.text.lower()


def _local_model_dir() -> Path | None:
    env = os.environ.get("DUAL_MEMORY_MODEL_DIR")
    if env:
        return Path(env)
    snaps = Path.home() / ".cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots"
    found = sorted(snaps.glob("*/onnx/model.onnx")) if snaps.exists() else []
    return found[0].parent.parent if found else None


@pytest.mark.skipif(_local_model_dir() is None, reason="MiniLM ONNX model not downloaded (not needed in CI)")
def test_onnx_embedder_semantics() -> None:
    pytest.importorskip("onnxruntime")
    from dual_memory import OnnxEmbedder

    e = OnnxEmbedder(model_dir=_local_model_dir())
    assert e.dim == 384
    q, near, far = e.embed(["What city is she living in?", "She moved to Lisbon.", "I like chess."])
    assert q @ near > q @ far
    assert np.allclose(np.linalg.norm(e.embed(["x", "y z"]), axis=1), 1.0, atol=1e-4)


def test_onnx_embedder_missing_dir(tmp_path: Path) -> None:
    pytest.importorskip("onnxruntime")
    pytest.importorskip("tokenizers")
    from dual_memory import OnnxEmbedder

    with pytest.raises(FileNotFoundError):
        OnnxEmbedder(model_dir=tmp_path)
