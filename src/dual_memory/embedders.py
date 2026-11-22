"""Embedders turn text into unit-length vectors.

Two are built in:

- ``HashingEmbedder``: no model, no download, deterministic. Words, word pairs
  and character trigrams are hashed into a fixed-size vector. Good for tests
  and as a fallback; it matches shared words, not meaning.
- ``OnnxEmbedder``: a small sentence-embedding model (by default
  ``sentence-transformers/all-MiniLM-L6-v2``, Apache-2.0) run on the CPU with
  onnxruntime. Needs ``pip install "dual-memory[onnx]"``; the model is fetched
  from the Hugging Face Hub on first use and cached.

Anything with a ``name``, a ``dim`` and an ``embed(texts)`` method works.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from .text import words

log = logging.getLogger(__name__)

Vectors = npt.NDArray[np.float32]

DEFAULT_ONNX_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@runtime_checkable
class Embedder(Protocol):
    """The embedder interface: a stable name, a dimension and a batch embed."""

    @property
    def name(self) -> str: ...

    @property
    def dim(self) -> int: ...

    def embed(self, texts: list[str]) -> Vectors: ...


def _l2_normalize(m: Vectors) -> Vectors:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    out: Vectors = (m / norms).astype(np.float32)
    return out


class HashingEmbedder:
    """Deterministic feature-hashing embedder (no model needed)."""

    def __init__(self, dim: int = 384, char_ngrams: bool = True) -> None:
        if dim < 16:
            raise ValueError("dim must be at least 16")
        self._dim = dim
        self._char_ngrams = char_ngrams

    @property
    def name(self) -> str:
        return f"hashing-{self._dim}{'-c3' if self._char_ngrams else ''}"

    @property
    def dim(self) -> int:
        return self._dim

    def _features(self, text: str) -> list[tuple[str, float]]:
        ws = words(text)
        feats: list[tuple[str, float]] = [("w:" + w, 1.0) for w in ws]
        feats += [("b:" + a + "_" + b, 0.7) for a, b in zip(ws, ws[1:], strict=False)]
        if self._char_ngrams:
            for w in ws:
                padded = f"#{w}#"
                feats += [("c:" + padded[i : i + 3], 0.3) for i in range(len(padded) - 2)]
        return feats

    def embed(self, texts: list[str]) -> Vectors:
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feat, weight in self._features(text):
                h = int.from_bytes(hashlib.blake2b(feat.encode(), digest_size=8).digest(), "little")
                sign = 1.0 if (h >> 63) & 1 else -1.0
                out[row, h % self._dim] += sign * weight
        return _l2_normalize(out)


class OnnxEmbedder:
    """Sentence embeddings from an ONNX transformer, mean-pooled and normalised."""

    def __init__(
        self,
        model: str = DEFAULT_ONNX_MODEL,
        model_dir: str | Path | None = None,
        max_length: int = 256,
        batch_size: int = 64,
        threads: int = 2,
    ) -> None:
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError as exc:  # pragma: no cover - depends on optional deps
            raise ImportError('OnnxEmbedder needs the onnx extra: pip install "dual-memory[onnx]"') from exc
        onnx_path, tok_path = self._locate(model, model_dir)
        self._model = model
        self._batch = batch_size
        self._tok = Tokenizer.from_file(str(tok_path))
        self._tok.enable_truncation(max_length=max_length)
        self._tok.enable_padding()
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        self._sess = ort.InferenceSession(
            str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._inputs = {i.name for i in self._sess.get_inputs()}
        self._dim = int(self.embed(["probe"]).shape[1])

    @staticmethod
    def _locate(model: str, model_dir: str | Path | None) -> tuple[Path, Path]:
        if model_dir is not None:
            base = Path(model_dir)
            onnx = base / "onnx" / "model.onnx"
            if not onnx.exists():
                onnx = base / "model.onnx"
            tok = base / "tokenizer.json"
            if not onnx.exists() or not tok.exists():
                raise FileNotFoundError(f"model.onnx and tokenizer.json not found in {base}")
            return onnx, tok
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:  # pragma: no cover
            raise ImportError('pip install "dual-memory[onnx]" to download models') from exc
        return (
            Path(hf_hub_download(model, "onnx/model.onnx")),
            Path(hf_hub_download(model, "tokenizer.json")),
        )

    @property
    def name(self) -> str:
        return f"onnx:{self._model}"

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> Vectors:
        if not texts:
            return np.zeros((0, getattr(self, "_dim", 0)), dtype=np.float32)
        chunks = [self._embed_batch(texts[i : i + self._batch]) for i in range(0, len(texts), self._batch)]
        return np.vstack(chunks)

    def _embed_batch(self, texts: list[str]) -> Vectors:
        enc = self._tok.encode_batch(texts)
        ids = np.array([e.ids for e in enc], dtype=np.int64)
        mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
        feeds: dict[str, Any] = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self._inputs:
            feeds["token_type_ids"] = np.zeros_like(ids)
        hidden = self._sess.run(None, feeds)[0]
        m = mask[..., None].astype(np.float32)
        pooled = (hidden * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-9, None)
        return _l2_normalize(pooled.astype(np.float32))


def default_embedder(prefer: str = "onnx") -> Embedder:
    """Return the ONNX model if it can be loaded, otherwise the hashing embedder.

    The fallback is logged as a warning so it never happens silently.
    """
    if prefer == "hashing":
        return HashingEmbedder()
    if prefer != "onnx":
        raise ValueError("prefer must be 'onnx' or 'hashing'")
    try:
        return OnnxEmbedder()
    except Exception as exc:  # noqa: BLE001 - any load failure means fall back
        log.warning("ONNX embedder unavailable (%s); using HashingEmbedder", exc)
        return HashingEmbedder()
