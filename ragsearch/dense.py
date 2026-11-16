"""Dense retrieval: bi-encoder embeddings in an exact inner-product index (FAISS or numpy).

Embeddings are L2-normalised, so inner product equals cosine similarity.
"""
from __future__ import annotations

import os
import pickle
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Query-side instructions recommended by the model authors (passages are encoded as-is,
# except for the E5 family which prefixes both sides).
QUERY_PREFIX = {
    "bge-": "Represent this sentence for searching relevant passages: ",
    "e5-": "query: ",
}
PASSAGE_PREFIX = {"e5-": "passage: "}


def model_key(name: str) -> str:
    """File-system friendly key for a model name."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name.split("/")[-1])


def pick_device(pref: str = "auto") -> str:
    if pref and pref != "auto":
        return pref
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


class Encoder:
    name: str = "encoder"
    dim: int = 0

    def encode_passages(self, texts: Sequence[str], show_progress: bool = False) -> np.ndarray:
        raise NotImplementedError

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError


class STEncoder(Encoder):
    """Sentence-BERT style bi-encoder via sentence-transformers."""

    def __init__(self, name: str, device: str = "auto", batch_size: int = 64, max_seq_length: Optional[int] = 256):
        from sentence_transformers import SentenceTransformer

        self.name = name
        self.device = pick_device(device)
        self.batch_size = batch_size
        self.model = SentenceTransformer(name, device=self.device)
        if max_seq_length:
            self.model.max_seq_length = min(int(max_seq_length), int(self.model.max_seq_length or max_seq_length))
        get_dim = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        self.dim = int(get_dim())
        low = name.lower()
        self.query_prefix = next((p for k, p in QUERY_PREFIX.items() if k in low), "")
        self.passage_prefix = next((p for k, p in PASSAGE_PREFIX.items() if k in low), "")

    def _encode(self, texts: Sequence[str], show_progress: bool) -> np.ndarray:
        emb = self.model.encode(
            list(texts), batch_size=self.batch_size, show_progress_bar=show_progress,
            convert_to_numpy=True, normalize_embeddings=True,
        )
        return np.ascontiguousarray(emb, dtype=np.float32)

    def encode_passages(self, texts: Sequence[str], show_progress: bool = False) -> np.ndarray:
        if self.passage_prefix:
            texts = [self.passage_prefix + t for t in texts]
        return self._encode(texts, show_progress)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        if self.query_prefix:
            texts = [self.query_prefix + t for t in texts]
        return self._encode(texts, False)


class LSAEncoder(Encoder):
    """TF-IDF + truncated SVD (latent semantic analysis).

    A classical dense baseline that needs no model download. Used for tests, as an
    ablation row, and as a last-resort fallback when no neural encoder is available.
    """

    def __init__(self, dim: int = 256, seed: int = 0):
        self.name = "lsa-%d" % dim
        self.dim = dim
        self.seed = seed
        self.vectorizer = None
        self.svd = None

    def fit(self, texts: Sequence[str]) -> "LSAEncoder":
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=2 if len(texts) > 50 else 1,
                                          stop_words="english", max_features=200000)
        mat = self.vectorizer.fit_transform(texts)
        n_comp = max(2, min(self.dim, mat.shape[1] - 1, mat.shape[0] - 1))
        self.svd = TruncatedSVD(n_components=n_comp, random_state=self.seed)
        self.svd.fit(mat)
        self.dim = n_comp
        return self

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        if self.vectorizer is None:
            raise RuntimeError("LSAEncoder must be fitted (or loaded) before encoding")
        emb = self.svd.transform(self.vectorizer.transform(list(texts))).astype(np.float32)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        return np.ascontiguousarray(emb / np.maximum(norms, 1e-9))

    def encode_passages(self, texts: Sequence[str], show_progress: bool = False) -> np.ndarray:
        return self._encode(texts)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts)

    def save(self, path: str) -> None:
        with open(path, "wb") as fh:
            pickle.dump({"dim": self.dim, "seed": self.seed, "vectorizer": self.vectorizer, "svd": self.svd}, fh)

    @classmethod
    def load(cls, path: str) -> "LSAEncoder":
        with open(path, "rb") as fh:
            state = pickle.load(fh)
        enc = cls(dim=state["dim"], seed=state["seed"])
        enc.vectorizer, enc.svd = state["vectorizer"], state["svd"]
        enc.name = "lsa-%d" % enc.dim
        return enc


_ENCODER_CACHE: Dict[Tuple[str, str], Encoder] = {}


def get_encoder(name: str, device: str = "auto", batch_size: int = 64, max_seq_length: Optional[int] = 256) -> Encoder:
    key = (name, device)
    if key not in _ENCODER_CACHE:
        _ENCODER_CACHE[key] = STEncoder(name, device=device, batch_size=batch_size, max_seq_length=max_seq_length)
    return _ENCODER_CACHE[key]


def _resolve_backend(backend: str) -> str:
    backend = os.environ.get("RAGSEARCH_DENSE_BACKEND", backend or "auto")
    if backend == "numpy":
        return "numpy"
    try:
        import faiss  # noqa: F401

        return "faiss"
    except Exception:
        if backend == "faiss":
            raise
        return "numpy"


class DenseIndex:
    """Exact maximum-inner-product search over normalised embeddings."""

    def __init__(self, embeddings: np.ndarray, backend: str = "auto"):
        self.emb = np.ascontiguousarray(embeddings, dtype=np.float32)
        self.backend = _resolve_backend(backend)
        self._faiss = None
        if self.backend == "faiss" and self.emb.shape[0] > 0:
            import faiss

            self._faiss = faiss.IndexFlatIP(self.emb.shape[1])
            self._faiss.add(self.emb)

    @property
    def size(self) -> int:
        return int(self.emb.shape[0])

    def search(self, queries: np.ndarray, top_n: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Return (ids, scores), each of shape (n_queries, min(top_n, size))."""
        q = np.ascontiguousarray(np.atleast_2d(queries), dtype=np.float32)
        k = min(top_n, self.size)
        if k == 0:
            return np.zeros((q.shape[0], 0), dtype=np.int64), np.zeros((q.shape[0], 0), dtype=np.float32)
        if self._faiss is not None:
            scores, ids = self._faiss.search(q, k)
            return ids.astype(np.int64), scores.astype(np.float32)
        all_ids, all_scores = [], []
        for start in range(0, q.shape[0], 32):
            block = q[start: start + 32] @ self.emb.T
            if k < self.size:
                part = np.argpartition(-block, k - 1, axis=1)[:, :k]
            else:
                part = np.tile(np.arange(self.size), (block.shape[0], 1))
            part_scores = np.take_along_axis(block, part, axis=1)
            order = np.lexsort((part, -part_scores), axis=1)
            all_ids.append(np.take_along_axis(part, order, axis=1))
            all_scores.append(np.take_along_axis(part_scores, order, axis=1))
        return np.vstack(all_ids).astype(np.int64), np.vstack(all_scores).astype(np.float32)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.save(path, self.emb)

    @classmethod
    def load(cls, path: str, backend: str = "auto") -> "DenseIndex":
        return cls(np.load(path), backend=backend)


def encode_in_batches(encoder: Encoder, texts: List[str], chunk: int = 20000, show_progress: bool = True,
                      checkpoint: Optional[str] = None) -> np.ndarray:
    """Encode a large list in chunks, checkpointing to disk so an interrupted run resumes."""
    done: List[np.ndarray] = []
    start = 0
    ckpt = Path(checkpoint) if checkpoint else None
    if ckpt is not None and ckpt.exists():
        try:
            saved = np.load(ckpt)
            if saved.ndim == 2 and saved.shape[0] <= len(texts):
                done.append(saved)
                start = saved.shape[0]
        except Exception:
            start = 0
    while start < len(texts):
        part = encoder.encode_passages(texts[start: start + chunk], show_progress=show_progress)
        done.append(part)
        start += len(part)
        if ckpt is not None and start < len(texts):
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            with open(ckpt, "wb") as fh:
                np.save(fh, np.vstack(done))
            print("  encoded %d / %d passages" % (start, len(texts)), flush=True)
    if ckpt is not None and ckpt.exists():
        ckpt.unlink()
    if not done:
        return np.zeros((0, encoder.dim), dtype=np.float32)
    return np.vstack(done)
