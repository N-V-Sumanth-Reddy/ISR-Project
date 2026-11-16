"""BM25 over a sparse term-frequency matrix.

k1 and b are applied at query time, so the dev-split grid search needs no re-indexing.
Scoring follows Lucene / Robertson & Zaragoza (2009):

    idf(t)     = ln(1 + (N - df + 0.5) / (df + 0.5))
    score(q,d) = sum_t idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * |d| / avgdl))
"""
from __future__ import annotations

import json
from array import array
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from .textproc import Tokenizer


class BM25Index:
    def __init__(self, k1: float = 0.9, b: float = 0.4, stem: bool = True, stopwords: bool = True):
        self.k1 = float(k1)
        self.b = float(b)
        self.tokenizer = Tokenizer(stem=stem, stopwords=stopwords)
        self.vocab: Dict[str, int] = {}
        self.indptr = np.zeros(1, dtype=np.int64)      # CSC layout: one posting list per term
        self.indices = np.zeros(0, dtype=np.int32)     # passage ids
        self.data = np.zeros(0, dtype=np.float32)      # term frequencies
        self.doc_len = np.zeros(0, dtype=np.float32)
        self.n_docs = 0
        self.avgdl = 0.0
        self.idf = np.zeros(0, dtype=np.float32)

    # ------------------------------------------------------------------ build
    def build(self, texts: Iterable[str]) -> "BM25Index":
        rows, cols, vals = array("i"), array("i"), array("i")
        doc_len: List[int] = []
        vocab: Dict[str, int] = {}
        for doc_idx, text in enumerate(texts):
            tokens = self.tokenizer(text)
            doc_len.append(len(tokens))
            for term, tf in Counter(tokens).items():
                tid = vocab.get(term)
                if tid is None:
                    tid = len(vocab)
                    vocab[term] = tid
                rows.append(doc_idx)
                cols.append(tid)
                vals.append(tf)
        self.vocab = vocab
        self.n_docs = len(doc_len)
        self.doc_len = np.asarray(doc_len, dtype=np.float32)
        self.avgdl = float(self.doc_len.mean()) if self.n_docs else 0.0

        rows_np = np.frombuffer(rows, dtype=np.int32) if len(rows) else np.zeros(0, np.int32)
        cols_np = np.frombuffer(cols, dtype=np.int32) if len(cols) else np.zeros(0, np.int32)
        vals_np = np.frombuffer(vals, dtype=np.int32) if len(vals) else np.zeros(0, np.int32)
        order = np.argsort(cols_np, kind="stable")     # stable keeps passage ids ascending per term
        self.indices = rows_np[order].astype(np.int32)
        self.data = vals_np[order].astype(np.float32)
        counts = np.bincount(cols_np, minlength=len(vocab))
        self.indptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        self._finalise()
        return self

    def _finalise(self) -> None:
        df = np.diff(self.indptr).astype(np.float64)
        self.idf = np.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)

    # ----------------------------------------------------------------- search
    def score(self, query: str, k1: Optional[float] = None, b: Optional[float] = None) -> np.ndarray:
        k1 = self.k1 if k1 is None else float(k1)
        b = self.b if b is None else float(b)
        scores = np.zeros(self.n_docs, dtype=np.float32)
        if self.n_docs == 0:
            return scores
        norm = k1 * (1.0 - b + b * self.doc_len / max(self.avgdl, 1e-9))
        for term, qtf in Counter(self.tokenizer(query)).items():
            tid = self.vocab.get(term)
            if tid is None:
                continue
            lo, hi = self.indptr[tid], self.indptr[tid + 1]
            docs = self.indices[lo:hi]
            tf = self.data[lo:hi]
            scores[docs] += qtf * self.idf[tid] * tf * (k1 + 1.0) / (tf + norm[docs])
        return scores

    def search(self, query: str, top_n: int = 100, k1: Optional[float] = None,
               b: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Return (passage_ids, scores) sorted by score desc, ties broken by id."""
        scores = self.score(query, k1, b)
        return top_n_sorted(scores, top_n, positive_only=True)

    # ------------------------------------------------------------------- disk
    def save(self, directory: str) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        np.savez(d / "bm25.npz", indptr=self.indptr, indices=self.indices, data=self.data, doc_len=self.doc_len)
        meta = {"k1": self.k1, "b": self.b, "n_docs": self.n_docs, "avgdl": self.avgdl,
                "tokenizer": self.tokenizer.signature()}
        (d / "bm25_meta.json").write_text(json.dumps(meta))
        terms = [None] * len(self.vocab)
        for term, tid in self.vocab.items():
            terms[tid] = term
        (d / "bm25_vocab.json").write_text(json.dumps(terms))

    def save_params(self, directory: str) -> None:
        """Persist only k1 and b (after tuning on the dev split)."""
        path = Path(directory) / "bm25_meta.json"
        meta = json.loads(path.read_text())
        meta["k1"], meta["b"] = self.k1, self.b
        path.write_text(json.dumps(meta))

    @classmethod
    def load(cls, directory: str) -> "BM25Index":
        d = Path(directory)
        meta = json.loads((d / "bm25_meta.json").read_text())
        idx = cls(k1=meta["k1"], b=meta["b"], stem=meta["tokenizer"]["stem"], stopwords=meta["tokenizer"]["stopwords"])
        arrays = np.load(d / "bm25.npz")
        idx.indptr, idx.indices, idx.data, idx.doc_len = (arrays[k] for k in ("indptr", "indices", "data", "doc_len"))
        idx.n_docs, idx.avgdl = int(meta["n_docs"]), float(meta["avgdl"])
        idx.vocab = {term: i for i, term in enumerate(json.loads((d / "bm25_vocab.json").read_text()))}
        idx._finalise()
        return idx


def top_n_sorted(scores: np.ndarray, top_n: int, positive_only: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Indices and values of the top_n scores, sorted by (-score, index)."""
    n = scores.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)
    k = min(top_n, n)
    if k < n:
        cand = np.argpartition(-scores, k - 1)[:k]
        # argpartition picks arbitrarily among ties at the cut-off; pull in every tied
        # element so the final order is deterministic.
        cutoff = scores[cand].min()
        tied = np.flatnonzero(scores == cutoff)
        if tied.size > 1 and tied.size <= 10 * k:
            cand = np.union1d(cand, tied)
    else:
        cand = np.arange(n)
    order = np.lexsort((cand, -scores[cand]))
    cand = cand[order][:k]
    vals = scores[cand]
    if positive_only:
        keep = vals > 0
        cand, vals = cand[keep], vals[keep]
    return cand.astype(np.int64), vals.astype(np.float32)
