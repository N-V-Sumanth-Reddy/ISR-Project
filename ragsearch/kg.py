"""Lightweight knowledge-graph extension: a bipartite passage-entity graph.

"Entities" are salient terms (rare-enough words, hyphenated/alphanumeric compounds such
as IL-6 or SARS-CoV-2, and for small collections two-word phrases). Two passages are
linked when they share entities; two entities are linked when they co-occur in a passage.

The graph is used in three optional ways, each measured against the same baselines:
  1. expansion / re-ranking: activate the entities of the query and of the top fused
     passages, propagate one step back to passages, and add that ranking to RRF;
  2. answer checking: flag specific terms in an answer sentence that do not occur in
     the passages it cites;
  3. visualisation: the evidence graph of a result set (Graphviz DOT).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse

_ENT_TOKEN = r"(?u)\b[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)*\b"
_ENT_RE = re.compile(_ENT_TOKEN)


def _analyzer_tokens(text: str) -> List[str]:
    return [t.lower() for t in _ENT_RE.findall(text) if len(t) >= 3]


class EntityGraph:
    def __init__(self, max_entities_per_passage: int = 12, min_df: int = 2, max_df_ratio: float = 0.05):
        self.max_entities = int(max_entities_per_passage)
        self.min_df = int(min_df)
        self.max_df_ratio = float(max_df_ratio)
        self.entities: List[str] = []
        self.index: Dict[str, int] = {}
        self.idf = np.zeros(0, dtype=np.float32)
        self.A = sparse.csr_matrix((0, 0), dtype=np.float32)   # passages x entities
        self.AT = sparse.csr_matrix((0, 0), dtype=np.float32)

    # ------------------------------------------------------------------ build
    def build(self, texts: Sequence[str]) -> "EntityGraph":
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer

        n = len(texts)
        ngram = (1, 2) if n <= 30000 else (1, 1)      # phrases only where memory allows
        min_df = self.min_df if n >= 20 else 1
        max_df = max(self.max_df_ratio, (min_df + 1) / max(n, 1)) if n >= 20 else 1.0
        vec = CountVectorizer(lowercase=True, token_pattern=_ENT_TOKEN, ngram_range=ngram,
                              stop_words=list(ENGLISH_STOP_WORDS), min_df=min_df, max_df=max_df,
                              max_features=400000, dtype=np.float32)
        try:
            counts = vec.fit_transform(texts).tocsr()
        except ValueError:                             # empty vocabulary
            counts = sparse.csr_matrix((n, 0), dtype=np.float32)
            vec.vocabulary_ = {}
        vocab = sorted(vec.vocabulary_.items(), key=lambda kv: kv[1])
        keep = np.array([len(term.replace(" ", "")) >= 3 and not term.replace(" ", "").isdigit()
                         for term, _ in vocab], dtype=bool) if vocab else np.zeros(0, dtype=bool)
        if vocab:
            counts = counts[:, np.flatnonzero(keep)]
        self.entities = [term for (term, _), k in zip(vocab, keep) if k]
        self.index = {term: i for i, term in enumerate(self.entities)}

        df = np.asarray((counts > 0).sum(axis=0)).ravel().astype(np.float64)
        self.idf = (np.log((1.0 + n) / (1.0 + df)) + 1.0).astype(np.float32)
        # log-scale tf, then keep the top-M entities per passage
        counts = counts.tocsr()
        tfidf = sparse.csr_matrix((np.log1p(counts.data) * self.idf[counts.indices], counts.indices, counts.indptr),
                                  shape=counts.shape, dtype=np.float32)
        tfidf = _row_top_k(tfidf, self.max_entities)
        used = np.flatnonzero(np.diff(tfidf.tocsc().indptr) > 0)      # drop entities no passage kept
        tfidf = tfidf[:, used].tocsr()
        self.entities = [self.entities[i] for i in used]
        self.index = {term: i for i, term in enumerate(self.entities)}
        self.idf = self.idf[used]
        norms = np.sqrt(np.asarray(tfidf.multiply(tfidf).sum(axis=1)).ravel())
        norms[norms == 0] = 1.0
        self.A = sparse.diags(1.0 / norms).dot(tfidf).tocsr().astype(np.float32)
        self.AT = self.A.T.tocsr()
        return self

    # ---------------------------------------------------------------- queries
    def text_entities(self, text: str) -> List[int]:
        toks = _analyzer_tokens(text)
        found: List[int] = []
        for i, tok in enumerate(toks):
            for cand in ((tok + " " + toks[i + 1]) if i + 1 < len(toks) else None, tok):
                if cand is not None and cand in self.index and self.index[cand] not in found:
                    found.append(self.index[cand])
        return found

    def passage_entities(self, pid: int, top: int = 8) -> List[Tuple[str, float]]:
        row = self.A.getrow(pid)
        order = np.argsort(-row.data)[:top]
        return [(self.entities[row.indices[i]], float(row.data[i])) for i in order]

    def propagate(self, query: str, seed_pids: Sequence[int], seed_weights: Optional[Sequence[float]] = None,
                  query_weight: float = 0.5) -> np.ndarray:
        """Score every passage by a two-hop walk: (query + seed passages) -> entities -> passages."""
        n_pass, n_ent = self.A.shape
        if n_ent == 0:
            return np.zeros(n_pass, dtype=np.float32)
        act = np.zeros(n_ent, dtype=np.float32)
        q_ents = self.text_entities(query)
        if q_ents:
            q_vec = np.zeros(n_ent, dtype=np.float32)
            q_vec[q_ents] = self.idf[q_ents]
            act += query_weight * q_vec / max(float(np.linalg.norm(q_vec)), 1e-9)
        if len(seed_pids):
            w = np.ones(len(seed_pids), dtype=np.float32) if seed_weights is None else np.asarray(seed_weights, dtype=np.float32)
            seed = np.zeros(n_pass, dtype=np.float32)
            seed[np.asarray(seed_pids, dtype=np.int64)] = w / max(float(w.sum()), 1e-9)
            s_vec = self.AT.dot(seed)
            norm = float(np.linalg.norm(s_vec))
            if norm > 0:
                act += (1.0 - query_weight if q_ents else 1.0) * s_vec / norm
        return np.asarray(self.A.dot(act)).ravel().astype(np.float32)

    # ---------------------------------------------------------- answer checks
    def unsupported_terms(self, sentence: str, support_texts: Sequence[str], question: str = "",
                          min_idf_quantile: float = 0.5) -> List[str]:
        """Graph entities in ``sentence`` that occur in none of the supporting passages."""
        if not self.entities:
            return []
        support = set()
        for text in list(support_texts) + [question]:
            toks = _analyzer_tokens(text)
            support.update(toks)
            support.update(a + " " + b for a, b in zip(toks, toks[1:]))
        cutoff = float(np.quantile(self.idf, min_idf_quantile))
        out = []
        for eid in self.text_entities(sentence):
            term = self.entities[eid]
            if self.idf[eid] >= cutoff and term not in support and all(p not in support for p in term.split()):
                out.append(term)
        return out

    # ----------------------------------------------------------- visualisation
    def evidence_graph(self, pids: Sequence[int], labels: Sequence[str], query: str = "",
                       max_entities: int = 14) -> Dict[str, list]:
        """Nodes and edges linking the given passages through shared / query entities."""
        if not self.entities or not len(pids):
            return {"passages": list(labels), "entities": [], "edges": []}
        q_ents = set(self.text_entities(query))
        sub = self.A[np.asarray(pids, dtype=np.int64)]
        support = np.asarray((sub > 0).sum(axis=0)).ravel()
        weight = np.asarray(sub.sum(axis=0)).ravel()
        cand = [e for e in np.flatnonzero(support) if support[e] >= 2 or e in q_ents]
        cand.sort(key=lambda e: (-(e in q_ents), -support[e], -weight[e]))
        cand = cand[:max_entities]
        edges = []
        for row, _ in enumerate(pids):
            r = sub.getrow(row)
            present = dict(zip(r.indices.tolist(), r.data.tolist()))
            for e in cand:
                if e in present:
                    edges.append((row, int(e), float(present[e])))
        ents = [{"id": int(e), "label": self.entities[e], "in_query": bool(e in q_ents), "passages": int(support[e])}
                for e in cand]
        return {"passages": list(labels), "entities": ents, "edges": edges}

    def evidence_dot(self, pids: Sequence[int], labels: Sequence[str], query: str = "") -> str:
        g = self.evidence_graph(pids, labels, query)
        lines = ["graph evidence {", "  layout=neato; overlap=false; splines=true;",
                 '  node [fontname="Helvetica", fontsize=10];']
        for i, label in enumerate(g["passages"]):
            lines.append('  p%d [label="%s", shape=box, style="rounded,filled", fillcolor="#dbeafe", color="#1d4ed8"];'
                         % (i, label.replace('"', "'")))
        for ent in g["entities"]:
            fill = "#fde68a" if ent["in_query"] else "#f1f5f9"
            lines.append('  e%d [label="%s", shape=ellipse, style=filled, fillcolor="%s", color="#64748b"];'
                         % (ent["id"], ent["label"].replace('"', "'"), fill))
        for p, e, w in g["edges"]:
            lines.append("  p%d -- e%d [penwidth=%.1f, color=\"#94a3b8\"];" % (p, e, 0.6 + 2.0 * min(w, 1.0)))
        lines.append("}")
        return "\n".join(lines)

    # ------------------------------------------------------------------- disk
    def save(self, directory: str) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        sparse.save_npz(d / "kg_matrix.npz", self.A)
        np.save(d / "kg_idf.npy", self.idf)
        (d / "kg_entities.json").write_text(json.dumps(self.entities))

    @classmethod
    def load(cls, directory: str) -> "EntityGraph":
        d = Path(directory)
        g = cls()
        g.A = sparse.load_npz(d / "kg_matrix.npz").tocsr().astype(np.float32)
        g.AT = g.A.T.tocsr()
        g.idf = np.load(d / "kg_idf.npy")
        g.entities = json.loads((d / "kg_entities.json").read_text())
        g.index = {term: i for i, term in enumerate(g.entities)}
        return g

    def stats(self) -> Dict[str, float]:
        n_pass, n_ent = self.A.shape
        return {"passages": int(n_pass), "entities": int(n_ent), "edges": int(self.A.nnz),
                "entities_per_passage": float(self.A.nnz / max(n_pass, 1))}


def _row_top_k(mat: sparse.csr_matrix, k: int) -> sparse.csr_matrix:
    """Keep the k largest entries of every row of a CSR matrix."""
    mat = mat.tocsr()
    indptr, indices, data = mat.indptr, mat.indices, mat.data
    keep = np.ones(data.shape[0], dtype=bool)
    lengths = np.diff(indptr)
    for row in np.flatnonzero(lengths > k):
        lo, hi = indptr[row], indptr[row + 1]
        seg = data[lo:hi]
        drop = np.argpartition(-seg, k - 1)[k:]
        keep[lo + drop] = False
    new_data, new_indices = data[keep], indices[keep]
    kept_before = np.concatenate([[0], np.cumsum(keep)])
    new_indptr = kept_before[indptr]
    return sparse.csr_matrix((new_data, new_indices, new_indptr), shape=mat.shape, dtype=np.float32)
