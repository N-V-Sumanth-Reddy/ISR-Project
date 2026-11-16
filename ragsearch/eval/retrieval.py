"""Document-level retrieval runs for benchmark evaluation.

The system retrieves passages. For a document-level benchmark such as TREC-COVID each
retriever's passage list is collapsed to documents by MaxP (a document's score is that of
its best passage); fusion and re-ranking then operate on the document lists.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..bm25 import top_n_sorted
from ..collection import Collection
from ..fusion import rrf, weighted_fusion

DocList = List[Tuple[str, float, int]]          # (doc_id, score, best passage id), best first


def collapse_to_docs(pids: Sequence[int], scores: Sequence[float], doc_ids: Sequence[str], top_docs: int = 100) -> DocList:
    out: DocList = []
    seen = set()
    for pid, score in zip(pids, scores):
        doc = doc_ids[int(pid)]
        if doc in seen:
            continue
        seen.add(doc)
        out.append((doc, float(score), int(pid)))
        if len(out) >= top_docs:
            break
    return out


def bm25_doc_runs(col: Collection, queries: Dict[str, str], k1: Optional[float] = None, b: Optional[float] = None,
                  top_docs: int = 100, passage_depth: int = 1000) -> Dict[str, DocList]:
    runs = {}
    for qid, text in queries.items():
        pids, scores = col.bm25.search(text, top_n=passage_depth, k1=k1, b=b)
        runs[qid] = collapse_to_docs(pids, scores, col.doc_ids, top_docs)
    return runs


def dense_doc_runs(col: Collection, queries: Dict[str, str], model: str, top_docs: int = 100,
                   passage_depth: int = 1000) -> Dict[str, DocList]:
    qids = list(queries)
    if not qids:
        return {}
    q_vecs = col.encoder(model).encode_queries([queries[q] for q in qids])
    ids, scores = col.dense[model].search(q_vecs, top_n=passage_depth)
    return {qid: collapse_to_docs(ids[i], scores[i], col.doc_ids, top_docs) for i, qid in enumerate(qids)}


def _best_pid(*lists: DocList) -> Dict[str, int]:
    best: Dict[str, int] = {}
    for lst in lists:
        for doc, _, pid in lst:
            best.setdefault(doc, pid)
    return best


def rrf_doc_runs(a: Dict[str, DocList], b: Dict[str, DocList], rrf_k: int = 60, depth: int = 100,
                 names: Tuple[str, str] = ("bm25", "dense"), weights: Optional[Dict[str, float]] = None) -> Dict[str, DocList]:
    out = {}
    for qid in a:
        la, lb = a[qid], b.get(qid, [])
        best = _best_pid(la, lb)
        fused = rrf({names[0]: [d for d, _, _ in la], names[1]: [d for d, _, _ in lb]}, k=rrf_k, depth=depth, weights=weights)
        out[qid] = [(doc, float(score), best[doc]) for doc, score in fused]
    return out


def weighted_doc_runs(bm25: Dict[str, DocList], dense: Dict[str, DocList], alpha) -> Dict[str, DocList]:
    """alpha is the dense weight: a float, or a per-query dict (dynamic alpha)."""
    out = {}
    for qid in bm25:
        lb, ld = bm25[qid], dense.get(qid, [])
        best = _best_pid(lb, ld)
        a = alpha[qid] if isinstance(alpha, dict) else alpha
        fused = weighted_fusion({d: s for d, s, _ in ld}, {d: s for d, s, _ in lb}, alpha=float(a))
        out[qid] = [(doc, float(score), best[doc]) for doc, score in fused]
    return out


def rerank_doc_runs(col: Collection, queries: Dict[str, str], runs: Dict[str, DocList], reranker, top_n: int = 50) -> Dict[str, DocList]:
    """Re-score the top_n documents (by their best passage) with a cross-encoder."""
    out = {}
    for qid, lst in runs.items():
        head, tail = lst[:top_n], lst[top_n:]
        scores = reranker.score(queries[qid], [col.passages[pid].index_text for _, _, pid in head])
        order = sorted(range(len(head)), key=lambda i: (-float(scores[i]), i))
        floor = min([float(s) for s in scores], default=0.0)
        # documents below the re-ranking cut keep their fused order, placed after the head
        out[qid] = [(head[i][0], float(scores[i]), head[i][2]) for i in order] + \
                   [(doc, floor - 1.0 - j, pid) for j, (doc, _, pid) in enumerate(tail)]
    return out


def kg_doc_runs(col: Collection, queries: Dict[str, str], base: Dict[str, DocList], rrf_k: int = 60, depth: int = 100,
                kg_weight: float = 1.0, base_weight: float = 2.0, n_seeds: int = 10, passage_depth: int = 1000) -> Dict[str, DocList]:
    """Entity-graph expansion: seeds = best passages of the top fused documents."""
    if col.kg is None:
        return base
    out = {}
    for qid, lst in base.items():
        seeds = [pid for _, _, pid in lst[:n_seeds]]
        weights = [1.0 / (rrf_k + r) for r in range(1, len(seeds) + 1)]
        kg_scores = col.kg.propagate(queries[qid], seeds, weights)
        pids, scores = top_n_sorted(kg_scores, passage_depth, positive_only=True)
        kg_docs = collapse_to_docs(pids, scores, col.doc_ids, depth)
        best = _best_pid(lst, kg_docs)
        fused = rrf({"base": [d for d, _, _ in lst], "kg": [d for d, _, _ in kg_docs]}, k=rrf_k, depth=depth,
                    weights={"base": base_weight, "kg": kg_weight})
        out[qid] = [(doc, float(score), best[doc]) for doc, score in fused]
    return out


def as_run(runs: Dict[str, DocList]) -> Dict[str, List[str]]:
    return {qid: [doc for doc, _, _ in lst] for qid, lst in runs.items()}


def overlap_at_k(a: Dict[str, DocList], b: Dict[str, DocList], k: int = 10) -> float:
    """Mean fraction of the top-k shared by two runs (how complementary two retrievers are)."""
    vals = []
    for qid in a:
        sa = {d for d, _, _ in a[qid][:k]}
        sb = {d for d, _, _ in b.get(qid, [])[:k]}
        vals.append(len(sa & sb) / float(k))
    return float(np.mean(vals)) if vals else 0.0
