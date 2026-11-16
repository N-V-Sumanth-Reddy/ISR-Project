"""Hybrid retrieval over a Collection: BM25, dense, RRF / weighted fusion, re-ranking, KG."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .bm25 import top_n_sorted
from .chunking import Passage
from .collection import Collection
from .fusion import rrf, weighted_fusion
from .rerank import get_reranker


@dataclass
class Hit:
    pid: int
    rank: int
    score: float
    bm25_rank: Optional[int] = None
    bm25_score: Optional[float] = None
    dense_rank: Optional[int] = None
    dense_score: Optional[float] = None
    fused_score: Optional[float] = None
    rerank_score: Optional[float] = None
    kg_rank: Optional[int] = None

    @property
    def found_by(self) -> str:
        """Which retriever surfaced this passage inside its own top-``depth`` list."""
        if self.bm25_rank is not None and self.dense_rank is not None:
            return "both"
        if self.bm25_rank is not None:
            return "bm25"
        if self.dense_rank is not None:
            return "dense"
        return "kg" if self.kg_rank is not None else "none"

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["found_by"] = self.found_by
        return d


@dataclass
class RetrievalResult:
    query: str
    mode: str
    hits: List[Hit]
    passages: List[Passage]
    signals: Dict[str, Optional[float]] = field(default_factory=dict)
    timings: Dict[str, float] = field(default_factory=dict)
    settings: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {"query": self.query, "mode": self.mode, "hits": [h.to_dict() for h in self.hits],
                "signals": self.signals, "timings": self.timings, "settings": self.settings}


class Retriever:
    def __init__(self, collection: Collection, cfg=None):
        self.col = collection
        self.cfg = cfg or collection.cfg

    # --------------------------------------------------------- first stage
    def first_stage(self, query: str, depth: int, k1: Optional[float] = None, b: Optional[float] = None,
                    dense_model: Optional[str] = None) -> Dict:
        out: Dict = {"timings": {}}
        t0 = time.time()
        b_ids, b_scores = self.col.bm25.search(query, top_n=depth, k1=k1, b=b)
        out["bm25"] = (b_ids, b_scores)
        out["timings"]["bm25"] = time.time() - t0

        model = dense_model or self.col.default_dense_model()
        out["dense_model"] = model
        if model is not None and model in self.col.dense:
            t1 = time.time()
            q_vec = self.col.encoder(model).encode_queries([query])
            d_ids, d_scores = self.col.dense[model].search(q_vec, top_n=depth)
            out["dense"] = (d_ids[0], d_scores[0])
            out["timings"]["dense"] = time.time() - t1
        else:
            out["dense"] = (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32))
        return out

    # --------------------------------------------------------------- search
    def search(self, query: str, mode: Optional[str] = None, top_k: Optional[int] = None,
               rerank: Optional[bool] = None, use_kg: Optional[bool] = None, depth: Optional[int] = None,
               rrf_k: Optional[int] = None, fusion: Optional[str] = None, alpha: Optional[float] = None,
               k1: Optional[float] = None, b: Optional[float] = None, dense_model: Optional[str] = None,
               rerank_top_n: Optional[int] = None) -> RetrievalResult:
        cfg = self.cfg
        mode = mode or cfg.retrieval.mode
        top_k = int(top_k or cfg.retrieval.top_k)
        rerank = cfg.retrieval.rerank if rerank is None else rerank
        use_kg = cfg.retrieval.kg if use_kg is None else use_kg
        depth = int(depth or cfg.fusion.depth)
        rrf_k = int(rrf_k or cfg.fusion.rrf_k)
        fusion = fusion or cfg.fusion.method
        alpha = cfg.fusion.alpha if alpha is None else alpha
        rerank_top_n = int(rerank_top_n or cfg.rerank.top_n)
        if mode not in ("bm25", "dense", "hybrid"):
            raise ValueError("mode must be bm25, dense or hybrid")

        t_all = time.time()
        stage = self.first_stage(query, depth, k1, b, dense_model)
        b_ids, b_scores = stage["bm25"]
        d_ids, d_scores = stage["dense"]
        timings = dict(stage["timings"])
        has_dense = d_ids.size > 0
        if mode in ("dense", "hybrid") and not has_dense:
            mode = "bm25"                                   # no dense index available: degrade gracefully

        b_rank = {int(p): r for r, p in enumerate(b_ids, start=1)}
        d_rank = {int(p): r for r, p in enumerate(d_ids, start=1)}
        b_score = {int(p): float(s) for p, s in zip(b_ids, b_scores)}
        d_score = {int(p): float(s) for p, s in zip(d_ids, d_scores)}

        # The fused list is always computed: it feeds the confidence signal and the badges.
        fused_rrf = rrf({"bm25": [int(p) for p in b_ids], "dense": [int(p) for p in d_ids]}, k=rrf_k, depth=depth)
        fused_score: Dict[int, float] = {}
        if mode == "bm25":
            ranking = [(int(p), float(s)) for p, s in zip(b_ids, b_scores)]
        elif mode == "dense":
            ranking = [(int(p), float(s)) for p, s in zip(d_ids, d_scores)]
        elif fusion == "weighted":
            ranking = [(int(p), float(s)) for p, s in weighted_fusion(d_score, b_score, alpha=float(alpha))]
            fused_score = dict(ranking)
        else:
            ranking = [(int(p), float(s)) for p, s in fused_rrf]
            fused_score = dict(ranking)

        kg_rank: Dict[int, int] = {}
        if use_kg and self.col.kg is not None and ranking:
            t_kg = time.time()
            seeds = [p for p, _ in ranking[:10]]
            weights = [1.0 / (rrf_k + r) for r in range(1, len(seeds) + 1)]
            kg_scores = self.col.kg.propagate(query, seeds, weights)
            kg_ids, _ = top_n_sorted(kg_scores, depth, positive_only=True)
            kg_rank = {int(p): r for r, p in enumerate(kg_ids, start=1)}
            base = [p for p, _ in ranking[:depth]]
            w_kg = float(self.cfg.get("kg", {}).get("weight", 1.0))
            # the first-stage ranking counts once per retriever it stands for
            w_base = 2.0 if mode == "hybrid" else 1.0
            ranking = [(int(p), float(s)) for p, s in
                       rrf({"base": base, "kg": [int(p) for p in kg_ids]}, k=rrf_k, depth=depth,
                           weights={"base": w_base, "kg": w_kg})]
            timings["kg"] = time.time() - t_kg

        rerank_score: Dict[int, float] = {}
        if rerank and ranking:
            t_rr = time.time()
            head = ranking[:rerank_top_n]
            ce = get_reranker(cfg.rerank.model, cfg.dense.get("device", "auto"))
            scores = ce.score(query, [self.col.passages[p].index_text for p, _ in head])
            rerank_score = {p: float(s) for (p, _), s in zip(head, scores)}
            order = sorted(range(len(head)), key=lambda i: (-scores[i], i))
            ranking = [(head[i][0], float(scores[i])) for i in order] + ranking[rerank_top_n:]
            timings["rerank"] = time.time() - t_rr

        hits = []
        for rank, (pid, score) in enumerate(ranking[:top_k], start=1):
            hits.append(Hit(pid=pid, rank=rank, score=score, bm25_rank=b_rank.get(pid), bm25_score=b_score.get(pid),
                            dense_rank=d_rank.get(pid), dense_score=d_score.get(pid),
                            fused_score=fused_score.get(pid), rerank_score=rerank_score.get(pid),
                            kg_rank=kg_rank.get(pid)))
        timings["total"] = time.time() - t_all

        top10_b, top10_d = set(list(b_rank)[:10]), set(list(d_rank)[:10])
        signals = {
            "rrf_top": float(fused_rrf[0][1]) if fused_rrf else 0.0,
            "dense_top": float(d_scores[0]) if has_dense else None,
            "bm25_top": float(b_scores[0]) if b_ids.size else 0.0,
            "ce_top": max(rerank_score.values()) if rerank_score else None,
            "overlap10": (len(top10_b & top10_d) / 10.0) if has_dense else None,
        }
        settings = {"mode": mode, "top_k": top_k, "rerank": bool(rerank), "kg": bool(use_kg and self.col.kg is not None),
                    "depth": depth, "rrf_k": rrf_k, "fusion": fusion, "alpha": alpha,
                    "dense_model": stage.get("dense_model"), "k1": k1 if k1 is not None else self.col.bm25.k1,
                    "b": b if b is not None else self.col.bm25.b}
        return RetrievalResult(query=query, mode=mode, hits=hits, passages=[self.col.passages[h.pid] for h in hits],
                               signals=signals, timings=timings, settings=settings)
