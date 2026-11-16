"""Rank fusion: Reciprocal Rank Fusion, weighted score fusion, and DAT-style dynamic alpha."""
from __future__ import annotations

from typing import Dict, Hashable, List, Mapping, Optional, Sequence, Tuple

Ranking = Sequence[Hashable]


def rrf(rankings: Mapping[str, Ranking], k: int = 60, depth: Optional[int] = 100,
        weights: Optional[Mapping[str, float]] = None) -> List[Tuple[Hashable, float]]:
    """Reciprocal Rank Fusion (Cormack et al., 2009).

        score(d) = sum_r  w_r / (k + rank_r(d))      with ranks starting at 1

    Only the top ``depth`` items of each ranking contribute. Ties are broken by the best
    individual rank, then by first appearance, so the output is deterministic.
    """
    scores: Dict[Hashable, float] = {}
    best_rank: Dict[Hashable, int] = {}
    first_seen: Dict[Hashable, int] = {}
    counter = 0
    for name, ranking in rankings.items():
        w = 1.0 if weights is None else float(weights.get(name, 1.0))
        items = ranking if depth is None else ranking[:depth]
        for rank, item in enumerate(items, start=1):
            scores[item] = scores.get(item, 0.0) + w / (k + rank)
            if item not in best_rank or rank < best_rank[item]:
                best_rank[item] = rank
            if item not in first_seen:
                first_seen[item] = counter
                counter += 1
    return sorted(scores.items(), key=lambda kv: (-kv[1], best_rank[kv[0]], first_seen[kv[0]]))


def minmax(scores: Mapping[Hashable, float]) -> Dict[Hashable, float]:
    if not scores:
        return {}
    lo, hi = min(scores.values()), max(scores.values())
    if hi - lo < 1e-12:
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def weighted_fusion(dense: Mapping[Hashable, float], sparse: Mapping[Hashable, float],
                    alpha: float = 0.5) -> List[Tuple[Hashable, float]]:
    """Convex combination of min-max normalised scores: alpha * dense + (1 - alpha) * bm25.

    Items missing from one list get 0 for that list (the usual hybrid-search convention).
    """
    nd, ns = minmax(dense), minmax(sparse)
    order: Dict[Hashable, int] = {}
    for item in list(dense) + list(sparse):
        order.setdefault(item, len(order))
    fused = {item: alpha * nd.get(item, 0.0) + (1.0 - alpha) * ns.get(item, 0.0) for item in order}
    return sorted(fused.items(), key=lambda kv: (-kv[1], order[kv[0]]))


def dat_alpha(dense_top1_score: float, bm25_top1_score: float) -> float:
    """Dynamic alpha in the spirit of DAT (Hsu & Tzeng, 2025).

    An LLM rates the top-1 result of each retriever on a 0-5 scale; the dense weight is

        alpha = 0.5                    if both ratings are 0
              = 1.0                    if dense is 5 and BM25 is not
              = 0.0                    if BM25 is 5 and dense is not
              = S_dense / (S_dense + S_bm25)   otherwise
    """
    sd, sb = float(dense_top1_score), float(bm25_top1_score)
    if sd <= 0 and sb <= 0:
        return 0.5
    if sd >= 5 and sb < 5:
        return 1.0
    if sb >= 5 and sd < 5:
        return 0.0
    return sd / (sd + sb)
