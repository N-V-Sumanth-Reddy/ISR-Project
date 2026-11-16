"""Retrieval metrics (trec_eval conventions) and small-sample statistics."""
from __future__ import annotations

import math
import random
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

Run = Mapping[str, Sequence[str]]            # query id -> ranked document ids
Qrels = Mapping[str, Mapping[str, int]]      # query id -> {document id: graded relevance}


def ndcg_at_k(ranked: Sequence[str], rels: Mapping[str, int], k: int = 10) -> float:
    """nDCG@k with linear gain and log2 discount, as computed by trec_eval / pytrec_eval / BEIR."""
    dcg = 0.0
    for i, doc in enumerate(ranked[:k]):
        gain = rels.get(doc, 0)
        if gain > 0:
            dcg += gain / math.log2(i + 2)
    ideal = sorted((g for g in rels.values() if g > 0), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(ranked: Sequence[str], rels: Mapping[str, int], k: int = 100) -> float:
    relevant = {d for d, g in rels.items() if g > 0}
    if not relevant:
        return 0.0
    return len(relevant.intersection(ranked[:k])) / len(relevant)


def capped_recall_at_k(ranked: Sequence[str], rels: Mapping[str, int], k: int = 100) -> float:
    """Recall with the denominator capped at k (BEIR's R_cap@k, used for TREC-COVID
    because each topic has hundreds of relevant documents)."""
    relevant = {d for d, g in rels.items() if g > 0}
    if not relevant:
        return 0.0
    return len(relevant.intersection(ranked[:k])) / min(len(relevant), k)


def precision_at_k(ranked: Sequence[str], rels: Mapping[str, int], k: int = 10) -> float:
    return sum(1 for d in ranked[:k] if rels.get(d, 0) > 0) / float(k)


def mrr_at_k(ranked: Sequence[str], rels: Mapping[str, int], k: int = 10) -> float:
    for i, doc in enumerate(ranked[:k], start=1):
        if rels.get(doc, 0) > 0:
            return 1.0 / i
    return 0.0


METRICS: Dict[str, Callable[[Sequence[str], Mapping[str, int]], float]] = {
    "ndcg@10": lambda r, q: ndcg_at_k(r, q, 10),
    "recall@100": lambda r, q: recall_at_k(r, q, 100),
    "capped_recall@100": lambda r, q: capped_recall_at_k(r, q, 100),
    "p@10": lambda r, q: precision_at_k(r, q, 10),
    "mrr@10": lambda r, q: mrr_at_k(r, q, 10),
    "recall@1": lambda r, q: recall_at_k(r, q, 1),
    "recall@5": lambda r, q: recall_at_k(r, q, 5),
    "recall@10": lambda r, q: recall_at_k(r, q, 10),
}


def evaluate_run(run: Run, qrels: Qrels, qids: Optional[Sequence[str]] = None,
                 metrics: Sequence[str] = ("ndcg@10", "recall@100", "capped_recall@100")) -> Dict[str, Dict[str, float]]:
    """Per-query values for each metric. Queries missing from the run score 0."""
    qids = list(qids) if qids is not None else list(qrels.keys())
    return {m: {q: METRICS[m](run.get(q, []), qrels.get(q, {})) for q in qids} for m in metrics}


def mean(values: Sequence[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def bootstrap_ci(values: Sequence[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> Tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean."""
    values = list(values)
    if len(values) < 2:
        m = mean(values)
        return (m, m)
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return (lo, hi)


def paired_randomization_test(a: Sequence[float], b: Sequence[float], n_perm: int = 10000, seed: int = 0) -> float:
    """Two-sided paired randomisation (sign-flip) test on per-query differences."""
    diffs = [x - y for x, y in zip(a, b)]
    if not diffs or all(abs(d) < 1e-12 for d in diffs):
        return 1.0
    observed = abs(sum(diffs))
    rng = random.Random(seed)
    extreme = 0
    for _ in range(n_perm):
        total = sum(d if rng.random() < 0.5 else -d for d in diffs)
        if abs(total) >= observed - 1e-12:
            extreme += 1
    return (extreme + 1) / (n_perm + 1)


def sign_test(wins: int, losses: int) -> float:
    """Two-sided exact binomial sign test (ties dropped)."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2.0 ** n)
    return min(1.0, 2.0 * tail)


def wilson_interval(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def auroc(pos: Sequence[float], neg: Sequence[float]) -> float:
    """Probability that a random positive scores higher than a random negative."""
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else (0.5 if p == n else 0.0)
    return wins / (len(pos) * len(neg))


def cohen_kappa(a: Sequence[str], b: Sequence[str]) -> float:
    labels = sorted(set(a) | set(b))
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pe = sum((list(a).count(l) / n) * (list(b).count(l) / n) for l in labels)
    return (po - pe) / (1 - pe) if abs(1 - pe) > 1e-12 else 1.0
