"""Abstention: threshold tuning on dev, and abstention / over-refusal metrics."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from .metrics import auroc, wilson_interval


def tune_threshold(answerable: Sequence[float], unanswerable: Sequence[float], max_over_refusal: float = 0.10) -> Dict:
    """Choose a threshold t: abstain when signal < t.

    Maximises Youden's J = (unanswerable caught) - (answerable refused) among thresholds
    whose over-refusal on the dev answerable questions is at most ``max_over_refusal``.
    Ties go to the lower threshold (fewer refusals).
    """
    ans = [float(v) for v in answerable if v is not None]
    una = [float(v) for v in unanswerable if v is not None]
    if not ans or not una:
        return {"threshold": None, "auroc": None, "tpr": None, "fpr": None, "j": None}
    values = sorted(set(ans + una))
    cands = [values[0] - 1e-9] + [(a + b) / 2.0 for a, b in zip(values, values[1:])] + [values[-1] + 1e-9]
    best = None
    for t in cands:
        tpr = sum(1 for v in una if v < t) / len(una)
        fpr = sum(1 for v in ans if v < t) / len(ans)
        if fpr > max_over_refusal + 1e-12:
            continue
        j = tpr - fpr
        if best is None or j > best["j"] + 1e-12:
            best = {"threshold": t, "tpr": tpr, "fpr": fpr, "j": j}
    best = best or {"threshold": None, "tpr": None, "fpr": None, "j": None}
    # higher signal should mean "answerable": AUROC of answerable (positive) over unanswerable
    best["auroc"] = auroc(ans, una)
    best["n_answerable"], best["n_unanswerable"] = len(ans), len(una)
    return best


def rates_at_threshold(answerable: Sequence[float], unanswerable: Sequence[float], threshold: float) -> Dict:
    """Apply a fixed threshold (tuned elsewhere, e.g. on dev) to new questions; abstain when signal < threshold.

    ``caught`` is the share of unanswerable questions abstained on, ``refused`` the share of
    answerable questions abstained on (over-refusal), each with a Wilson 95% interval.
    """
    ans = [float(v) for v in answerable if v is not None]
    una = [float(v) for v in unanswerable if v is not None]
    n_caught = sum(1 for v in una if v < threshold)
    n_refused = sum(1 for v in ans if v < threshold)
    out = {"threshold": float(threshold), "n_answerable": len(ans), "n_unanswerable": len(una),
           "n_caught": n_caught, "n_refused": n_refused,
           "caught": n_caught / len(una) if una else None, "refused": n_refused / len(ans) if ans else None}
    if una:
        out["caught_ci"] = wilson_interval(n_caught, len(una))
    if ans:
        out["refused_ci"] = wilson_interval(n_refused, len(ans))
    return out


def abstention_metrics(abstained_on_unanswerable: Sequence[bool], abstained_on_answerable: Sequence[bool]) -> Dict:
    """Positive class = "should abstain" (the question is not answerable from the collection)."""
    tp = sum(1 for a in abstained_on_unanswerable if a)
    fn = len(abstained_on_unanswerable) - tp
    fp = sum(1 for a in abstained_on_answerable if a)
    tn = len(abstained_on_answerable) - fp
    recall = tp / (tp + fn) if (tp + fn) else None
    precision = tp / (tp + fp) if (tp + fp) else None
    f1 = (2 * precision * recall / (precision + recall)) if precision and recall else (0.0 if (tp + fn) else None)
    over = fp / (fp + tn) if (fp + tn) else None
    total = tp + fn + fp + tn
    out = {"n_unanswerable": tp + fn, "n_answerable": fp + tn, "abstained_unanswerable": tp, "abstained_answerable": fp,
           "abstention_accuracy": recall, "over_refusal_rate": over, "abstention_precision": precision, "abstention_f1": f1,
           "overall_accuracy": (tp + tn) / total if total else None}
    if tp + fn:
        out["abstention_accuracy_ci"] = wilson_interval(tp, tp + fn)
    if fp + tn:
        out["over_refusal_ci"] = wilson_interval(fp, fp + tn)
    return out


def apply_policy(llm_abstained: bool, signal_value: Optional[float], threshold: Optional[float], policy: str) -> bool:
    below = threshold is not None and signal_value is not None and float(signal_value) < float(threshold)
    if policy == "llm":
        return bool(llm_abstained)
    if policy == "threshold":
        return bool(below)
    if policy == "combined":
        return bool(llm_abstained or below)
    raise ValueError(policy)
