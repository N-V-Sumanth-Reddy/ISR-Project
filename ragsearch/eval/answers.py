"""End-to-end answer evaluation: closed-book LLM vs. the RAG system on a QA benchmark.

A benchmark is a dict:
    name        str
    collection  Collection
    test        list of questions   {qid, question, reference, label?, gold_doc, answerable}
    dev         list of questions or None
    style       extra instruction appended to both prompts (e.g. yes/no format)

Everything that calls an LLM goes through the SQLite cache, so the pipeline can be re-run
or resumed at no cost.
"""
from __future__ import annotations

import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from ..generate import (NOT_FOUND, answer_closed_book, answer_rag, answer_text_for_judging,
                        is_closed_book_refusal)
from ..llm import LLM
from ..rerank import get_reranker
from ..retriever import RetrievalResult, Retriever
from . import citation as citation_eval
from . import faithfulness as faith_eval
from .abstention import abstention_metrics, apply_policy, rates_at_threshold, tune_threshold
from .judge import decision_of, pairwise
from .metrics import mean, sign_test, wilson_interval

SIGNALS = ("dense_top", "rrf_top", "bm25_top", "ce_top", "overlap10")
K_MAX = 10


def log(msg: str) -> None:
    print(msg, flush=True)


def _mean_or_none(values) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return mean(vals) if vals else None


# ------------------------------------------------------------------------- retrieval

def retrieve(retriever: Retriever, question: str, settings: Dict, reranker=None) -> RetrievalResult:
    """Retrieve K_MAX passages with the given settings; add the cross-encoder confidence signal."""
    res = retriever.search(question, mode=settings.get("mode"), top_k=K_MAX, rerank=settings.get("rerank", False),
                           use_kg=settings.get("kg", False))
    if res.signals.get("ce_top") is None and reranker is not None and res.passages:
        scores = reranker.score(question, [p.index_text for p in res.passages])
        res.signals["ce_top"] = float(max(scores))
    return res


def top_k_view(res: RetrievalResult, k: int) -> RetrievalResult:
    return RetrievalResult(query=res.query, mode=res.mode, hits=res.hits[:k], passages=res.passages[:k],
                           signals=res.signals, timings=res.timings, settings=dict(res.settings, top_k=k))


def gold_rank(res: RetrievalResult, gold_doc: Optional[str]) -> Optional[int]:
    if gold_doc is None:
        return None
    for i, p in enumerate(res.passages, start=1):
        if p.doc_id == gold_doc:
            return i
    return None


# ------------------------------------------------------------------------ generation

def rag_record(q: Dict, res: RetrievalResult, k: int, llm: LLM, style: str, kg=None) -> Dict:
    view = top_k_view(res, k)
    ans = answer_rag(q["question"], view, llm, style=style, kg=kg)
    sents = ans.sentences
    return {
        "raw": ans.raw, "text": ans.text, "abstained_llm": ans.abstained, "citations": ans.citations,
        "invalid_citations": ans.invalid_citations, "n_sentences": len(sents), "n_uncited": ans.n_uncited_sentences,
        "unsupported_numbers": sorted({n for s in sents for n in s.unsupported_numbers}),
        "unsupported_terms": sorted({t for s in sents for t in s.unsupported_terms}),
        "seconds": ans.seconds, "prompt_tokens": ans.prompt_tokens, "completion_tokens": ans.completion_tokens,
        "k": k, "passage_pids": [h.pid for h in view.hits], "passage_docs": [p.doc_id for p in view.passages],
        "found_by": [h.found_by for h in view.hits],
    }


def utility(records: Sequence[Dict], policy: str, signal: Optional[str], threshold: Optional[float],
            rag_key: str = "rag") -> Dict:
    """Fraction of questions handled correctly: an answerable question must be answered
    (and, when gold labels exist, with the right decision); an out-of-collection question
    must be declined."""
    ok = ans_total = ans_ok = una_total = una_ok = 0
    for r in records:
        rag = r[rag_key]
        abst = apply_policy(rag["abstained_llm"], r["signals"].get(signal) if signal else None, threshold, policy)
        if r["answerable"]:
            ans_total += 1
            good = (not abst) and (r.get("label") is None or rag.get("decision") == r["label"])
            ans_ok += int(good)
        else:
            una_total += 1
            good = abst
            una_ok += int(good)
        ok += int(good)
    n = len(records)
    return {"utility": ok / n if n else None, "answerable_ok": ans_ok / ans_total if ans_total else None,
            "unanswerable_ok": una_ok / una_total if una_total else None, "n": n}


# ------------------------------------------------------------------------ main entry

def run(bench: Dict, cfg, gen_llm: LLM, judge_llm: LLM, out_dir: Path, settings: Optional[Dict] = None,
        transfer: Optional[Dict] = None, do_judge: bool = True, do_citation: bool = True,
        do_faithfulness: bool = True, entails=None) -> Dict:
    """Run the full answer evaluation for one benchmark and write results to ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    col = bench["collection"]
    style = bench.get("style", "")
    retriever = Retriever(col, cfg)
    settings = dict({"mode": cfg.retrieval.mode, "rerank": cfg.retrieval.rerank, "kg": cfg.retrieval.kg}, **(settings or {}))
    try:
        reranker = get_reranker(cfg.rerank.model, cfg.dense.get("device", "auto"))
    except Exception as exc:                      # the cross-encoder is optional
        log("  (cross-encoder unavailable, ce_top signal skipped: %s)" % str(exc)[:120])
        reranker = None
    use_kg_check = col.kg if settings.get("kg") else None

    def prepare(questions: Sequence[Dict]) -> List[Dict]:
        recs = []
        for q in questions:
            res = retrieve(retriever, q["question"], settings, reranker)
            rec = dict(q)
            rec["signals"] = dict(res.signals)
            rec["gold_rank"] = gold_rank(res, q.get("gold_doc"))
            rec["_res"] = res
            recs.append(rec)
        return recs

    # ---------------------------------------------------------------- dev tuning
    tuning: Dict = {"source": "dev"}
    k_grid = list(cfg.eval.get("top_k_grid", [3, 5, 8]))
    if bench.get("dev"):
        dev = prepare(bench["dev"])
        log("[%s] dev tuning on %d questions (top-k grid %s)" % (bench["name"], len(dev), k_grid))
        per_k = {}
        for k in k_grid:
            for i, r in enumerate(dev):
                rag = rag_record(r, r["_res"], k, gen_llm, style, use_kg_check)
                rag["decision"] = None if rag["abstained_llm"] else decision_of(rag["text"], r["question"], judge_llm if r.get("label") else None)
                r["rag_k%d" % k] = rag
                if (i + 1) % 20 == 0:
                    log("    k=%d: %d/%d" % (k, i + 1, len(dev)))
            u = utility(dev, "llm", None, None, rag_key="rag_k%d" % k)
            u["gold_in_top_k"] = mean([1.0 if (r["gold_rank"] is not None and r["gold_rank"] <= k) else 0.0 for r in dev if r["answerable"]])
            u["mean_prompt_tokens"] = mean([r["rag_k%d" % k]["prompt_tokens"] for r in dev])
            u["mean_seconds"] = mean([r["rag_k%d" % k]["seconds"] for r in dev])
            per_k[str(k)] = u
            log("  k=%d: dev utility %.3f (answerable ok %.3f, out-of-collection declined %.3f)" % (
                k, u["utility"], u["answerable_ok"] or 0.0, u["unanswerable_ok"] or 0.0))
        best_k = max(k_grid, key=lambda k: (per_k[str(k)]["utility"], -k))
        sig_tuning = {}
        for sig in SIGNALS:
            ans = [r["signals"].get(sig) for r in dev if r["answerable"]]
            una = [r["signals"].get(sig) for r in dev if not r["answerable"]]
            if any(v is None for v in ans + una):
                continue
            sig_tuning[sig] = tune_threshold(ans, una)
        usable = {s: t for s, t in sig_tuning.items() if t.get("threshold") is not None and t.get("auroc") is not None}
        best_sig = max(usable, key=lambda s: usable[s]["auroc"]) if usable else None
        threshold = usable[best_sig]["threshold"] if best_sig else None
        policies = {}
        for r in dev:
            r["rag"] = r["rag_k%d" % best_k]
        for policy in ("llm", "threshold", "combined"):
            policies[policy] = utility(dev, policy, best_sig, threshold)
        tuning.update({"top_k": {"grid": per_k, "best": best_k}, "signals": sig_tuning, "signal": best_sig,
                       "threshold": threshold, "dev_policies": policies, "n_dev": len(dev)})
    else:
        tuning = dict(transfer or {})
        tuning["source"] = "transferred" if transfer else "defaults"
        best_k = int((transfer or {}).get("top_k", {}).get("best", cfg.retrieval.top_k))
        best_sig = (transfer or {}).get("signal")
        threshold = (transfer or {}).get("threshold")
    primary_policy = "combined" if threshold is not None else "llm"
    (out_dir / "dev_tuning.json").write_text(json.dumps(tuning, indent=2, default=str))
    log("[%s] top-k = %d, abstention signal = %s, threshold = %s, policy = %s" % (
        bench["name"], best_k, best_sig, ("%.4f" % threshold) if threshold is not None else None, primary_policy))

    # ---------------------------------------------------------------------- test
    test = prepare(bench["test"])
    log("[%s] test: %d questions (%d answerable, %d out-of-collection)" % (
        bench["name"], len(test), sum(r["answerable"] for r in test), sum(not r["answerable"] for r in test)))
    for i, r in enumerate(test):
        cb = answer_closed_book(r["question"], gen_llm, style)
        r["closed_book"] = {"text": cb.text, "refused": is_closed_book_refusal(cb.text), "seconds": cb.seconds,
                            "prompt_tokens": cb.prompt_tokens, "completion_tokens": cb.completion_tokens}
        r["rag"] = rag_record(r, r["_res"], best_k, gen_llm, style, use_kg_check)
        if (i + 1) % 20 == 0:
            log("    generated %d/%d" % (i + 1, len(test)))

    has_labels = any(r.get("label") for r in test)
    for r in test:
        rag = r["rag"]
        r["policies"] = {p: apply_policy(rag["abstained_llm"], r["signals"].get(best_sig) if best_sig else None, threshold, p)
                         for p in ("llm", "threshold", "combined")}
        abst = r["policies"][primary_policy]
        reason = None
        if abst:
            reason = "llm" if rag["abstained_llm"] else "threshold"
        r["final"] = {"text": NOT_FOUND if abst else rag["text"], "abstained": abst, "reason": reason}
        if has_labels:
            dec_llm = judge_llm if r.get("label") else None
            rag["decision"] = None if rag["abstained_llm"] else decision_of(rag["text"], r["question"], dec_llm)
            r["final"]["decision"] = None if abst else rag["decision"]
            r["closed_book"]["decision"] = decision_of(r["closed_book"]["text"], r["question"], dec_llm)

    answerable = [r for r in test if r["answerable"]]
    unanswerable = [r for r in test if not r["answerable"]]
    summary: Dict = {
        "bench": bench["name"], "n_test_answerable": len(answerable), "n_test_unanswerable": len(unanswerable),
        "top_k": best_k, "settings": settings, "generator_model": gen_llm.model, "judge_model": judge_llm.model,
        "abstention_signal": best_sig, "abstention_threshold": threshold, "primary_policy": primary_policy,
        "tuning_source": tuning.get("source"), "collection": {"name": col.name, **{k: col.meta.get(k) for k in ("n_documents", "n_passages", "chunk_words")}},
    }
    summary["retrieval"] = {
        "gold_in_top_k": mean([1.0 if (r["gold_rank"] is not None and r["gold_rank"] <= best_k) else 0.0 for r in answerable]),
        "gold_at_rank_1": mean([1.0 if r["gold_rank"] == 1 else 0.0 for r in answerable]),
        "gold_in_top_10": mean([1.0 if r["gold_rank"] is not None else 0.0 for r in answerable]),
    }

    # ----------------------------------------------------------- pairwise judging
    if do_judge and answerable:
        log("[%s] pairwise judging (%d questions, both orders)" % (bench["name"], len(answerable)))
        for i, r in enumerate(answerable):
            r["judge"] = pairwise(judge_llm, r["question"], r["reference"],
                                  answer_text_for_judging(r["final"]["text"]), r["closed_book"]["text"])
            if (i + 1) % 25 == 0:
                log("    judged %d/%d" % (i + 1, len(answerable)))
        verdicts = Counter(r["judge"]["verdict"] for r in answerable)
        wins, losses, ties = verdicts.get("x", 0), verdicts.get("y", 0), verdicts.get("tie", 0)
        n = len(answerable)
        compared = [r for r in answerable if not r["judge"].get("identical")]
        answered = [r for r in answerable if not r["final"]["abstained"]]
        v_ans = Counter(r["judge"]["verdict"] for r in answered)
        unparsed = sum(1 for r in compared for o in ("order_xy", "order_yx") if not r["judge"][o]["parsed"])
        summary["judge"] = {
            "n": n, "rag_wins": wins, "closed_book_wins": losses, "ties": ties,
            "rag_win_rate": wins / n, "closed_book_win_rate": losses / n, "tie_rate": ties / n,
            "rag_win_rate_ci": wilson_interval(wins, n), "sign_test_p": sign_test(wins, losses),
            "rag_share_of_decided": wins / (wins + losses) if (wins + losses) else None,
            "position_consistency": mean([1.0 if r["judge"]["consistent"] else 0.0 for r in compared]) if compared else None,
            "unparsed_judgments": unparsed,
            "when_rag_answered": {"n": len(answered), "rag_wins": v_ans.get("x", 0), "closed_book_wins": v_ans.get("y", 0), "ties": v_ans.get("tie", 0)},
        }

    # --------------------------------------------------------- decision accuracy
    if has_labels and answerable:
        n = len(answerable)
        cb_ok = [r["closed_book"].get("decision") == r["label"] for r in answerable]
        rag_ok = [r["final"].get("decision") == r["label"] for r in answerable]
        answered = [r for r in answerable if not r["final"]["abstained"]]
        only_rag = sum(1 for a, b in zip(rag_ok, cb_ok) if a and not b)
        only_cb = sum(1 for a, b in zip(rag_ok, cb_ok) if b and not a)
        by_label = {}
        for lab in sorted({r["label"] for r in answerable}):
            sub = [r for r in answerable if r["label"] == lab]
            by_label[lab] = {"n": len(sub), "closed_book": mean([float(r["closed_book"].get("decision") == lab) for r in sub]),
                             "rag": mean([float(r["final"].get("decision") == lab) for r in sub])}
        summary["decision_accuracy"] = {
            "n": n,
            "closed_book": {"accuracy": sum(cb_ok) / n, "ci": wilson_interval(sum(cb_ok), n),
                            "no_decision": sum(1 for r in answerable if r["closed_book"].get("decision") is None)},
            "rag": {"accuracy": sum(rag_ok) / n, "ci": wilson_interval(sum(rag_ok), n), "coverage": len(answered) / n,
                    "accuracy_when_answered": mean([float(r["final"].get("decision") == r["label"]) for r in answered]) if answered else None,
                    "no_decision": sum(1 for r in answered if r["final"].get("decision") is None)},
            "mcnemar": {"rag_only_correct": only_rag, "closed_book_only_correct": only_cb, "p": sign_test(only_rag, only_cb)},
            "by_label": by_label,
            "majority_baseline": max(Counter(r["label"] for r in answerable).values()) / n,
        }

    # ------------------------------------------------- citations and post-check
    answered_all = [r for r in test if not r["final"]["abstained"]]
    n_sent = sum(r["rag"]["n_sentences"] for r in answered_all)
    summary["post_check"] = {
        "n_answers": len(answered_all), "n_sentences": n_sent,
        "answers_with_invalid_citation": mean([1.0 if r["rag"]["invalid_citations"] else 0.0 for r in answered_all]) if answered_all else None,
        "answers_with_no_citation": mean([1.0 if not r["rag"]["citations"] else 0.0 for r in answered_all]) if answered_all else None,
        "uncited_sentence_rate": (sum(r["rag"]["n_uncited"] for r in answered_all) / n_sent) if n_sent else None,
        "answers_with_unsupported_number": mean([1.0 if r["rag"]["unsupported_numbers"] else 0.0 for r in answered_all]) if answered_all else None,
        "mean_citations_per_answer": mean([len(r["rag"]["citations"]) for r in answered_all]) if answered_all else None,
    }
    if answerable:
        ans_cited = [r for r in answerable if not r["final"]["abstained"] and r["rag"]["citations"]]
        hits = [any(r["rag"]["passage_docs"][c - 1] == r["gold_doc"] for c in r["rag"]["citations"]) for r in ans_cited]
        summary["post_check"]["cites_gold_document"] = mean([float(h) for h in hits]) if hits else None

    if do_citation and answered_all and entails is not None:
        log("[%s] citation quality (ALCE-style) on %d answers" % (bench["name"], len(answered_all)))
        per = []
        for r in answered_all:
            passages = r["_res"].passages[:best_k]
            r["citation"] = citation_eval.citation_quality(r["rag"]["text"], passages, entails)
            per.append(r["citation"])
        summary["citation"] = citation_eval.aggregate(per)
        summary["citation"]["backend"] = getattr(entails, "backend_name", "nli")
        summary["citation"]["nli_window"] = getattr(entails, "nli_window", None)

    if do_faithfulness and answered_all:
        log("[%s] faithfulness (RAGAs-style) on %d answers" % (bench["name"], len(answered_all)))
        scores = []
        for i, r in enumerate(answered_all):
            r["faithfulness"] = faith_eval.faithfulness(judge_llm, r["question"], r["rag"]["text"], r["_res"].passages[:best_k])
            if r["faithfulness"]["score"] is not None:
                scores.append(r["faithfulness"]["score"])
            if (i + 1) % 25 == 0:
                log("    checked %d/%d" % (i + 1, len(answered_all)))
        summary["faithfulness"] = {
            "n": len(scores), "mean": mean(scores) if scores else None,
            "fully_faithful_rate": mean([1.0 if s >= 0.999 else 0.0 for s in scores]) if scores else None,
            "n_claims": sum(len(r["faithfulness"]["claims"]) for r in answered_all),
            "answerable_mean": _mean_or_none([r["faithfulness"]["score"] for r in answered_all if r["answerable"]]),
            "out_of_collection_mean": _mean_or_none([r["faithfulness"]["score"] for r in answered_all if not r["answerable"]]),
        }

    # ----------------------------------------------------------------- abstention
    abst: Dict = {"signal": best_sig, "threshold_value": threshold, "primary_policy": primary_policy}
    for policy in ("llm", "threshold", "combined"):
        abst[policy] = abstention_metrics([r["policies"][policy] for r in unanswerable], [r["policies"][policy] for r in answerable])
    retrieved = [r for r in answerable if r["gold_rank"] is not None and r["gold_rank"] <= best_k]
    missed = [r for r in answerable if r not in retrieved]
    abst["over_refusal_when_gold_retrieved"] = mean([float(r["policies"][primary_policy]) for r in retrieved]) if retrieved else None
    abst["refusal_when_gold_missed"] = mean([float(r["policies"][primary_policy]) for r in missed]) if missed else None
    abst["n_gold_missed"] = len(missed)
    abst["closed_book_refusal_unanswerable"] = mean([float(r["closed_book"]["refused"]) for r in unanswerable]) if unanswerable else None
    abst["closed_book_refusal_answerable"] = mean([float(r["closed_book"]["refused"]) for r in answerable]) if answerable else None
    # every candidate signal on the held-out test questions: AUROC (threshold-free), and what
    # that signal's own dev-tuned threshold catches / wrongly refuses when applied unchanged
    abst["signal_test_auroc"] = {}
    abst["signal_test_at_dev_threshold"] = {}
    dev_signals = tuning.get("signals") or {}
    for sig in SIGNALS:
        a = [r["signals"].get(sig) for r in answerable]
        u = [r["signals"].get(sig) for r in unanswerable]
        if a and u and not any(v is None for v in a + u):
            abst["signal_test_auroc"][sig] = tune_threshold(a, u)["auroc"]
            dev_thr = (dev_signals.get(sig) or {}).get("threshold")
            if dev_thr is not None:
                abst["signal_test_at_dev_threshold"][sig] = rates_at_threshold(a, u, dev_thr)
    summary["abstention"] = abst

    # -------------------------------------------------------------------- latency
    summary["latency"] = {
        "rag_seconds_mean": mean([r["rag"]["seconds"] for r in test]),
        "closed_book_seconds_mean": mean([r["closed_book"]["seconds"] for r in test]),
        "rag_prompt_tokens_mean": mean([r["rag"]["prompt_tokens"] for r in test]),
        "rag_completion_tokens_mean": mean([r["rag"]["completion_tokens"] for r in test]),
        "retrieval_seconds_mean": mean([r["_res"].timings.get("total", 0.0) for r in test]),
    }

    # ------------------------------------------------------------------- failures
    for r in test:
        r["failure"] = categorise(r, best_k, has_labels)
    summary["failures"] = dict(Counter(r["failure"]["category"] for r in test))
    summary["failures_answerable"] = dict(Counter(r["failure"]["category"] for r in answerable))
    summary["failures_out_of_collection"] = dict(Counter(r["failure"]["category"] for r in unanswerable))

    write_outputs(out_dir, test, summary, col, best_k)
    return summary


# ----------------------------------------------------------------------- failures

CATEGORY_HELP = {
    "ok": "answered and consistent with the gold label / reference",
    "retrieval_miss": "the gold document was not among the top-k passages",
    "over_refusal": "gold document retrieved, but the system declined to answer",
    "wrong_conclusion": "gold document retrieved and cited material available, but the conclusion disagrees with the gold label",
    "no_clear_decision": "answered, but without a clear yes / no / maybe conclusion",
    "judged_worse": "answered, but the judge preferred the closed-book answer",
    "citation_problem": "right conclusion, but a citation is missing, invalid, or not supported by the cited passage",
    "unfaithful": "right conclusion, but at least one claim is not supported by the retrieved passages",
    "correct_abstention": "out-of-collection question, declined",
    "false_answer": "out-of-collection question, but the system answered from related passages",
}


def categorise(r: Dict, k: int, has_labels: bool) -> Dict:
    rag, final = r["rag"], r["final"]
    flags = []
    if not r["answerable"]:
        cat = "correct_abstention" if final["abstained"] else "false_answer"
        return {"category": cat, "flags": flags}
    got_gold = r["gold_rank"] is not None and r["gold_rank"] <= k
    if not got_gold and r.get("gold_doc") is not None:
        flags.append("retrieval_miss")
    if final["abstained"]:
        flags.append("abstained_%s" % final["reason"])
    if rag["invalid_citations"]:
        flags.append("invalid_citation")
    if not final["abstained"] and rag["n_uncited"]:
        flags.append("uncited_sentence")
    if r.get("citation") and r["citation"].get("recall") is not None and r["citation"]["recall"] < 1.0:
        flags.append("unsupported_citation")
    if r.get("faithfulness") and r["faithfulness"].get("score") is not None and r["faithfulness"]["score"] < 0.999:
        flags.append("unfaithful_claim")

    wrong = False
    if has_labels and r.get("label") and not final["abstained"]:
        wrong = final.get("decision") != r["label"]
    elif not has_labels and r.get("judge") and not final["abstained"]:
        wrong = r["judge"]["verdict"] == "y"

    if "retrieval_miss" in flags and (final["abstained"] or wrong):
        cat = "retrieval_miss"
    elif final["abstained"]:
        cat = "over_refusal"
    elif wrong and has_labels and final.get("decision") is None:
        cat = "no_clear_decision"
    elif wrong:
        cat = "wrong_conclusion" if has_labels else "judged_worse"
    elif "invalid_citation" in flags or "uncited_sentence" in flags or "unsupported_citation" in flags:
        cat = "citation_problem"
    elif "unfaithful_claim" in flags:
        cat = "unfaithful"
    else:
        cat = "ok"
    return {"category": cat, "flags": flags}


# ------------------------------------------------------------------------ outputs

def _public(r: Dict) -> Dict:
    return {k: v for k, v in r.items() if not k.startswith("_")}


def write_outputs(out_dir: Path, test: Sequence[Dict], summary: Dict, col, k: int) -> None:
    with open(out_dir / "records.jsonl", "w", encoding="utf-8") as fh:
        for r in test:
            rec = _public(r)
            rec["passages"] = [{"n": i, "doc_id": p.doc_id, "location": p.location(), "text": p.text}
                               for i, p in enumerate(r["_res"].passages[:k], start=1)]
            fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    with open(out_dir / "failures.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["qid", "answerable", "category", "flags", "gold_rank", "label", "rag_decision", "closed_book_decision",
                    "judge_verdict", "question", "rag_answer", "closed_book_answer", "reference"])
        for r in test:
            w.writerow([r["qid"], r["answerable"], r["failure"]["category"], ";".join(r["failure"]["flags"]), r["gold_rank"],
                        r.get("label"), r["final"].get("decision"), r["closed_book"].get("decision"),
                        {"x": "rag", "y": "closed_book", "tie": "tie"}.get((r.get("judge") or {}).get("verdict"), ""),
                        r["question"], r["final"]["text"], r["closed_book"]["text"], r.get("reference", "")])

    judged = [r for r in test if r.get("judge") and not r["judge"].get("identical")]
    rng = random.Random(7)
    sample = rng.sample(judged, min(50, len(judged)))
    # Blind sheet for the human validation of the judge: the key (which answer is which
    # system, and what the judge said) goes to a separate file.
    key: Dict[str, Dict[str, str]] = {}
    with open(out_dir / "judge_handcheck.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["qid", "question", "reference_answer", "answer_1", "answer_2", "human_verdict"])
        for r in sample:
            flip = rng.random() < 0.5
            a1, a2 = (r["closed_book"]["text"], answer_text_for_judging(r["final"]["text"])) if flip else \
                     (answer_text_for_judging(r["final"]["text"]), r["closed_book"]["text"])
            key[str(r["qid"])] = {"1": "closed_book" if flip else "rag", "2": "rag" if flip else "closed_book",
                                  "judge": {"x": "rag", "y": "closed_book", "tie": "tie"}[r["judge"]["verdict"]]}
            w.writerow([r["qid"], r["question"], r.get("reference", ""), a1, a2, ""])
    (out_dir / "judge_handcheck_key.json").write_text(json.dumps(key, indent=1))
