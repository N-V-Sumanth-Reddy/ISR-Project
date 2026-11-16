"""Retrieval evaluation: BM25 vs dense vs hybrid (RRF) vs hybrid + re-ranker, plus the
fusion ablations (RRF k, weighted fusion, DAT-style dynamic alpha) and the KG extension.

TREC-COVID (BEIR): nDCG@10 and Recall@100 with graded relevance judgments.
PubMedQA: known-item retrieval (find the abstract a question was written about).

BM25's k1 and b, the weighted-fusion alpha and the KG weight are tuned on the dev queries
only; the test queries are used once, for reporting.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import csv
import time
from typing import Dict, List, Optional

from ragsearch.collection import Collection
from ragsearch.eval import retrieval as R
from ragsearch.eval.judge import rate_passage
from ragsearch.eval.metrics import METRICS, bootstrap_ci, evaluate_run, mean, paired_randomization_test
from ragsearch.fusion import dat_alpha
from ragsearch.rerank import get_reranker
from ragsearch.retriever import Retriever
from ragsearch.runtime import Timer, base_parser, config_from_args, get_llms, load_json, results_dir, save_json

K1_GRID = [0.6, 0.9, 1.2, 1.6, 2.0]
B_GRID = [0.3, 0.4, 0.5, 0.75, 0.9]
ALPHA_GRID = [round(0.1 * i, 1) for i in range(11)]
RRF_K_GRID = [10, 30, 60, 100]
KG_WEIGHT_GRID = [0.25, 0.5, 1.0]


def short(model: str) -> str:
    return model.split("/")[-1]


def split_means(run, qrels, splits: Dict[str, List[str]], metrics) -> Dict:
    out = {}
    for split, qids in splits.items():
        per = evaluate_run(run, qrels, qids, metrics)
        out[split] = {}
        for m in metrics:
            vals = [per[m][q] for q in qids]
            lo, hi = bootstrap_ci(vals)
            out[split][m] = {"mean": mean(vals), "ci": [lo, hi], "n": len(vals)}
    return out


def evaluate_collection(name: str, cfg, metrics: List[str], primary_metric: str, llm=None) -> Dict:
    col = Collection.load(name, cfg)
    bench = load_json(col.dir / "benchmark.json")
    if name == "trec-covid":
        queries, qrels, dev, test = bench["queries"], bench["qrels"], bench["dev"], bench["test"]
    else:                                             # QA benchmark: the gold document is the only relevant one
        dev_q = [q for q in bench.get("dev", []) if q["answerable"]]
        test_q = [q for q in bench["test"] if q["answerable"]]
        queries = {q["qid"]: q["question"] for q in dev_q + test_q}
        qrels = {q["qid"]: {q["gold_doc"]: 1} for q in dev_q + test_q}
        dev, test = [q["qid"] for q in dev_q], [q["qid"] for q in test_q]
    splits = {"dev": dev, "test": test, "all": dev + test}
    splits = {k: v for k, v in splits.items() if v}
    tune_split = "dev" if dev else "test"
    dev_queries = {q: queries[q] for q in splits[tune_split]}
    models = col.dense_models
    primary = cfg.dense.model if cfg.dense.model in models else (models[0] if models else None)
    rrf_k, depth = cfg.fusion.rrf_k, cfg.fusion.depth
    print("[%s] %d passages, %d dev / %d test queries, dense models: %s" % (name, len(col.passages), len(dev), len(test), [short(m) for m in models]))

    def score(run_lists, qids, metric=primary_metric) -> float:
        per = evaluate_run(R.as_run(run_lists), qrels, qids, [metric])[metric]
        return mean([per[q] for q in qids])

    # ------------------------------------------------------------ BM25 tuning (dev only)
    grid = []
    for k1 in K1_GRID:
        for b in B_GRID:
            runs = R.bm25_doc_runs(col, dev_queries, k1, b, top_docs=20, passage_depth=200)
            grid.append({"k1": k1, "b": b, primary_metric: score(runs, splits[tune_split])})
    best = max(grid, key=lambda g: (g[primary_metric], -abs(g["k1"] - 0.9) - abs(g["b"] - 0.4)))
    k1, b = best["k1"], best["b"]
    print("  BM25 tuned on %s: k1=%.2f b=%.2f (%s %.4f; default %.4f)" % (
        tune_split, k1, b, primary_metric, best[primary_metric],
        next(g[primary_metric] for g in grid if g["k1"] == 0.9 and g["b"] == 0.4)))

    # the tuned parameters become the collection's defaults (used by the answer pipeline and the app)
    col.bm25.k1, col.bm25.b = k1, b
    col.bm25.save_params(str(col.dir))
    col.meta["bm25_tuned"] = {"k1": k1, "b": b, "tuned_on": tune_split, "metric": primary_metric}
    col._save_meta()

    # ------------------------------------------------------------------------- runs
    systems: List[Dict] = []

    def add(key: str, label: str, group: str, runs: Dict[str, R.DocList], **extra) -> None:
        systems.append(dict({"key": key, "label": label, "group": group, "_runs": runs}, **extra))

    bm25_default = R.bm25_doc_runs(col, queries, 0.9, 0.4)
    bm25 = R.bm25_doc_runs(col, queries, k1, b)
    add("bm25_default", "BM25 (k1=0.9, b=0.4)", "single", bm25_default)
    add("bm25", "BM25 (tuned: k1=%.1f, b=%.2f)" % (k1, b), "single", bm25)

    dense, hybrid = {}, {}
    for m in models:
        dense[m] = R.dense_doc_runs(col, queries, m)
        add("dense_" + short(m), "Dense (%s)" % short(m), "single", dense[m], model=m)
    for m in models:
        hybrid[m] = R.rrf_doc_runs(bm25, dense[m], rrf_k, depth)
        add("hybrid_" + short(m), "Hybrid RRF (BM25 + %s)" % short(m), "hybrid", hybrid[m], model=m)

    reranker = None
    try:
        reranker = get_reranker(cfg.rerank.model, cfg.dense.get("device", "auto"))
    except Exception as exc:
        print("  (cross-encoder unavailable, re-ranking rows skipped: %s)" % str(exc)[:120])
    if reranker is not None:
        top_n = cfg.rerank.top_n
        add("bm25_rerank", "BM25 + re-ranker", "rerank", R.rerank_doc_runs(col, queries, bm25, reranker, top_n))
        for m in models:
            add("hybrid_rerank_" + short(m), "Hybrid RRF (BM25 + %s) + re-ranker" % short(m), "rerank",
                R.rerank_doc_runs(col, queries, hybrid[m], reranker, top_n), model=m)

    extras: Dict = {"bm25_grid": grid, "bm25_tuned": {"k1": k1, "b": b}, "tuned_on": tune_split}
    if primary is not None:
        hyb, den = hybrid[primary], dense[primary]
        # weighted fusion: alpha tuned on dev
        curve = []
        for a in ALPHA_GRID:
            w = R.weighted_doc_runs(bm25, den, a)
            curve.append(dict({"alpha": a}, **{s: score(w, q) for s, q in splits.items()}))
        best_a = max(curve, key=lambda c: (c[tune_split], -abs(c["alpha"] - 0.5)))["alpha"]
        add("weighted", "Weighted fusion (alpha=%.1f, tuned)" % best_a, "fusion", R.weighted_doc_runs(bm25, den, best_a))
        extras["alpha_curve"], extras["alpha_tuned"] = curve, best_a
        # RRF k sensitivity
        extras["rrf_k"] = [dict({"k": kk}, **{s: score(R.rrf_doc_runs(bm25, den, kk, depth), q) for s, q in splits.items()})
                           for kk in RRF_K_GRID]
        # KG extension: weight tuned on dev
        if col.kg is not None:
            kg_grid = [dict({"weight": w}, **{s: score(R.kg_doc_runs(col, queries, hyb, rrf_k, depth, kg_weight=w), q)
                                              for s, q in splits.items()}) for w in KG_WEIGHT_GRID]
            best_w = max(kg_grid, key=lambda c: c[tune_split])["weight"]
            add("hybrid_kg", "Hybrid RRF + entity graph (w=%.2f)" % best_w, "kg",
                R.kg_doc_runs(col, queries, hyb, rrf_k, depth, kg_weight=best_w))
            extras["kg_grid"], extras["kg_weight"], extras["kg_stats"] = kg_grid, best_w, col.kg.stats()
        # DAT-style dynamic alpha: the local LLM rates each retriever's top-1 result
        if llm is not None:
            alphas, ratings = {}, {}
            for qid, text in queries.items():
                b_top, d_top = bm25[qid][:1], den[qid][:1]
                sb = rate_passage(llm, text, col.passages[b_top[0][2]].index_text) if b_top else 0
                sd = sb if (b_top and d_top and b_top[0][0] == d_top[0][0]) else \
                    (rate_passage(llm, text, col.passages[d_top[0][2]].index_text) if d_top else 0)
                alphas[qid], ratings[qid] = dat_alpha(sd, sb), {"bm25": sb, "dense": sd}
            add("dat", "Dynamic alpha (DAT-style, %s)" % llm.model, "fusion", R.weighted_doc_runs(bm25, den, alphas))
            extras["dat"] = {"alphas": alphas, "ratings": ratings, "mean_alpha": mean(list(alphas.values())), "llm": llm.model}

    # ---------------------------------------------------------------------- metrics
    per_query: Dict[str, Dict[str, float]] = {}
    for s in systems:
        run = R.as_run(s["_runs"])
        s["metrics"] = split_means(run, qrels, splits, metrics)
        per_query[s["key"]] = evaluate_run(run, qrels, splits["all"], [primary_metric])[primary_metric]

    def paired(a: str, b_: str) -> Optional[Dict]:
        if a not in per_query or b_ not in per_query:
            return None
        qids = splits["test"]
        va, vb = [per_query[a][q] for q in qids], [per_query[b_][q] for q in qids]
        return {"a": a, "b": b_, "metric": primary_metric, "mean_a": mean(va), "mean_b": mean(vb), "diff": mean(va) - mean(vb),
                "p": paired_randomization_test(va, vb), "wins": sum(x > y + 1e-12 for x, y in zip(va, vb)),
                "losses": sum(y > x + 1e-12 for x, y in zip(va, vb)), "n": len(qids)}

    tests = []
    if primary is not None:
        h, d = "hybrid_" + short(primary), "dense_" + short(primary)
        pairs = [(h, "bm25"), (h, d), ("hybrid_rerank_" + short(primary), h), ("bm25_rerank", "bm25"),
                 ("hybrid_kg", h), ("weighted", h), ("dat", h), ("bm25", "bm25_default")]
        for m in models:
            if m != primary:
                pairs += [("hybrid_" + short(m), "bm25"), ("hybrid_" + short(m), "dense_" + short(m)),
                          ("hybrid_rerank_" + short(m), "hybrid_" + short(m))]
        tests = [t for t in (paired(a, b_) for a, b_ in pairs) if t]
        qids = splits["test"]
        bm, de, hy = (mean([per_query[k][q] for q in qids]) for k in ("bm25", d, h))
        extras["hypothesis"] = {
            "statement": "hybrid matches or beats the better single retriever on %s (test queries)" % primary_metric,
            "bm25": bm, "dense": de, "hybrid": hy, "better_single": max(bm, de), "holds": bool(hy >= max(bm, de) - 1e-9),
            "oracle_best_of_two": mean([max(per_query["bm25"][q], per_query[d][q]) for q in qids]),
            "overlap_at_10_bm25_dense": R.overlap_at_k({q: bm25[q] for q in qids}, {q: dense[primary][q] for q in qids}, 10),
            "per_model": {short(m): {"dense": mean([per_query["dense_" + short(m)][q] for q in qids]),
                                     "hybrid": mean([per_query["hybrid_" + short(m)][q] for q in qids])} for m in models},
        }

    # ---------------------------------------------------------------------- latency
    latency = {}
    if primary is not None:
        retriever = Retriever(col, cfg)
        sample = [queries[q] for q in splits["test"][:20]]
        retriever.search(sample[0], mode="hybrid", top_k=10)                     # warm-up
        configs = {"bm25": dict(mode="bm25"), "dense": dict(mode="dense"), "hybrid": dict(mode="hybrid"),
                   "hybrid + KG": dict(mode="hybrid", use_kg=True)}
        if reranker is not None:
            configs["hybrid + re-ranker"] = dict(mode="hybrid", rerank=True)
        for label, kw in configs.items():
            t0 = time.time()
            for text in sample:
                retriever.search(text, top_k=10, k1=k1, b=b, **kw)
            latency[label] = 1000.0 * (time.time() - t0) / len(sample)

    out_dir = results_dir(cfg, "retrieval", name)
    with open(out_dir / "per_query.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        keys = [s["key"] for s in systems]
        w.writerow(["qid", "split", "query"] + keys)
        for q in splits["all"]:
            w.writerow([q, "dev" if q in dev else "test", queries[q]] + ["%.4f" % per_query[k][q] for k in keys])
    run_dir = out_dir / "runs"
    run_dir.mkdir(exist_ok=True)
    for s in systems:
        save_json(run_dir / (s["key"] + ".json"), {q: [d for d, _, _ in lst[:100]] for q, lst in s["_runs"].items()})
    result = {
        "collection": {"name": name, **col.meta}, "primary_metric": primary_metric, "metrics": metrics,
        "primary_dense_model": primary, "n_dev": len(dev), "n_test": len(test),
        "systems": [{k: v for k, v in s.items() if not k.startswith("_")} for s in systems],
        "significance": tests, "latency_ms_per_query": latency, **extras,
    }
    save_json(out_dir / "metrics.json", result)
    print("  %-52s %s" % ("system", "  ".join("%s(test)" % m for m in metrics)))
    for s in systems:
        print("  %-52s %s" % (s["label"][:52], "  ".join("%12.4f" % s["metrics"]["test"][m]["mean"] for m in metrics)))
    return result


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--only", default="trec-covid,pubmedqa")
    p.add_argument("--dat", action="store_true", help="also run the DAT-style dynamic alpha (needs the local LLM)")
    args = p.parse_args()
    cfg = config_from_args(args)
    llm = None
    if args.dat:
        gen, _ = get_llms(cfg)
        if gen.is_up():
            llm = gen
        else:
            print("local LLM not reachable: skipping the DAT-style fusion row")
    for name in [n.strip() for n in args.only.split(",") if n.strip()]:
        if not Collection.exists(name, cfg):
            print("collection %s not built, skipping" % name)
            continue
        with Timer("retrieval evaluation: " + name):
            if name == "trec-covid":
                evaluate_collection(name, cfg, ["ndcg@10", "recall@100", "capped_recall@100", "p@10"], "ndcg@10", llm)
            else:
                evaluate_collection(name, cfg, ["recall@1", "recall@5", "recall@10", "mrr@10"], "mrr@10", llm if name == "trec-covid" else None)


if __name__ == "__main__":
    main()
