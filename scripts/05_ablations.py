"""Ablations that need their own indexes or extra generation runs.

  chunk size      PubMedQA known-item retrieval with 50 / 100 / 200 / 400-word passages
                  (and, in the "full" profile, TREC-COVID with 100 / 200 / 400 words)
  retrieval mode  (full profile) end-to-end RAG answers with BM25 / dense / hybrid /
                  hybrid + re-ranker / hybrid + entity graph

top-k, fusion method and re-ranking at the retrieval level are covered by
03_eval_retrieval.py and 04_eval_answers.py.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

from ragsearch.collection import Collection
from ragsearch.config import resolve
from ragsearch.datasets import ensure_trec_covid, load_beir, load_pubmedqa, pubmedqa_documents
from ragsearch.eval import answers
from ragsearch.eval import retrieval as R
from ragsearch.eval.judge import decision_of
from ragsearch.eval.metrics import evaluate_run, mean
from ragsearch.generate import YESNO_STYLE
from ragsearch.rerank import get_reranker
from ragsearch.retriever import Retriever
from ragsearch.runtime import Timer, base_parser, config_from_args, get_llms, load_json, results_dir, save_json


def _mean_metric(run, qrels, qids, metric) -> float:
    per = evaluate_run(run, qrels, qids, [metric])[metric]
    return mean([per[q] for q in qids])


def _retrieval_rows(col, queries, qrels, qids, model, metrics, k1=None, b=None, rrf_k=60, depth=100):
    bm25 = R.bm25_doc_runs(col, queries, k1, b)
    dense = R.dense_doc_runs(col, queries, model)
    hybrid = R.rrf_doc_runs(bm25, dense, rrf_k, depth)
    return {name: {m: _mean_metric(R.as_run(runs), qrels, qids, m) for m in metrics}
            for name, runs in (("bm25", bm25), ("dense", dense), ("hybrid", hybrid))}


def chunk_ablation_pubmedqa(cfg) -> None:
    main = Collection.load("pubmedqa", cfg, dense_models=[], load_kg=False)
    bench = load_json(main.dir / "benchmark.json")
    records, _ = load_pubmedqa(resolve(cfg, "data_dir"))
    held_out = set(bench["held_out_documents"])
    docs = pubmedqa_documents(records, sorted(k for k in records if k not in held_out))
    test = [q for q in bench["test"] if q["answerable"]]
    queries = {q["qid"]: q["question"] for q in test}
    qrels = {q["qid"]: {q["gold_doc"]: 1} for q in test}
    model = cfg.dense.model
    rows = []
    for cw in (50, 100, 200, 400):
        col = Collection.build("pubmedqa_cw%d" % cw, docs, cfg, index_dir=resolve(cfg, "index_dir") / "_ablation" / ("pubmedqa_cw%d" % cw),
                               dense_models=[model], build_kg=False, chunk_words=cw, overlap_words=cw // 5, verbose=False)
        res = _retrieval_rows(col, queries, qrels, list(queries), model, ["recall@1", "recall@5", "mrr@10"])
        rows.append({"chunk_words": cw, "overlap_words": cw // 5, "n_passages": len(col.passages),
                     "mean_passage_words": mean([len(p.text.split()) for p in col.passages]), "systems": res})
        print("  %3d words: %5d passages | MRR@10 bm25 %.3f dense %.3f hybrid %.3f" % (
            cw, len(col.passages), res["bm25"]["mrr@10"], res["dense"]["mrr@10"], res["hybrid"]["mrr@10"]))
    save_json(results_dir(cfg, "ablations") / "chunk_size_pubmedqa.json",
              {"dataset": "pubmedqa", "n_queries": len(queries), "dense_model": model, "rows": rows})


def chunk_ablation_trec(cfg) -> None:
    main = Collection.load("trec-covid", cfg, dense_models=[cfg.dense.model], load_kg=False)
    bench = load_json(main.dir / "benchmark.json")
    queries, qrels, test = bench["queries"], bench["qrels"], bench["test"]
    model = cfg.dense.model
    tuned = main.meta.get("bm25_tuned", {})
    k1, b = tuned.get("k1"), tuned.get("b")
    docs = None
    rows = []
    for cw in (100, 200, 400):
        if cw == main.meta.get("chunk_words"):
            col = main
        else:
            if docs is None:
                docs, _, _ = load_beir(ensure_trec_covid(resolve(cfg, "data_dir")))
                keep = {p.doc_id for p in main.passages}
                docs = [d for d in docs if d.doc_id in keep]
            col = Collection.build("trec-covid_cw%d" % cw, docs, cfg, index_dir=resolve(cfg, "index_dir") / "_ablation" / ("trec-covid_cw%d" % cw),
                                   dense_models=[model], build_kg=False, chunk_words=cw, overlap_words=cw // 5)
        res = _retrieval_rows(col, queries, qrels, test, model, ["ndcg@10", "recall@100"], k1, b)
        rows.append({"chunk_words": cw, "overlap_words": cw // 5, "n_passages": len(col.passages), "systems": res})
        print("  %3d words: %6d passages | nDCG@10 bm25 %.3f dense %.3f hybrid %.3f" % (
            cw, len(col.passages), res["bm25"]["ndcg@10"], res["dense"]["ndcg@10"], res["hybrid"]["ndcg@10"]))
    save_json(results_dir(cfg, "ablations") / "chunk_size_trec_covid.json",
              {"dataset": "trec-covid", "n_queries": len(test), "dense_model": model, "bm25": {"k1": k1, "b": b}, "rows": rows})


def answer_mode_ablation(cfg) -> None:
    gen, judge = get_llms(cfg)
    if not gen.is_up():
        print("local LLM not reachable: skipping the answer-level retrieval-mode ablation")
        return
    col = Collection.load("pubmedqa", cfg)
    bench = load_json(col.dir / "benchmark.json")
    tuning = load_json(results_dir(cfg, "answers", "pubmedqa") / "dev_tuning.json", {})
    k = int(tuning.get("top_k", {}).get("best", cfg.retrieval.top_k))
    retriever = Retriever(col, cfg)
    try:
        get_reranker(cfg.rerank.model, cfg.dense.get("device", "auto"))
        has_ce = True
    except Exception:
        has_ce = False
    modes = {"BM25": dict(mode="bm25"), "Dense": dict(mode="dense"), "Hybrid (RRF)": dict(mode="hybrid")}
    if has_ce:
        modes["Hybrid + re-ranker"] = dict(mode="hybrid", rerank=True)
    if col.kg is not None:
        modes["Hybrid + entity graph"] = dict(mode="hybrid", kg=True)
    rows = []
    for label, settings in modes.items():
        recs = []
        for q in bench["test"]:
            res = answers.retrieve(retriever, q["question"], settings, None)
            rec = dict(q)
            rec["signals"] = res.signals
            rec["gold_rank"] = answers.gold_rank(res, q.get("gold_doc"))
            rec["rag"] = answers.rag_record(q, res, k, gen, YESNO_STYLE)
            rec["rag"]["decision"] = None if rec["rag"]["abstained_llm"] else decision_of(rec["rag"]["text"], q["question"], judge)
            recs.append(rec)
        ans = [r for r in recs if r["answerable"]]
        una = [r for r in recs if not r["answerable"]]
        u = answers.utility(recs, "llm", None, None)
        row = {"mode": label, "top_k": k,
               "gold_in_top_k": mean([1.0 if (r["gold_rank"] is not None and r["gold_rank"] <= k) else 0.0 for r in ans]),
               "decision_accuracy": mean([float(r["rag"]["decision"] == r["label"]) for r in ans]),
               "coverage": mean([0.0 if r["rag"]["abstained_llm"] else 1.0 for r in ans]),
               "abstention_accuracy": mean([float(r["rag"]["abstained_llm"]) for r in una]) if una else None,
               "utility": u["utility"], "n_answerable": len(ans), "n_out_of_collection": len(una)}
        rows.append(row)
        print("  %-22s gold@k %.3f | accuracy %.3f | coverage %.3f | declined %.3f" % (
            label, row["gold_in_top_k"], row["decision_accuracy"], row["coverage"], row["abstention_accuracy"] or 0.0))
    save_json(results_dir(cfg, "ablations") / "answer_modes_pubmedqa.json",
              {"dataset": "pubmedqa", "generator": gen.model, "policy": "LLM abstention only", "rows": rows})


def main() -> None:
    p = base_parser(__doc__)
    args = p.parse_args()
    cfg = config_from_args(args)
    if Collection.exists("pubmedqa", cfg):
        with Timer("ablation: chunk size (PubMedQA)"):
            chunk_ablation_pubmedqa(cfg)
    if cfg.eval.get("trec_chunk_ablation") and Collection.exists("trec-covid", cfg):
        with Timer("ablation: chunk size (TREC-COVID)"):
            chunk_ablation_trec(cfg)
    if cfg.eval.get("answer_mode_ablation") and Collection.exists("pubmedqa", cfg):
        with Timer("ablation: retrieval mode, end to end (PubMedQA)"):
            answer_mode_ablation(cfg)


if __name__ == "__main__":
    main()
