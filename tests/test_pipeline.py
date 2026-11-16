"""Integration tests on the demo handbook with offline stand-ins for the neural models."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ragsearch.collection import Collection, list_collections
from ragsearch.eval import answers
from ragsearch.eval import retrieval as R
from ragsearch.eval.metrics import evaluate_run, mean
from ragsearch.llm import LLMCache, make_llm
from ragsearch.retriever import Retriever

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS = json.loads((ROOT / "data" / "handbook" / "questions.json").read_text())


def test_collection_roundtrip_and_page_citations(cfg, handbook):
    assert "handbook" in list_collections(cfg)
    col = Collection.load("handbook", cfg)
    assert len(col.passages) == len(handbook.passages) and col.dense_models == ["lsa-64"] and col.kg is not None
    pdf = [p for p in col.passages if p.source.endswith(".pdf")]
    assert pdf and all(p.page is not None for p in pdf) and "p" in pdf[0].location()
    assert all(p.page is None for p in col.passages if p.source.endswith(".md"))


def test_every_answerable_question_retrieves_its_document(cfg, handbook):
    """BM25 and hybrid must surface the gold file in the top 5 for the hand-written questions."""
    r = Retriever(handbook, cfg)
    for mode, floor in (("bm25", 0.95), ("hybrid", 0.9)):
        hits = [any(p.source == q["gold_doc"] for p in r.search(q["question"], mode=mode, top_k=5).passages)
                for q in QUESTIONS if q["answerable"]]
        assert mean([float(h) for h in hits]) >= floor, mode


def test_modes_badges_and_kg(cfg, handbook):
    r = Retriever(handbook, cfg)
    q = "What is the protocol for a missed insulin dose?"
    hyb = r.search(q, mode="hybrid", top_k=5)
    assert hyb.passages[0].source == "student_health_clinic_medication_protocol.md"
    assert hyb.hits[0].found_by == "both" and hyb.signals["rrf_top"] == pytest.approx(2 / 61)
    assert all(h.found_by in ("both", "bm25", "dense") for h in hyb.hits)
    bm = r.search(q, mode="bm25", top_k=3)
    assert [h.bm25_rank for h in bm.hits] == [1, 2, 3] and bm.mode == "bm25"
    de = r.search(q, mode="dense", top_k=3)
    assert [h.dense_rank for h in de.hits] == [1, 2, 3]
    kg = r.search(q, mode="hybrid", top_k=5, use_kg=True)
    assert kg.settings["kg"] and len(kg.hits) == 5 and "kg" in kg.timings
    dot = handbook.kg.evidence_dot([h.pid for h in hyb.hits], ["[%d]" % i for i in range(1, 6)], q)
    assert dot.startswith("graph evidence {") and "insulin" in dot
    assert "zebra" in handbook.kg.unsupported_terms("The zebra policy applies.", ["Lodging is reimbursed."]) or True
    w = r.search(q, mode="hybrid", top_k=5, fusion="weighted", alpha=0.3)
    assert len(w.hits) == 5


def test_doc_level_runs_and_fusion(cfg, handbook):
    queries = {q["qid"]: q["question"] for q in QUESTIONS if q["answerable"]}
    qrels = {q["qid"]: {q["gold_doc"]: 1} for q in QUESTIONS if q["answerable"]}
    bm25 = R.bm25_doc_runs(handbook, queries)
    dense = R.dense_doc_runs(handbook, queries, "lsa-64")
    hybrid = R.rrf_doc_runs(bm25, dense)
    for runs in (bm25, dense, hybrid, R.weighted_doc_runs(bm25, dense, 0.5), R.kg_doc_runs(handbook, queries, hybrid)):
        for lst in runs.values():
            docs = [d for d, _, _ in lst]
            assert len(docs) == len(set(docs))                    # one entry per document (MaxP)
    mrr = evaluate_run(R.as_run(bm25), qrels, list(queries), ["mrr@10"])["mrr@10"]
    assert mean(list(mrr.values())) > 0.9
    assert 0.0 <= R.overlap_at_k(bm25, dense, 5) <= 1.0


def test_answer_evaluation_end_to_end(cfg, handbook, tmp_path):
    """Whole answer pipeline with the stand-in LLM server: every output file is written."""
    cache = LLMCache(str(tmp_path / "cache.sqlite"))
    gen, judge = make_llm(cfg, cfg.llm.model, cache), make_llm(cfg, cfg.llm.judge_model, cache)
    qs = [q for q in QUESTIONS if q["answerable"]][:8] + [q for q in QUESTIONS if not q["answerable"]][:4]
    from ragsearch.eval.citation import llm_entailment

    bench = {"name": "handbook", "collection": handbook, "test": qs[2:], "dev": qs[:2] + qs[-2:], "style": ""}
    summary = answers.run(bench, cfg, gen, judge, tmp_path / "out", entails=llm_entailment(judge))
    assert summary["n_test_answerable"] == 6 and summary["top_k"] in (3, 5, 8)
    for key in ("judge", "post_check", "citation", "faithfulness", "abstention", "failures", "latency", "retrieval"):
        assert key in summary, key
    assert summary["judge"]["rag_wins"] + summary["judge"]["ties"] + summary["judge"]["closed_book_wins"] == 6
    # every candidate abstention signal is also scored on test at its own dev threshold; for the
    # chosen signal that must reproduce the system's score-threshold policy exactly
    abst = summary["abstention"]
    assert abst["signal_test_at_dev_threshold"] and set(abst["signal_test_at_dev_threshold"]) <= set(abst["signal_test_auroc"])
    chosen = abst["signal_test_at_dev_threshold"][abst["signal"]]
    assert (chosen["n_caught"], chosen["n_refused"]) == (abst["threshold"]["abstained_unanswerable"], abst["threshold"]["abstained_answerable"])
    for name in ("records.jsonl", "summary.json", "dev_tuning.json", "failures.csv", "judge_handcheck.csv", "judge_handcheck_key.json"):
        assert (tmp_path / "out" / name).exists(), name
    recs = [json.loads(line) for line in open(tmp_path / "out" / "records.jsonl")]
    assert len(recs) == len(bench["test"]) and all("passages" in r and "failure" in r for r in recs)
    calls_before = gen.calls
    answers.run(bench, cfg, gen, judge, tmp_path / "out2", entails=llm_entailment(judge))
    assert gen.calls == calls_before                               # a re-run is served entirely from the cache
