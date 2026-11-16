from __future__ import annotations

import math
import random
from collections import Counter

import numpy as np
import pytest

from ragsearch.bm25 import BM25Index, top_n_sorted
from ragsearch.chunking import Document, chunk_documents, chunk_units
from ragsearch.dense import DenseIndex
from ragsearch.eval import metrics as M
from ragsearch.eval.abstention import abstention_metrics, apply_policy, tune_threshold
from ragsearch.fusion import dat_alpha, rrf, weighted_fusion
from ragsearch.textproc import (Tokenizer, extract_citations, extract_numbers, first_decision, split_answer_sentences,
                                split_sentences, strip_citations)

WORDS = ("virus vaccine trial patient dose protein spike antibody mask policy leave travel student data lab risk "
         "outcome treatment control infection hospital mortality symptom fever cough immune response cell gene").split()


def random_corpus(n=300, seed=0):
    rng = random.Random(seed)
    return [" ".join(rng.choice(WORDS) for _ in range(rng.randint(5, 60))) for _ in range(n)]


# ------------------------------------------------------------------------------ text

def test_citation_parsing():
    assert extract_citations("A [1]. B [2][3]. C [1, 4] and [5-7].") == [1, 2, 3, 4, 5, 6, 7]
    assert strip_citations("Rest for 3 days [1][2].") == "Rest for 3 days."
    sents = split_answer_sentences("Yes, the dose is 5 mg [1]. It is given daily. [2][3] No more than 10 mg (see [9]).")
    assert [c for _, c in sents] == [[1], [2, 3], [9]]
    assert sents[0][0] == "Yes, the dose is 5 mg."


def test_sentences_numbers_decision():
    s = split_sentences("Smith et al. found a 3.5 mg dose, e.g. in adults. It worked! Did it last?")
    assert len(s) == 3 and s[0].startswith("Smith et al. found")
    assert extract_numbers("Paid 3,500 dollars for 12 items at 4.5% on day 3.") == ["3500", "12", "4.5", "3"]
    assert first_decision("Yes, it does [1].") == "yes"
    assert first_decision("**No**. It did not.") == "no"
    assert first_decision("Maybe; the evidence is mixed.") == "maybe"
    assert first_decision("The study shows improvement.") is None


def test_tokenizer_stems_and_stops():
    tok = Tokenizer()
    assert tok("The vaccines were tested in hospitals") == ["vaccin", "were", "test", "hospit"]


# -------------------------------------------------------------------------- chunking

def test_chunks_respect_size_and_overlap():
    sents = [("Sentence number %d has exactly seven words." % i, 1 + i // 20) for i in range(60)]
    chunks = chunk_units(sents, chunk_words=50, overlap_words=14)
    assert all(len(c[0].split()) <= 50 for c in chunks)
    for a, b in zip(chunks, chunks[1:]):                      # consecutive chunks share the overlap sentences
        assert a[0].split(". ")[-1].rstrip(".") in b[0]
    assert chunks[0][1] == 1 and chunks[-1][2] == 3           # first / last page tracked
    joined = " ".join(c[0] for c in chunks)
    assert all(("number %d " % i) in joined for i in range(60))


def test_long_sentence_and_title_only_documents():
    long = " ".join("w%d" % i for i in range(500))
    passages = chunk_documents([Document("d1", "T", "src", [(None, long)]), Document("d2", "Only a title", "src", [(None, "")])], 100, 20)
    assert all(len(p.text.split()) <= 100 for p in passages)
    assert passages[-1].doc_id == "d2" and passages[-1].text == "Only a title"
    assert [p.pid for p in passages] == list(range(len(passages)))


# ------------------------------------------------------------------------------ BM25

def brute_bm25(corpus, query, k1, b, tok):
    docs = [tok(d) for d in corpus]
    n, avg = len(docs), sum(len(d) for d in docs) / len(docs)
    df = Counter(t for d in docs for t in set(d))
    scores = []
    for d in docs:
        tf = Counter(d)
        s = 0.0
        for t, qtf in Counter(tok(query)).items():
            if tf[t]:
                idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                s += qtf * idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * len(d) / avg))
        scores.append(s)
    return np.array(scores)


@pytest.mark.parametrize("k1,b", [(0.9, 0.4), (1.2, 0.75), (2.0, 0.3)])
def test_bm25_matches_reference_formula(k1, b):
    corpus = random_corpus()
    idx = BM25Index(k1=0.9, b=0.4).build(corpus)
    for q in ("vaccine trial dose", "travel policy for student leave", "immune immune response"):
        want = brute_bm25(corpus, q, k1, b, idx.tokenizer)
        got = idx.score(q, k1=k1, b=b)
        assert np.allclose(got, want, rtol=1e-4, atol=1e-5)
        ids, scores = idx.search(q, top_n=10, k1=k1, b=b)
        assert list(ids) == list(np.lexsort((np.arange(len(want)), -want))[:10]) or np.allclose(scores, np.sort(want)[::-1][:10], rtol=1e-4)


def test_bm25_save_load_and_rank_bm25_agreement(tmp_path):
    # a Zipf-like vocabulary, so query terms are rare as in real text
    rng = random.Random(3)
    vocab = ["term%d" % i for i in range(1500)]
    weights = [1.0 / (i + 1) for i in range(len(vocab))]
    corpus = [" ".join(rng.choices(vocab, weights, k=rng.randint(20, 120))) for _ in range(400)]
    idx = BM25Index().build(corpus)
    idx.save(str(tmp_path))
    again = BM25Index.load(str(tmp_path))
    q = "term40 term75 term120 term300"
    assert np.allclose(idx.score(q), again.score(q))
    rank_bm25 = pytest.importorskip("rank_bm25")
    ref = rank_bm25.BM25Okapi([idx.tokenizer(d) for d in corpus], k1=idx.k1, b=idx.b)
    top_ref = [int(i) for i in np.argsort(-ref.get_scores(idx.tokenizer(q)))[:10]]
    ours = [int(i) for i in idx.search(q, 10)[0]]
    assert ours[0] == top_ref[0] and len(set(top_ref) & set(ours)) >= 8     # idf variants differ slightly


def test_top_n_is_deterministic_on_ties():
    scores = np.array([1.0, 3.0, 3.0, 2.0, 3.0, 0.0], dtype=np.float32)
    ids, vals = top_n_sorted(scores, 2)
    assert list(ids) == [1, 2] and list(vals) == [3.0, 3.0]


# ----------------------------------------------------------------------------- dense

def test_dense_backends_agree():
    rng = np.random.RandomState(0)
    emb = rng.randn(500, 32).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    q = emb[:5] + 0.01 * rng.randn(5, 32).astype(np.float32)
    ids_np, sc_np = DenseIndex(emb, backend="numpy").search(q, 7)
    assert list(ids_np[:, 0]) == [0, 1, 2, 3, 4]
    pytest.importorskip("faiss")
    ids_f, sc_f = DenseIndex(emb, backend="faiss").search(q, 7)
    assert (ids_f == ids_np).all() and np.allclose(sc_f, sc_np, atol=1e-5)


# ---------------------------------------------------------------------------- fusion

def test_rrf_values_and_order():
    fused = dict(rrf({"a": ["x", "y", "z"], "b": ["y", "x", "w"]}, k=60))
    assert fused["x"] == pytest.approx(1 / 61 + 1 / 62) and fused["y"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["z"] == pytest.approx(1 / 63) and fused["w"] == pytest.approx(1 / 63)
    order = [d for d, _ in rrf({"a": ["x", "y", "z"], "b": ["y", "x", "w"]}, k=60)]
    assert order[:2] == ["x", "y"] and set(order[2:]) == {"z", "w"}
    assert [d for d, _ in rrf({"a": ["p", "q"], "b": ["q", "p"]}, depth=1)] == ["p", "q"]


def test_weighted_and_dynamic_alpha():
    dense, sparse = {"x": 0.9, "y": 0.5, "z": 0.1}, {"y": 12.0, "z": 3.0, "w": 6.0}
    assert weighted_fusion(dense, sparse, alpha=1.0)[0][0] == "x"
    assert weighted_fusion(dense, sparse, alpha=0.0)[0][0] == "y"
    assert dat_alpha(0, 0) == 0.5 and dat_alpha(5, 3) == 1.0 and dat_alpha(2, 5) == 0.0 and dat_alpha(3, 1) == 0.75


# --------------------------------------------------------------------------- metrics

def test_metrics_match_trec_eval():
    pytrec_eval = pytest.importorskip("pytrec_eval")
    rng = random.Random(1)
    docs = ["d%d" % i for i in range(300)]
    qrels = {"q%d" % q: {d: rng.choice([0, 0, 1, 2]) for d in rng.sample(docs, 80)} for q in range(12)}
    run = {q: rng.sample(docs, 150) for q in qrels}
    ev = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut.10", "recall.100", "recip_rank"})
    ref = ev.evaluate({q: {d: float(len(r) - i) for i, d in enumerate(r)} for q, r in run.items()})
    for q in qrels:
        assert M.ndcg_at_k(run[q], qrels[q], 10) == pytest.approx(ref[q]["ndcg_cut_10"], abs=1e-6)
        assert M.recall_at_k(run[q], qrels[q], 100) == pytest.approx(ref[q]["recall_100"], abs=1e-6)
        assert M.mrr_at_k(run[q], qrels[q], 150) == pytest.approx(ref[q]["recip_rank"], abs=1e-6)


def test_small_sample_statistics():
    assert M.capped_recall_at_k(["a", "b"], {"a": 1, "b": 1, "c": 1}, k=2) == 1.0
    assert M.sign_test(10, 0) == pytest.approx(2 / 1024) and M.sign_test(5, 5) == 1.0
    lo, hi = M.bootstrap_ci([0.2, 0.4, 0.6, 0.8] * 10)
    assert lo < 0.5 < hi
    assert M.paired_randomization_test([1.0] * 12, [0.0] * 12) < 0.01
    assert M.paired_randomization_test([0.5, 0.4], [0.5, 0.4]) == 1.0
    assert M.auroc([0.9, 0.8], [0.1, 0.2]) == 1.0 and M.auroc([0.5], [0.5]) == 0.5
    assert M.cohen_kappa(["a", "b", "a"], ["a", "b", "a"]) == 1.0
    lo, hi = M.wilson_interval(8, 10)
    assert 0.44 < lo < 0.5 and 0.94 < hi < 0.98


# ------------------------------------------------------------------------ abstention

def test_threshold_tuning_and_policies():
    t = tune_threshold(answerable=[0.8, 0.7, 0.75, 0.9, 0.6], unanswerable=[0.2, 0.3, 0.65])
    assert 0.3 < t["threshold"] <= 0.6 and t["tpr"] == pytest.approx(2 / 3) and t["fpr"] == 0.0
    assert t["auroc"] == pytest.approx(14 / 15)
    assert apply_policy(False, 0.2, 0.5, "combined") and not apply_policy(False, 0.7, 0.5, "combined")
    assert apply_policy(True, 0.7, 0.5, "combined") and not apply_policy(True, 0.7, 0.5, "threshold")
    m = abstention_metrics([True, True, False], [False, False, False, True])
    assert m["abstention_accuracy"] == pytest.approx(2 / 3) and m["over_refusal_rate"] == 0.25
