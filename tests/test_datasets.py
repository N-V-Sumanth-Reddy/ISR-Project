"""PubMedQA corpus definition: what is searchable, and that no answer leaks into the index."""
from __future__ import annotations

import pytest

from ragsearch.chunking import chunk_documents
from ragsearch.datasets import pubmedqa_documents, pubmedqa_questions, pubmedqa_splits
from ragsearch.eval.abstention import rates_at_threshold


def _records(n: int = 200):
    """Synthetic PubMedQA records (same fields as ori_pqal.json). Every field that must stay out
    of the index carries a unique marker, so a leak shows up as a substring match."""
    recs = {}
    for i in range(n):
        pmid = str(10_000_000 + i)
        recs[pmid] = {
            "QUESTION": "Does treatment QTITLE%d improve outcome?" % i,
            "CONTEXTS": ["Background sentence about cohort %d." % i, "Methods: we enrolled %d patients." % (i + 20),
                         "Results showed a difference in group %d." % i],
            "LABELS": ["BACKGROUND", "METHODS", "RESULTS"],
            "MESHES": ["Humans"],
            "YEAR": "2010",
            "final_decision": "yes",
            "LONG_ANSWER": "REFANSWER%d: the treatment improves outcome." % i,
        }
    return recs


def _splits(recs, test_ids):
    return pubmedqa_splits(recs, test_ids, n_test_ans=40, n_test_unans=15, n_dev_ans=20, n_dev_unans=10, seed=13)


def test_pubmedqa_corpus_is_the_non_held_out_abstracts():
    recs = _records()
    test_ids = sorted(recs)[:100]                 # stands in for the official test half
    s = _splits(recs, test_ids)
    held_out = set(s["test_unanswerable"]) | set(s["dev_unanswerable"])
    collection = set(s["collection"])

    assert collection == set(recs) - held_out     # corpus = every abstract except the withheld ones
    assert not (collection & held_out)
    assert set(s["test_answerable"]) <= collection and set(s["dev_answerable"]) <= collection
    # the four question sets are disjoint, test questions come from the test half, dev from the rest
    sets = [set(s[k]) for k in ("test_answerable", "test_unanswerable", "dev_answerable", "dev_unanswerable")]
    assert sum(len(x) for x in sets) == len(set().union(*sets))
    assert set(s["test_answerable"]) | set(s["test_unanswerable"]) <= set(test_ids)
    assert not ((set(s["dev_answerable"]) | set(s["dev_unanswerable"])) & set(test_ids))
    assert _splits(recs, test_ids) == s           # seeded, so reruns index the same corpus

    # out-of-collection questions have no gold document; answerable ones point at their own abstract
    assert all(q["gold_doc"] is None for q in pubmedqa_questions(recs, s["test_unanswerable"], answerable=False))
    assert all(q["gold_doc"] == q["qid"] for q in pubmedqa_questions(recs, s["test_answerable"], answerable=True))


def test_pubmedqa_index_text_has_no_reference_answer_title_or_label():
    recs = _records()
    s = _splits(recs, sorted(recs)[:100])
    docs = pubmedqa_documents(recs, s["collection"])
    passages = chunk_documents(docs, chunk_words=12, overlap_words=3)   # small chunks: many passages per abstract
    indexed = "\n".join(p.index_text for p in passages)

    assert {d.doc_id for d in docs} == set(s["collection"])
    assert all(d.title == "" for d in docs)
    assert "REFANSWER" not in indexed             # no conclusion / reference answer
    assert "QTITLE" not in indexed                # no title (= the question)
    for pmid in s["test_unanswerable"] + s["dev_unanswerable"]:
        assert "cohort %d." % (int(pmid) - 10_000_000) not in indexed   # no withheld abstract
    for pmid in s["collection"]:                  # but the whole abstract body is searchable
        i = int(pmid) - 10_000_000
        assert "cohort %d." % i in indexed and "group %d." % i in indexed


def test_rates_at_threshold():
    r = rates_at_threshold(answerable=[0.9, 0.8, 0.4, 0.7], unanswerable=[0.1, 0.6, 0.3], threshold=0.5)
    assert (r["n_caught"], r["n_refused"]) == (2, 1)
    assert r["caught"] == pytest.approx(2 / 3) and r["refused"] == pytest.approx(0.25)
    assert r["caught_ci"][0] < r["caught"] < r["caught_ci"][1]
