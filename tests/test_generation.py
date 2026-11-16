from __future__ import annotations

import pytest

from ragsearch.chunking import Passage
from ragsearch.eval import citation as C
from ragsearch.eval import faithfulness as F
from ragsearch.eval.judge import decision_of, pairwise, rate_passage
from ragsearch.generate import (NOT_FOUND, answer_rag, is_closed_book_refusal, is_llm_abstention, post_check,
                                rag_messages, should_abstain_on_score)
from ragsearch.llm import LLMCache, LLMError, OllamaLLM, ScriptedLLM, parse_json_object
from ragsearch.retriever import Hit, RetrievalResult

P = [Passage(0, "d1", "Travel policy", "Lodging is reimbursed up to 180 dollars per night. In high-cost cities the limit is 240 dollars.", "travel.pdf", 2),
     Passage(1, "d2", "Leave policy", "Staff accrue 10 hours of vacation per month. Up to 240 hours carry over.", "leave.md"),
     Passage(2, "d3", "Lab manual", "Eyewash stations are flushed weekly by laboratory staff.", "lab.md")]


def result(signals=None):
    hits = [Hit(pid=i, rank=i + 1, score=1.0 / (i + 1), bm25_rank=i + 1, dense_rank=i + 1) for i in range(3)]
    return RetrievalResult("q", "hybrid", hits, P, signals or {"dense_top": 0.8, "rrf_top": 0.0328})


def test_prompt_contains_numbered_passages_and_rules():
    msgs = rag_messages("What is the lodging limit?", P)
    assert "[1] (Travel policy) Lodging" in msgs[1]["content"] and "[3] (Lab manual)" in msgs[1]["content"]
    assert NOT_FOUND in msgs[0]["content"] and "ONLY" in msgs[0]["content"]


def test_post_check_flags_invalid_uncited_and_unsupported_numbers():
    chk = post_check("The limit is 240 dollars in high-cost cities [1]. It rises to 300 dollars next year [7]. Staff accrue leave monthly.", P)
    assert chk["citations"] == [1] and chk["invalid_citations"] == [7]
    s1, s2, s3 = chk["sentences"]
    assert not s1.uncited and s1.unsupported_numbers == []
    assert s2.uncited and s2.invalid_citations == [7] and "300" in s2.unsupported_numbers
    assert s3.uncited and chk["n_uncited_sentences"] == 2


def test_abstention_detection():
    assert is_llm_abstention("Not found in the knowledge base.", 0)
    assert is_llm_abstention("The answer is not found in the provided passages.", 0)
    assert not is_llm_abstention("Lodging is 180 dollars [1]. The rest is not found in the knowledge base.", 1)
    assert is_closed_book_refusal("I don't know.") and not is_closed_book_refusal("Yes, it is 240 dollars.")
    assert should_abstain_on_score({"dense_top": 0.2}, "dense_top", 0.3)
    assert not should_abstain_on_score({"dense_top": 0.4}, "dense_top", 0.3)
    assert not should_abstain_on_score({"dense_top": None}, "dense_top", 0.3)


def test_answer_rag_paths():
    calls = []

    def fn(messages, json_mode):
        calls.append(messages)
        return "The limit is 240 dollars per night in high-cost cities [1]."

    llm = ScriptedLLM(fn)
    ans = answer_rag("What is the lodging limit in high-cost cities?", result(), llm)
    assert not ans.abstained and ans.citations == [1] and ans.n_uncited_sentences == 0
    low = answer_rag("q", result({"dense_top": 0.1}), llm, signal="dense_top", threshold=0.3)
    assert low.abstained and low.abstain_reason == "threshold" and low.text == NOT_FOUND and len(calls) == 1   # LLM not called
    refuse = answer_rag("q", result(), ScriptedLLM(lambda m, j: "Not found in the knowledge base."))
    assert refuse.abstained and refuse.abstain_reason == "llm"
    empty = answer_rag("q", RetrievalResult("q", "hybrid", [], [], {}), llm)
    assert empty.abstained and empty.abstain_reason == "no_passages"


def test_alce_citation_quality():
    def entails(premise, hypothesis):                 # word-overlap stand-in for NLI
        return all(w in premise.lower() for w in hypothesis.lower().rstrip(".").split() if len(w) > 3)

    q = C.citation_quality("Lodging limit is 240 dollars [1][3]. Eyewash stations are flushed weekly [2]. Staff accrue vacation.", P, entails)
    s1, s2, s3 = q["statements"]
    assert s1["supported"] and s1["citation_correct"] == [True, False]      # [3] is irrelevant: [1] alone suffices
    assert not s2["supported"] and s2["citation_correct"] == [False]        # cites the wrong passage
    assert not s3["supported"] and s3["citations"] == []                    # no citation -> no recall
    assert q["recall"] == pytest.approx(1 / 3) and q["precision"] == pytest.approx(1 / 3)
    agg = C.aggregate([q])
    assert agg["n_statements"] == 3 and agg["n_citations"] == 3

    # a bare "Yes." (the PubMedQA answer format) is not a checkable statement and is skipped
    q2 = C.citation_quality("Yes [1]. Lodging limit is 240 dollars [1].", P, entails)
    assert q2["n_statements"] == 1 and q2["n_skipped_labels"] == 1 and q2["precision"] == 1.0
    assert C.is_bare_label("Maybe.") and not C.is_bare_label("No difference was found.")


def test_nli_premise_windows():
    from ragsearch.rerank import premise_windows

    text = "One is here. Two is here. Three is here. Four is here."
    assert premise_windows(text, 0) == [text]
    assert premise_windows(text, 5) == [text]                      # shorter than a window: whole premise
    assert premise_windows(text, 2) == ["One is here. Two is here.", "Two is here. Three is here.",
                                        "Three is here. Four is here."]


def test_faithfulness_and_judge_with_scripted_models():
    def judge(messages, json_mode):
        sys = messages[0]["content"]
        if "atomic factual claims" in sys:
            return '{"claims": ["The limit is 240 dollars.", "It applies in Paris."]}'
        if "verify claims" in sys:
            return '{"verdicts": [{"claim": 1, "supported": true}, {"claim": 2, "supported": false}]}'
        if "impartial grader" in sys:                 # always prefers the answer containing "240"
            a = messages[1]["content"].split("Answer A:")[1].split("Answer B:")[0]
            return '{"reason": "matches", "winner": "%s"}' % ("A" if "240" in a else "B")
        if "0-5 scale" in sys:
            return "Score: 4"
        return '{"decision": "maybe"}'

    llm = ScriptedLLM(judge)
    f = F.faithfulness(llm, "q", "The limit is 240 dollars in Paris [1].", P)
    assert f["score"] == 0.5 and f["supported"] == [True, False]
    v = pairwise(llm, "q", "240 dollars", "It is 240 dollars.", "It is 100 dollars.")
    assert v["verdict"] == "x" and v["consistent"]
    biased = ScriptedLLM(lambda m, j: '{"winner": "A"}')         # a judge with pure position bias yields a tie
    assert pairwise(biased, "q", "ref", "one", "two")["verdict"] == "tie"
    assert pairwise(llm, "q", "ref", "same", "same")["identical"]
    assert rate_passage(llm, "q", "p") == 4
    assert decision_of("The data are inconclusive overall.", "q", llm) == "maybe" and decision_of("No, it does not.") == "no"
    assert parse_json_object('Sure! {"winner": "B"} hope that helps') == {"winner": "B"}


def test_ollama_client_cache_and_locality(fake_server, tmp_path):
    cache = LLMCache(str(tmp_path / "c.sqlite"))
    llm = OllamaLLM("llama3.1:8b", host=fake_server, cache=cache)
    assert llm.is_up() and "qwen2.5:7b" in llm.available_models()
    msgs = rag_messages("What is the lodging limit in high-cost cities?", P)
    first = llm.chat(msgs)
    second = llm.chat(msgs)
    assert "[1]" in first.text and second.cached and second.text == first.text and llm.calls == 1 and llm.cache_hits == 1
    with pytest.raises(LLMError):
        OllamaLLM("no-such-model", host=fake_server).chat(msgs)
    with pytest.raises(LLMError):                       # documents must not leave this machine by accident
        OllamaLLM("llama3.1:8b", host="https://api.example.com")
    with pytest.raises(LLMError):
        OllamaLLM("llama3.1:8b", host="http://127.0.0.1:9").chat(msgs)
