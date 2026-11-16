"""LLM-as-a-judge utilities (Zheng et al., 2023): reference-based pairwise comparison,
run in both presentation orders to control position bias; plus small rating helpers."""
from __future__ import annotations

import re
from typing import Dict, List, Optional

from ..llm import LLM, parse_json_object
from ..textproc import first_decision

JUDGE_SYSTEM = (
    "You are an impartial grader. You compare two candidate answers to a research question against a "
    "reference answer written by the authors of the study. Judge ONLY agreement with the reference: "
    "its conclusion (yes / no / uncertain) and its key facts. Ignore style, length and fluency.\n"
    "Scoring rules:\n"
    "- An answer that agrees with the reference beats one that contradicts it or misses its conclusion.\n"
    "- An answer that declines to answer (for example 'Not found in the knowledge base' or 'I don't know') "
    "is worse than an answer that agrees with the reference, but better than an answer that contradicts it.\n"
    "- If both answers agree with the reference equally well, or both are equally wrong, it is a tie.\n"
    'Reply with a JSON object: {"reason": "<one short sentence>", "winner": "A" or "B" or "tie"}'
)


def _judge_messages(question: str, reference: str, answer_a: str, answer_b: str) -> List[Dict[str, str]]:
    user = ("Question: %s\n\nReference answer: %s\n\nAnswer A: %s\n\nAnswer B: %s\n\n"
            "Which answer is more consistent with the reference answer?" % (question, reference, answer_a, answer_b))
    return [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}]


def _parse_winner(text: str) -> Optional[str]:
    obj = parse_json_object(text)
    cand = str(obj.get("winner", "")).strip().lower() if obj else ""
    if not cand:
        m = re.search(r"winner\W+(a|b|tie)\b", text, re.I) or re.search(r"\b(answer\s+)?(a|b|tie)\b\s*$", text.strip(), re.I)
        cand = m.group(m.lastindex).lower() if m else ""
    cand = cand.replace("answer", "").strip(" .\"'")
    if cand in ("a", "b"):
        return cand.upper()
    if cand.startswith("tie") or cand in ("both", "neither", "equal", "draw"):
        return "tie"
    return None


def judge_once(llm: LLM, question: str, reference: str, answer_a: str, answer_b: str) -> Dict:
    resp = llm.chat(_judge_messages(question, reference, answer_a, answer_b), max_tokens=200, json_mode=True)
    winner = _parse_winner(resp.text)
    obj = parse_json_object(resp.text) or {}
    return {"winner": winner or "tie", "parsed": winner is not None, "reason": str(obj.get("reason", ""))[:300], "raw": resp.text[:500]}


def pairwise(llm: LLM, question: str, reference: str, answer_x: str, answer_y: str) -> Dict:
    """Compare X and Y in both orders. X (or Y) wins only if it is preferred in both orders;
    anything else is a tie. ``consistent`` records whether the two orders agreed."""
    if answer_x.strip() == answer_y.strip():
        return {"verdict": "tie", "consistent": True, "order_xy": None, "order_yx": None, "identical": True}
    first = judge_once(llm, question, reference, answer_x, answer_y)       # X shown as A
    second = judge_once(llm, question, reference, answer_y, answer_x)      # X shown as B
    v1 = {"A": "x", "B": "y", "tie": "tie"}[first["winner"]]
    v2 = {"A": "y", "B": "x", "tie": "tie"}[second["winner"]]
    verdict = v1 if v1 == v2 else "tie"
    return {"verdict": verdict, "consistent": v1 == v2, "order_xy": first, "order_yx": second, "identical": False}


# --------------------------------------------------------------------- decision label

DECISION_SYSTEM = ("You read an answer to a yes/no research question and report what the answer concludes. "
                   'Reply with a JSON object: {"decision": "yes" or "no" or "maybe" or "none"}. '
                   'Use "maybe" when the answer says the evidence is mixed or uncertain, and "none" when the '
                   "answer declines to answer or gives no conclusion.")


def decision_of(answer: str, question: str = "", llm: Optional[LLM] = None) -> Optional[str]:
    """yes / no / maybe from the opening words; falls back to asking the judge model."""
    d = first_decision(answer)
    if d is not None or llm is None or not answer.strip():
        return d
    resp = llm.chat([{"role": "system", "content": DECISION_SYSTEM},
                     {"role": "user", "content": "Question: %s\n\nAnswer: %s" % (question, answer)}],
                    max_tokens=30, json_mode=True)
    obj = parse_json_object(resp.text) or {}
    cand = str(obj.get("decision", "")).strip().lower()
    return cand if cand in ("yes", "no", "maybe") else None


# ------------------------------------------------------------------ DAT-style ratings

RATE_SYSTEM = (
    "You rate how well a retrieved passage answers a search query, on a 0-5 scale:\n"
    "5 = directly answers the query; 3-4 = on topic and close, but not a direct answer; "
    "1-2 = loosely related; 0 = unrelated.\n"
    'Reply with a JSON object: {"score": <integer 0-5>}'
)


def rate_passage(llm: LLM, query: str, passage: str) -> int:
    resp = llm.chat([{"role": "system", "content": RATE_SYSTEM},
                     {"role": "user", "content": "Query: %s\n\nPassage: %s" % (query, " ".join(passage.split()[:260]))}],
                    max_tokens=20, json_mode=True)
    obj = parse_json_object(resp.text) or {}
    try:
        score = int(round(float(obj.get("score"))))
    except (TypeError, ValueError):
        m = re.search(r"\b([0-5])\b", resp.text)
        score = int(m.group(1)) if m else 0
    return max(0, min(5, score))
