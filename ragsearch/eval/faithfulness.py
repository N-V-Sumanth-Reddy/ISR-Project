"""RAGAs-style faithfulness (Es et al., 2024), re-implemented against the local judge LLM.

Step 1: the judge decomposes the answer into atomic claims.
Step 2: the judge checks each claim against the retrieved passages.
faithfulness = supported claims / all claims.

We do not call the `ragas` package: it expects a hosted LLM by default, and keeping the
two prompts here means every evaluation call stays on this machine and is cached.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from ..chunking import Passage
from ..generate import format_passages
from ..llm import LLM, parse_json_object
from ..textproc import split_answer_sentences, strip_citations

CLAIMS_SYSTEM = (
    "You break an answer into its atomic factual claims. Each claim must be a short, self-contained "
    "sentence that can be checked on its own (resolve pronouns using the question). Do not add facts "
    'that are not in the answer. Reply with a JSON object: {"claims": ["...", "..."]}'
)

VERIFY_SYSTEM = (
    "You verify claims against a context. For each claim decide whether the context states or directly "
    "implies it. Use only the context, not your own knowledge. Reply with a JSON object: "
    '{"verdicts": [{"claim": 1, "supported": true or false}, ...]} with one entry per claim, in order.'
)


def extract_claims(llm: LLM, question: str, answer: str) -> List[str]:
    clean = strip_citations(answer)
    resp = llm.chat([{"role": "system", "content": CLAIMS_SYSTEM},
                     {"role": "user", "content": "Question: %s\n\nAnswer: %s" % (question, clean)}],
                    max_tokens=300, json_mode=True)
    obj = parse_json_object(resp.text) or {}
    claims = obj.get("claims")
    if isinstance(claims, list):
        claims = [str(c).strip() for c in claims if isinstance(c, (str, int, float)) and str(c).strip()]
    if not claims:                                   # fall back to sentences
        claims = [s for s, _ in split_answer_sentences(answer)]
    return claims[:12]


def _as_bool(val) -> bool:
    return val is True or str(val).strip().lower() in ("true", "yes", "1", "supported")


def verify_claims(llm: LLM, claims: Sequence[str], passages: Sequence[Passage]) -> List[bool]:
    if not claims:
        return []
    listing = "\n".join("%d. %s" % (i, c) for i, c in enumerate(claims, start=1))
    resp = llm.chat([{"role": "system", "content": VERIFY_SYSTEM},
                     {"role": "user", "content": "Context:\n\n%s\n\nClaims:\n%s" % (format_passages(passages), listing)}],
                    max_tokens=40 + 25 * len(claims), json_mode=True)
    obj = parse_json_object(resp.text) or {}
    verdicts = obj.get("verdicts")
    out: List[Optional[bool]] = [None] * len(claims)
    if isinstance(verdicts, list):
        for pos, item in enumerate(verdicts):
            if isinstance(item, dict):
                idx = item.get("claim")
                idx = int(idx) - 1 if isinstance(idx, (int, float)) or (isinstance(idx, str) and idx.isdigit()) else pos
                if 0 <= idx < len(claims) and out[idx] is None:
                    out[idx] = _as_bool(item.get("supported"))
            elif pos < len(claims):
                out[pos] = _as_bool(item)
    if any(v is None for v in out):                  # unparsed entries: check one by one
        for i, v in enumerate(out):
            if v is None:
                single = llm.chat([{"role": "system", "content": VERIFY_SYSTEM},
                                   {"role": "user", "content": "Context:\n\n%s\n\nClaims:\n1. %s" % (format_passages(passages), claims[i])}],
                                  max_tokens=60, json_mode=True)
                o = parse_json_object(single.text) or {}
                vs = o.get("verdicts")
                out[i] = _as_bool(vs[0].get("supported")) if isinstance(vs, list) and vs and isinstance(vs[0], dict) else False
    return [bool(v) for v in out]


def faithfulness(llm: LLM, question: str, answer: str, passages: Sequence[Passage]) -> Dict:
    claims = extract_claims(llm, question, answer)
    verdicts = verify_claims(llm, claims, passages)
    score = (sum(verdicts) / len(verdicts)) if verdicts else None
    return {"score": score, "claims": claims, "supported": verdicts}
