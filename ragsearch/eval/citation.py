"""ALCE-style citation quality (Gao et al., 2023).

For every answer statement s with cited passages C:
  citation recall    = 1 if C is non-empty and the cited passages together entail s
  citation precision = a citation c in C is counted as correct when the statement is
                       supported (recall = 1) and c is not irrelevant, where c is irrelevant
                       if c alone does not entail s AND C without c still entails s.

Entailment is decided by an NLI cross-encoder (default, as in ALCE) or by the local judge LLM.
As Wallat et al. (2024) point out, a citation that passes this test supports the statement
but did not necessarily cause it: this measures citation correctness, not faithfulness.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Sequence

from ..chunking import Passage
from ..llm import LLM, parse_json_object
from ..textproc import split_answer_sentences

Entails = Callable[[str, str], bool]

ENTAIL_SYSTEM = ("You check whether a passage supports a statement. Answer strictly from the passage. "
                 'Reply with a JSON object: {"supported": true or false}. '
                 "Say true only if the passage states or directly implies the whole statement.")


def llm_entailment(llm: LLM) -> Entails:
    def entails(premise: str, hypothesis: str) -> bool:
        resp = llm.chat([{"role": "system", "content": ENTAIL_SYSTEM},
                         {"role": "user", "content": "Passage: %s\n\nStatement: %s" % (premise, hypothesis)}],
                        max_tokens=20, json_mode=True)
        obj = parse_json_object(resp.text)
        if obj is not None and "supported" in obj:
            val = obj["supported"]
            return val is True or str(val).strip().lower() in ("true", "yes", "1")
        return resp.text.strip().lower().startswith(("true", "yes"))
    return entails


def _set_entails(entails: Entails, texts: Sequence[str], statement: str) -> bool:
    if not texts:
        return False
    if len(texts) == 1:
        return entails(texts[0], statement)
    # the concatenation may exceed an NLI model's window, so a single passage also suffices
    return entails("\n".join(texts), statement) or any(entails(t, statement) for t in texts)


def is_bare_label(text: str) -> bool:
    """A sentence that is only "Yes." / "No." / "Maybe." states no checkable claim on its own."""
    return text.strip().rstrip(".!").strip().lower() in ("yes", "no", "maybe")


def citation_quality(answer: str, passages: Sequence[Passage], entails: Entails) -> Dict:
    n = len(passages)
    statements = []
    skipped = 0
    for text, cites in split_answer_sentences(answer):
        if is_bare_label(text):          # PubMedQA answers open with the label; no passage can entail "Yes."
            skipped += 1
            continue
        valid = [c for c in dict.fromkeys(cites) if 1 <= c <= n]
        cited_texts = [passages[c - 1].index_text for c in valid]
        supported = _set_entails(entails, cited_texts, text)
        correct = []
        for i, c in enumerate(valid):
            if not supported:
                correct.append(False)
                continue
            alone = entails(cited_texts[i], text)
            rest = cited_texts[:i] + cited_texts[i + 1:]
            irrelevant = (not alone) and _set_entails(entails, rest, text)
            correct.append(not irrelevant)
        statements.append({"text": text, "citations": valid, "n_invalid": len(cites) - len(valid),
                           "supported": bool(supported), "citation_correct": correct})
    n_stmt = len(statements)
    n_cit = sum(len(s["citations"]) for s in statements)
    return {
        "statements": statements, "n_statements": n_stmt, "n_citations": n_cit, "n_skipped_labels": skipped,
        "recall": (sum(1 for s in statements if s["supported"]) / n_stmt) if n_stmt else None,
        "precision": (sum(sum(s["citation_correct"]) for s in statements) / n_cit) if n_cit else None,
    }


def aggregate(per_answer: List[Dict]) -> Dict:
    """Micro (over statements / citations) and macro (over answers) averages."""
    n_stmt = sum(a["n_statements"] for a in per_answer)
    n_cit = sum(a["n_citations"] for a in per_answer)
    sup = sum(sum(1 for s in a["statements"] if s["supported"]) for a in per_answer)
    cor = sum(sum(sum(s["citation_correct"]) for s in a["statements"]) for a in per_answer)
    rec = [a["recall"] for a in per_answer if a["recall"] is not None]
    pre = [a["precision"] for a in per_answer if a["precision"] is not None]
    return {"n_answers": len(per_answer), "n_statements": n_stmt, "n_citations": n_cit,
            "n_skipped_labels": sum(a.get("n_skipped_labels", 0) for a in per_answer),
            "citation_recall_micro": sup / n_stmt if n_stmt else None,
            "citation_precision_micro": cor / n_cit if n_cit else None,
            "citation_recall_macro": sum(rec) / len(rec) if rec else None,
            "citation_precision_macro": sum(pre) / len(pre) if pre else None}
