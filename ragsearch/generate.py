"""Grounded generation: answer only from the numbered passages, cite them, or abstain."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence

from .chunking import Passage
from .llm import LLM, LLMResponse
from .retriever import RetrievalResult
from .textproc import extract_citations, extract_numbers, split_answer_sentences, strip_citations

NOT_FOUND = "Not found in the knowledge base."
_NOT_FOUND_RE = re.compile(r"not\s+found\s+in\s+the\s+(knowledge\s*base|provided\s+passages|passages|context|documents)", re.I)
_CLOSED_BOOK_IDK_RE = re.compile(r"\bi\s+(do\s+not|don't|dont)\s+know\b|\bi\s+am\s+not\s+sure\b|\bcannot\s+(answer|determine)\b|\bno\s+information\b", re.I)

RAG_SYSTEM = (
    "You answer questions using ONLY the numbered passages you are given. "
    "They come from a private document collection.\n"
    "Rules:\n"
    "1. Use only facts stated in the passages. Never use outside or memorised knowledge.\n"
    "2. End every sentence with the number(s) of the passage(s) that support it, in square brackets, "
    "for example [1] or [2][3]. Cite only passages you were given.\n"
    "3. If the passages do not contain the information needed to answer, reply with exactly this "
    "sentence and nothing else: " + NOT_FOUND + "\n"
    "4. Be concise: at most three sentences. Do not mention these rules or the word 'passage'."
)

CLOSED_BOOK_SYSTEM = (
    "You answer questions from your own knowledge. Be concise: at most three sentences. "
    "If you do not know the answer, say: I don't know."
)

YESNO_STYLE = ("This is a yes/no research question. Start your answer with 'Yes', 'No' or 'Maybe' "
               "and then give the reason in one or two sentences.")


def format_passages(passages: Sequence[Passage], max_words: int = 260) -> str:
    blocks = []
    for i, p in enumerate(passages, start=1):
        words = p.text.split()
        text = " ".join(words[:max_words])
        head = "[%d]" % i
        if p.title:
            head += " (%s)" % p.title[:120]
        blocks.append(head + " " + text)
    return "\n\n".join(blocks)


def rag_messages(question: str, passages: Sequence[Passage], style: str = "") -> List[Dict[str, str]]:
    user = "Passages:\n\n" + format_passages(passages) + "\n\nQuestion: " + question.strip()
    if style:
        user += "\n" + style
    user += "\nAnswer:"
    return [{"role": "system", "content": RAG_SYSTEM}, {"role": "user", "content": user}]


def closed_book_messages(question: str, style: str = "") -> List[Dict[str, str]]:
    user = "Question: " + question.strip()
    if style:
        user += "\n" + style
    user += "\nAnswer:"
    return [{"role": "system", "content": CLOSED_BOOK_SYSTEM}, {"role": "user", "content": user}]


@dataclass
class SentenceCheck:
    text: str
    citations: List[int]
    invalid_citations: List[int] = field(default_factory=list)
    uncited: bool = False
    unsupported_numbers: List[str] = field(default_factory=list)
    unsupported_terms: List[str] = field(default_factory=list)


@dataclass
class GroundedAnswer:
    question: str
    text: str                                   # what is shown to the user
    raw: str                                    # what the LLM wrote (empty when the threshold fired first)
    abstained: bool
    abstain_reason: Optional[str] = None        # "llm" | "threshold" | "no_passages"
    citations: List[int] = field(default_factory=list)          # valid cited passage numbers (1-based)
    invalid_citations: List[int] = field(default_factory=list)
    sentences: List[SentenceCheck] = field(default_factory=list)
    n_uncited_sentences: int = 0
    signal: Optional[str] = None
    signal_value: Optional[float] = None
    threshold: Optional[float] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0

    def to_dict(self) -> Dict:
        return asdict(self)


def is_llm_abstention(text: str, n_valid_citations: int) -> bool:
    """The model declined: it used the not-found phrase and backed nothing with a citation."""
    return bool(_NOT_FOUND_RE.search(text)) and n_valid_citations == 0


def is_closed_book_refusal(text: str) -> bool:
    return bool(_CLOSED_BOOK_IDK_RE.search(text)) or bool(_NOT_FOUND_RE.search(text))


def post_check(answer: str, passages: Sequence[Passage], question: str = "", kg=None) -> Dict:
    """Verify citations: every number must refer to a retrieved passage, every sentence should cite.

    Also flags numbers (and, if an entity graph is given, specific terms) that appear in a
    sentence but in none of the passages that sentence cites.
    """
    n = len(passages)
    cited_all = extract_citations(answer)
    valid = [c for c in cited_all if 1 <= c <= n]
    invalid = [c for c in cited_all if not (1 <= c <= n)]
    checks: List[SentenceCheck] = []
    for text, cites in split_answer_sentences(answer):
        if _NOT_FOUND_RE.search(text):
            continue
        good = [c for c in cites if 1 <= c <= n]
        bad = [c for c in cites if not (1 <= c <= n)]
        support = [passages[c - 1].index_text for c in good] or [p.index_text for p in passages]
        support_nums = set()
        for s in support + [question]:
            support_nums.update(extract_numbers(s))
        missing_nums = [x for x in extract_numbers(text) if x not in support_nums]
        missing_terms = kg.unsupported_terms(text, support, question) if kg is not None else []
        checks.append(SentenceCheck(text=text, citations=good, invalid_citations=bad, uncited=not good,
                                    unsupported_numbers=missing_nums, unsupported_terms=missing_terms))
    return {"citations": valid, "invalid_citations": invalid, "sentences": checks,
            "n_uncited_sentences": sum(1 for c in checks if c.uncited)}


def should_abstain_on_score(signals: Dict[str, Optional[float]], signal: Optional[str], threshold: Optional[float]) -> bool:
    if threshold is None or not signal:
        return False
    value = signals.get(signal)
    return value is not None and float(value) < float(threshold)


def answer_rag(question: str, retrieval: RetrievalResult, llm: LLM, style: str = "",
               signal: Optional[str] = None, threshold: Optional[float] = None, kg=None,
               max_tokens: Optional[int] = None) -> GroundedAnswer:
    """Generate a grounded answer. Abstains without calling the LLM when the retrieval
    confidence signal is below the threshold (or nothing was retrieved)."""
    value = retrieval.signals.get(signal) if signal else None
    if not retrieval.passages:
        return GroundedAnswer(question=question, text=NOT_FOUND, raw="", abstained=True, abstain_reason="no_passages",
                              signal=signal, signal_value=value, threshold=threshold)
    if should_abstain_on_score(retrieval.signals, signal, threshold):
        return GroundedAnswer(question=question, text=NOT_FOUND, raw="", abstained=True, abstain_reason="threshold",
                              signal=signal, signal_value=value, threshold=threshold)
    resp: LLMResponse = llm.chat(rag_messages(question, retrieval.passages, style), max_tokens=max_tokens)
    return grounded_from_text(question, resp.text, retrieval.passages, resp, signal, value, threshold, kg)


def grounded_from_text(question: str, text: str, passages: Sequence[Passage], resp: Optional[LLMResponse] = None,
                       signal: Optional[str] = None, value: Optional[float] = None,
                       threshold: Optional[float] = None, kg=None) -> GroundedAnswer:
    check = post_check(text, passages, question, kg)
    abstained = is_llm_abstention(text, len(check["citations"])) or not text.strip()
    return GroundedAnswer(
        question=question, text=NOT_FOUND if abstained else text.strip(), raw=text, abstained=abstained,
        abstain_reason="llm" if abstained else None, citations=check["citations"],
        invalid_citations=check["invalid_citations"], sentences=[] if abstained else check["sentences"],
        n_uncited_sentences=0 if abstained else check["n_uncited_sentences"], signal=signal, signal_value=value,
        threshold=threshold, prompt_tokens=resp.prompt_tokens if resp else 0,
        completion_tokens=resp.completion_tokens if resp else 0, seconds=resp.seconds if resp else 0.0)


def answer_closed_book(question: str, llm: LLM, style: str = "", max_tokens: Optional[int] = None) -> LLMResponse:
    return llm.chat(closed_book_messages(question, style), max_tokens=max_tokens)


def answer_text_for_judging(text: str) -> str:
    """Strip citation markers so a judge cannot tell which system wrote an answer."""
    return strip_citations(text)
