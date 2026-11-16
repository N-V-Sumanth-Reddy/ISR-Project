"""Text utilities: tokenisation for BM25, sentence splitting, citation parsing."""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

# Lucene's default English stop set (what Anserini / Pyserini BM25 uses).
STOPWORDS = frozenset(
    "a an and are as at be but by for if in into is it no not of on or such that the "
    "their then there these they this to was will with".split()
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class Tokenizer:
    """Lowercase, split on non-alphanumerics, drop stop words, Porter-stem (memoised)."""

    def __init__(self, stem: bool = True, stopwords: bool = True):
        self.stem = stem
        self.stopwords = stopwords
        self._memo: Dict[str, str] = {}
        self._stemmer = None
        if stem:
            try:
                import snowballstemmer

                self._stemmer = snowballstemmer.stemmer("porter")
            except Exception:  # pragma: no cover - optional dependency
                self._stemmer = None
                self.stem = False

    def _stem(self, word: str) -> str:
        cached = self._memo.get(word)
        if cached is None:
            cached = self._stemmer.stemWord(word) if self._stemmer is not None else word
            self._memo[word] = cached
        return cached

    def __call__(self, text: str) -> List[str]:
        tokens = _TOKEN_RE.findall(text.lower())
        if self.stopwords:
            tokens = [t for t in tokens if t not in STOPWORDS]
        if self.stem:
            tokens = [self._stem(t) for t in tokens]
        return tokens

    def signature(self) -> Dict[str, bool]:
        return {"stem": bool(self.stem), "stopwords": bool(self.stopwords)}


# --------------------------------------------------------------------------- sentences

_ABBREV = (
    "e.g.", "i.e.", "et al.", "vs.", "fig.", "figs.", "eq.", "no.", "dr.", "mr.", "mrs.",
    "ms.", "prof.", "inc.", "ltd.", "approx.", "sec.", "ch.", "u.s.", "st.", "cf.",
)
_PLACEHOLDER = "․"  # one-dot leader, stands in for protected periods
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[\"'(\[]*[A-Z0-9])")


def split_sentences(text: str) -> List[str]:
    """Light-weight sentence splitter that protects common abbreviations and decimals."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    protected = text
    lowered = protected.lower()
    for abbr in _ABBREV:
        start = 0
        while True:
            idx = lowered.find(abbr, start)
            if idx < 0:
                break
            if idx == 0 or not lowered[idx - 1].isalnum():
                piece = protected[idx: idx + len(abbr)].replace(".", _PLACEHOLDER)
                protected = protected[:idx] + piece + protected[idx + len(abbr):]
            start = idx + len(abbr)
    # single capital initials such as "J. Smith"
    protected = re.sub(r"\b([A-Z])\.(?=\s+[A-Z][a-z])", lambda m: m.group(1) + _PLACEHOLDER, protected)
    parts = _SENT_SPLIT_RE.split(protected)
    return [p.replace(_PLACEHOLDER, ".").strip() for p in parts if p.strip()]


def word_count(text: str) -> int:
    return len(text.split())


# --------------------------------------------------------------------------- citations

# [1], [1][2], [1, 2], [1-3], [1 and 2]
_CITE_GROUP_RE = re.compile(r"\[\s*(\d+(?:\s*(?:,|;|-|–|and|&)\s*\d+)*)\s*\]")


def parse_citation_group(body: str) -> List[int]:
    nums: List[int] = []
    for part in re.split(r"\s*(?:,|;|and|&)\s*", body):
        part = part.strip()
        if not part:
            continue
        rng = re.match(r"^(\d+)\s*[-–]\s*(\d+)$", part)
        if rng:
            lo, hi = int(rng.group(1)), int(rng.group(2))
            if lo <= hi and hi - lo <= 20:
                nums.extend(range(lo, hi + 1))
            else:
                nums.extend([lo, hi])
        elif part.isdigit():
            nums.append(int(part))
    return nums


def extract_citations(text: str) -> List[int]:
    """All cited passage numbers in order of first appearance (deduplicated)."""
    seen: List[int] = []
    for match in _CITE_GROUP_RE.finditer(text):
        for n in parse_citation_group(match.group(1)):
            if n not in seen:
                seen.append(n)
    return seen


def strip_citations(text: str) -> str:
    out = _CITE_GROUP_RE.sub("", text)
    out = re.sub(r"\s+([.,;:!?])", r"\1", out)
    return re.sub(r"\s{2,}", " ", out).strip()


def split_answer_sentences(answer: str) -> List[Tuple[str, List[int]]]:
    """Split an answer into (sentence_without_citations, cited_numbers).

    A citation group that follows the sentence-final punctuation ("... dose. [2]")
    is attached to the sentence it follows.
    """
    text = re.sub(r"\s+", " ", answer).strip()
    if not text:
        return []
    # move trailing citation groups inside the sentence: "foo. [1][2] Bar" -> "foo [1][2]. Bar"
    text = re.sub(r"([.!?])((?:\s*\[\s*\d+(?:\s*(?:,|;|-|–|and|&)\s*\d+)*\s*\])+)", r"\2\1", text)
    out: List[Tuple[str, List[int]]] = []
    for sent in split_sentences(text):
        cites = extract_citations(sent)
        clean = strip_citations(sent)
        if clean and re.search(r"[A-Za-z0-9]", clean):
            out.append((clean, cites))
        elif cites and out:  # a dangling citation-only fragment belongs to the previous sentence
            prev_text, prev_cites = out[-1]
            out[-1] = (prev_text, prev_cites + [c for c in cites if c not in prev_cites])
    return out


_NUM_RE = re.compile(r"(?<![A-Za-z0-9])\d+(?:[.,]\d+)*%?")


def extract_numbers(text: str) -> List[str]:
    """Numeric tokens (normalised: thousands separators removed, trailing % kept apart)."""
    out = []
    for raw in _NUM_RE.findall(text):
        num = raw.rstrip("%")
        if re.fullmatch(r"\d{1,3}(,\d{3})+(\.\d+)?", num):
            num = num.replace(",", "")
        num = num.rstrip(".,")
        if num and num not in out:
            out.append(num)
    return out


def first_decision(answer: str) -> Optional[str]:
    """Map a free-text answer to yes / no / maybe from its opening words."""
    head = strip_citations(answer).strip().lower()[:80]
    head = re.sub(r"^[\"'*\s]+", "", head)
    if re.match(r"^yes\b", head):
        return "yes"
    if re.match(r"^no\b", head):
        return "no"
    if re.match(r"^(maybe|possibly|unclear|uncertain|inconclusive|it is unclear|it is uncertain|partially|partly)\b", head):
        return "maybe"
    return None
