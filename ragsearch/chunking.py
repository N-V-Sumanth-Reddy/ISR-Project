"""Split documents into overlapping passages of about N words, keeping file and page."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .textproc import split_sentences, word_count


@dataclass
class Document:
    doc_id: str
    title: str = ""
    source: str = ""                      # file path or dataset name
    pages: List[Tuple[Optional[int], str]] = field(default_factory=list)  # (page number, text)
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Passage:
    pid: int
    doc_id: str
    title: str
    text: str
    source: str = ""
    page: Optional[int] = None            # page of the first sentence
    page_end: Optional[int] = None        # page of the last sentence
    chunk: int = 0                        # index of this passage inside its document

    @property
    def index_text(self) -> str:
        """Text that is indexed: the title is prepended so every chunk carries it."""
        return (self.title + ". " + self.text) if self.title else self.text

    def location(self) -> str:
        loc = self.source or self.doc_id
        if self.page is not None:
            if self.page_end is not None and self.page_end != self.page:
                loc += ", pp. %d-%d" % (self.page, self.page_end)
            else:
                loc += ", p. %d" % self.page
        return loc

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Passage":
        return cls(**{k: d.get(k) for k in ("pid", "doc_id", "title", "text", "source", "page", "page_end", "chunk")})


def _hard_split(sentence: str, chunk_words: int, overlap_words: int) -> List[str]:
    words = sentence.split()
    step = max(1, chunk_words - overlap_words)
    return [" ".join(words[i: i + chunk_words]) for i in range(0, len(words), step) if words[i: i + chunk_words]]


def chunk_units(
    units: Sequence[Tuple[str, Optional[int]]], chunk_words: int = 200, overlap_words: int = 40
) -> List[Tuple[str, Optional[int], Optional[int]]]:
    """Pack (sentence, page) units into chunks of <= chunk_words words.

    Consecutive chunks share trailing sentences worth up to ``overlap_words`` words.
    Returns (text, first_page, last_page) triples.
    """
    if chunk_words <= 0:
        raise ValueError("chunk_words must be positive")
    overlap_words = max(0, min(overlap_words, chunk_words // 2))

    # sentences longer than a whole chunk are cut into word windows first
    flat: List[Tuple[str, Optional[int], int]] = []
    for sent, page in units:
        n = word_count(sent)
        if n == 0:
            continue
        if n > chunk_words:
            for piece in _hard_split(sent, chunk_words, overlap_words):
                flat.append((piece, page, word_count(piece)))
        else:
            flat.append((sent, page, n))

    chunks: List[Tuple[str, Optional[int], Optional[int]]] = []
    current: List[Tuple[str, Optional[int], int]] = []
    current_words = 0
    fresh = 0  # sentences in `current` that were not carried over as overlap

    def emit() -> None:
        chunks.append((" ".join(s for s, _, _ in current), current[0][1], current[-1][1]))

    for sent, page, n in flat:
        if current and current_words + n > chunk_words:
            emit()
            carry: List[Tuple[str, Optional[int], int]] = []
            carried = 0
            for item in reversed(current):
                if carried + item[2] > overlap_words:
                    break
                carry.insert(0, item)
                carried += item[2]
            if carried + n > chunk_words:  # overlap would not leave room for the new sentence
                carry, carried = [], 0
            current, current_words, fresh = carry, carried, 0
        current.append((sent, page, n))
        current_words += n
        fresh += 1
    if current and fresh > 0:
        emit()
    return chunks


def chunk_document(doc: Document, chunk_words: int = 200, overlap_words: int = 40) -> List[Tuple[str, Optional[int], Optional[int]]]:
    units: List[Tuple[str, Optional[int]]] = []
    for page, text in doc.pages:
        for sent in split_sentences(text or ""):
            units.append((sent, page))
    return chunk_units(units, chunk_words, overlap_words)


def chunk_documents(
    docs: Iterable[Document], chunk_words: int = 200, overlap_words: int = 40, start_pid: int = 0
) -> List[Passage]:
    passages: List[Passage] = []
    pid = start_pid
    for doc in docs:
        pieces = chunk_document(doc, chunk_words, overlap_words)
        if not pieces and doc.title:  # title-only records (some TREC-COVID entries have no abstract)
            pieces = [(doc.title, None, None)]
        for idx, (text, p0, p1) in enumerate(pieces):
            passages.append(
                Passage(pid=pid, doc_id=doc.doc_id, title=doc.title, text=text, source=doc.source,
                        page=p0, page_end=p1, chunk=idx)
            )
            pid += 1
    return passages
