"""Read a folder of private documents (PDF / text / markdown / HTML) into Documents."""
from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from .chunking import Document

SUPPORTED = {".pdf", ".txt", ".md", ".markdown", ".rst", ".html", ".htm"}


def _clean(text: str) -> str:
    text = text.replace("­", "")                      # soft hyphens
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)           # words hyphenated across lines
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _strip_markdown(text: str) -> str:
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    # headings and list items become their own sentences
    text = re.sub(r"^\s{0,3}#{1,6}\s*(.+?)\s*#*\s*$", lambda m: m.group(1).rstrip(".:") + ".", text, flags=re.M)
    text = re.sub(r"^\s*[-*+]\s+(.+?)\s*$", lambda m: m.group(1) if m.group(1)[-1:] in ".!?:" else m.group(1) + ".", text, flags=re.M)
    text = re.sub(r"^\s*\d+[.)]\s+(.+?)\s*$", lambda m: m.group(1) if m.group(1)[-1:] in ".!?:" else m.group(1) + ".", text, flags=re.M)
    text = re.sub(r"[*_`>|]{1,3}", " ", text)
    return text


def _strip_html(text: str) -> str:
    text = re.sub(r"(?is)<(script|style|nav|footer|header|noscript)[^>]*>.*?</\1>", " ", text)
    text = re.sub(r"(?i)<br\s*/?>|</p>|</div>|</li>|</h[1-6]>|</tr>", "\n", text)
    text = re.sub(r"<[^>]+>", " ", text)
    return html.unescape(text)


def _title_from_text(text: str, fallback: str) -> str:
    for line in text.splitlines():
        line = line.strip().lstrip("#").strip()
        if 3 <= len(line) <= 120:
            return line
    return fallback


def read_pdf(path: Path) -> Tuple[str, List[Tuple[Optional[int], str]]]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages: List[Tuple[Optional[int], str]] = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        text = _clean(text)
        if text:
            pages.append((i, text))
    title = ""
    try:
        title = (reader.metadata.title or "").strip() if reader.metadata else ""
    except Exception:
        title = ""
    return title, pages


def read_file(path: Path, root: Optional[Path] = None) -> Optional[Document]:
    """Read one file. Returns None for unsupported or empty files."""
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED:
        return None
    rel = str(path.relative_to(root)) if root and root in path.parents else path.name
    fallback_title = path.stem.replace("_", " ").replace("-", " ").strip()
    if suffix == ".pdf":
        title, pages = read_pdf(path)
        title = title or fallback_title
    else:
        raw = path.read_text(encoding="utf-8", errors="ignore")
        if suffix in {".html", ".htm"}:
            m = re.search(r"(?is)<title[^>]*>(.*?)</title>", raw)
            title = html.unescape(m.group(1)).strip() if m else fallback_title
            body = _clean(_strip_html(raw))
        elif suffix in {".md", ".markdown"}:
            title = _title_from_text(raw, fallback_title)
            body = _clean(_strip_markdown(raw))
        else:
            title = fallback_title
            body = _clean(raw)
        pages = [(None, body)] if body else []
    if not pages:
        return None
    return Document(doc_id=rel, title=title, source=rel, pages=pages)


def read_folder(folder: str) -> List[Document]:
    root = Path(folder)
    if not root.exists():
        raise FileNotFoundError(folder)
    docs: List[Document] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.name.startswith("."):
            doc = read_file(path, root)
            if doc is not None:
                docs.append(doc)
    return docs


def documents_from_records(records: Iterable[dict], source: str) -> List[Document]:
    """Build Documents from dataset records with keys: _id / id, title, text."""
    docs = []
    for rec in records:
        doc_id = str(rec.get("_id", rec.get("id")))
        docs.append(Document(doc_id=doc_id, title=(rec.get("title") or "").strip(), source=source,
                             pages=[(None, rec.get("text") or "")]))
    return docs
