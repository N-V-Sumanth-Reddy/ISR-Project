"""Public stand-ins for private knowledge bases: TREC-COVID (BEIR), PubMedQA, demo handbook."""
from __future__ import annotations

import csv
import gzip
import json
import random
import shutil
import zipfile
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import requests

from .chunking import Document

BEIR_TREC_COVID_ZIP = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/trec-covid.zip"
PUBMEDQA_RAW = "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/"


def download(url: str, dest: Path, timeout: int = 60, chunk: int = 1 << 20) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length") or 0)
        got = 0
        with open(tmp, "wb") as fh:
            for block in r.iter_content(chunk_size=chunk):
                fh.write(block)
                got += len(block)
                if total and got % (20 * chunk) < chunk:
                    print("  %s: %.0f%%" % (dest.name, 100.0 * got / total), flush=True)
    tmp.replace(dest)
    return dest


# ------------------------------------------------------------------ TREC-COVID (BEIR)

def _trec_covid_ready(d: Path) -> bool:
    return (d / "corpus.jsonl").exists() and (d / "queries.jsonl").exists() and (d / "qrels" / "test.tsv").exists()


def ensure_trec_covid(data_dir: Path) -> Path:
    """Download the BEIR release of TREC-COVID (about 171K documents, 50 topics, qrels)."""
    d = Path(data_dir) / "trec-covid"
    if _trec_covid_ready(d):
        return d
    errors = []
    try:
        print("downloading TREC-COVID (BEIR) ...", flush=True)
        zpath = download(BEIR_TREC_COVID_ZIP, Path(data_dir) / "trec-covid.zip")
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(Path(data_dir))
        zpath.unlink()
        if _trec_covid_ready(d):
            return d
        errors.append("BEIR zip did not contain the expected files")
    except Exception as exc:  # network / mirror problems: fall back to the Hugging Face copy
        errors.append("BEIR mirror: %s" % exc)
    try:
        from huggingface_hub import hf_hub_download

        print("falling back to the Hugging Face copy (BeIR/trec-covid) ...", flush=True)
        d.mkdir(parents=True, exist_ok=True)
        (d / "qrels").mkdir(exist_ok=True)
        for name in ("corpus", "queries"):
            gz = hf_hub_download(repo_id="BeIR/trec-covid", filename="%s.jsonl.gz" % name, repo_type="dataset")
            with gzip.open(gz, "rb") as src, open(d / ("%s.jsonl" % name), "wb") as dst:
                shutil.copyfileobj(src, dst)
        qrels = hf_hub_download(repo_id="BeIR/trec-covid-qrels", filename="test.tsv", repo_type="dataset")
        shutil.copy(qrels, d / "qrels" / "test.tsv")
        if _trec_covid_ready(d):
            return d
    except Exception as exc:
        errors.append("Hugging Face: %s" % exc)
    raise RuntimeError(
        "Could not download TREC-COVID automatically (%s).\nDownload %s by hand, unzip it into %s so that "
        "%s/corpus.jsonl exists, and re-run." % ("; ".join(errors), BEIR_TREC_COVID_ZIP, data_dir, d))


def iter_beir_corpus(d: Path) -> Iterator[dict]:
    with open(Path(d) / "corpus.jsonl", "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def load_beir(d: Path, split: str = "test") -> Tuple[List[Document], Dict[str, str], Dict[str, Dict[str, int]]]:
    """Read a BEIR-format dataset folder: corpus.jsonl, queries.jsonl, qrels/<split>.tsv."""
    d = Path(d)
    docs = [Document(doc_id=str(rec["_id"]), title=(rec.get("title") or "").strip(), source=d.name,
                     pages=[(None, rec.get("text") or "")]) for rec in iter_beir_corpus(d)]
    queries: Dict[str, str] = {}
    with open(d / "queries.jsonl", "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rec = json.loads(line)
                queries[str(rec["_id"])] = rec["text"]
    qrels: Dict[str, Dict[str, int]] = {}
    with open(d / "qrels" / ("%s.tsv" % split), "r", encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter="\t")
        for row in reader:
            if len(row) < 3 or not row[2].lstrip("-").isdigit():
                continue                                    # header line
            qrels.setdefault(str(row[0]), {})[str(row[1])] = int(row[2])
    queries = {q: t for q, t in queries.items() if q in qrels}
    return docs, queries, qrels


def split_queries(qids: List[str], n_dev: int, seed: int) -> Tuple[List[str], List[str]]:
    """Deterministic dev / test split of query ids."""
    ordered = sorted(qids, key=lambda q: (len(q), q))
    rng = random.Random(seed)
    dev = set(rng.sample(ordered, min(n_dev, len(ordered))))
    return [q for q in ordered if q in dev], [q for q in ordered if q not in dev]


# --------------------------------------------------------------------------- PubMedQA

def ensure_pubmedqa(data_dir: Path) -> Path:
    d = Path(data_dir) / "pubmedqa"
    for name in ("ori_pqal.json", "test_ground_truth.json"):
        if not (d / name).exists():
            print("downloading PubMedQA %s ..." % name, flush=True)
            download(PUBMEDQA_RAW + name, d / name)
    return d


def load_pubmedqa(data_dir: Path) -> Tuple[Dict[str, dict], List[str]]:
    d = ensure_pubmedqa(data_dir)
    records = json.loads((d / "ori_pqal.json").read_text(encoding="utf-8"))
    test_ids = sorted(json.loads((d / "test_ground_truth.json").read_text(encoding="utf-8")).keys())
    return records, test_ids


def pubmedqa_splits(records: Dict[str, dict], test_ids: List[str], n_test_ans: int = 100, n_test_unans: int = 30,
                    n_dev_ans: int = 40, n_dev_unans: int = 20, seed: int = 13) -> Dict[str, List[str]]:
    """Answerable and out-of-collection question sets.

    The "out-of-collection" questions are real PubMedQA questions whose source abstract
    is withheld from the index, so the collection cannot answer them.
    """
    rng = random.Random(seed)
    test_pool = sorted(test_ids)
    dev_pool = sorted(k for k in records if k not in set(test_ids))
    test_unans = sorted(rng.sample(test_pool, n_test_unans))
    rest = [k for k in test_pool if k not in set(test_unans)]
    test_ans = sorted(rng.sample(rest, n_test_ans))
    dev_unans = sorted(rng.sample(dev_pool, n_dev_unans))
    rest_dev = [k for k in dev_pool if k not in set(dev_unans)]
    dev_ans = sorted(rng.sample(rest_dev, n_dev_ans))
    held_out = set(test_unans) | set(dev_unans)
    return {"test_answerable": test_ans, "test_unanswerable": test_unans, "dev_answerable": dev_ans,
            "dev_unanswerable": dev_unans, "collection": sorted(k for k in records if k not in held_out)}


def pubmedqa_documents(records: Dict[str, dict], ids: List[str]) -> List[Document]:
    """One document per abstract. The conclusion (LONG_ANSWER) is NOT part of the context,
    and the title is left out because PubMedQA questions are the article titles."""
    docs = []
    for pmid in ids:
        rec = records[pmid]
        docs.append(Document(doc_id=pmid, title="", source="PubMed abstract PMID %s" % pmid,
                             pages=[(None, " ".join(rec["CONTEXTS"]))]))
    return docs


def pubmedqa_questions(records: Dict[str, dict], ids: List[str], answerable: bool) -> List[dict]:
    return [{"qid": pmid, "question": records[pmid]["QUESTION"], "reference": records[pmid]["LONG_ANSWER"],
             "label": records[pmid]["final_decision"], "gold_doc": pmid if answerable else None,
             "answerable": answerable} for pmid in ids]


# --------------------------------------------------------------------------- handbook

def load_handbook_questions(data_dir: Path) -> List[dict]:
    path = Path(data_dir) / "handbook" / "questions.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
