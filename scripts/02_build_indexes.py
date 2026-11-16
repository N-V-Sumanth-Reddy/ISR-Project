"""Chunk and index the collections: trec-covid, pubmedqa, handbook, or any folder of documents.

    python scripts/02_build_indexes.py                        # the three evaluation collections
    python scripts/02_build_indexes.py --folder ~/docs --name mydocs
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import json
import random

from ragsearch.collection import Collection
from ragsearch.config import resolve
from ragsearch.datasets import (ensure_trec_covid, load_beir, load_handbook_questions, load_pubmedqa,
                                pubmedqa_documents, pubmedqa_questions, pubmedqa_splits, split_queries)
from ragsearch.ingest import read_folder
from ragsearch.runtime import base_parser, config_from_args, save_json


def dense_models(cfg, extra: bool = True):
    names = [cfg.dense.model] + (list(cfg.dense.get("extra_models") or []) if extra else [])
    return list(dict.fromkeys(names))


def build_trec_covid(cfg, force: bool) -> None:
    if Collection.exists("trec-covid", cfg) and not force:
        print("trec-covid: already built (use --force to rebuild)")
        return
    data_dir = resolve(cfg, "data_dir")
    docs, queries, qrels = load_beir(ensure_trec_covid(data_dir))
    max_docs = cfg.eval.trec_covid.get("max_docs")
    if max_docs and len(docs) > max_docs:            # smoke profile: a small slice that keeps some relevant docs
        rng = random.Random(cfg.eval.seed)
        judged = {d for rel in qrels.values() for d, g in rel.items() if g > 0}
        keep = [d for d in docs if d.doc_id in judged]
        rng.shuffle(keep)
        keep = keep[: max_docs // 3]
        rest = [d for d in docs if d.doc_id not in judged]
        rng.shuffle(rest)
        docs = keep + rest[: max_docs - len(keep)]
        ids = {d.doc_id for d in docs}
        qrels = {q: {d: g for d, g in rel.items() if d in ids} for q, rel in qrels.items()}
    max_queries = cfg.eval.trec_covid.get("max_queries")
    if max_queries:
        keep_q = sorted(queries, key=lambda q: (len(q), q))[: int(max_queries)]
        queries = {q: queries[q] for q in keep_q}
        qrels = {q: qrels[q] for q in keep_q}
    col = Collection.build("trec-covid", docs, cfg, dense_models=dense_models(cfg))
    dev, test = split_queries(list(queries), cfg.eval.trec_covid.dev_queries, cfg.eval.seed)
    save_json(col.dir / "benchmark.json", {"queries": queries, "qrels": qrels, "dev": dev, "test": test})


def build_pubmedqa(cfg, force: bool) -> None:
    if Collection.exists("pubmedqa", cfg) and not force:
        print("pubmedqa: already built (use --force to rebuild)")
        return
    records, test_ids = load_pubmedqa(resolve(cfg, "data_dir"))
    e = cfg.eval.pubmedqa
    splits = pubmedqa_splits(records, test_ids, e.test_answerable, e.test_unanswerable, e.dev_answerable,
                             e.dev_unanswerable, cfg.eval.seed)
    col = Collection.build("pubmedqa", pubmedqa_documents(records, splits["collection"]), cfg, dense_models=dense_models(cfg))
    bench = {
        "test": pubmedqa_questions(records, splits["test_answerable"], True) + pubmedqa_questions(records, splits["test_unanswerable"], False),
        "dev": pubmedqa_questions(records, splits["dev_answerable"], True) + pubmedqa_questions(records, splits["dev_unanswerable"], False),
        "held_out_documents": splits["test_unanswerable"] + splits["dev_unanswerable"],
    }
    save_json(col.dir / "benchmark.json", bench)


def build_handbook(cfg, force: bool) -> None:
    if Collection.exists("handbook", cfg) and not force:
        print("handbook: already built (use --force to rebuild)")
        return
    data_dir = resolve(cfg, "data_dir")
    col = Collection.build("handbook", read_folder(str(data_dir / "handbook" / "docs")), cfg, dense_models=dense_models(cfg, extra=False))
    questions = load_handbook_questions(data_dir)
    limit = cfg.eval.get("handbook_limit")
    if limit:
        ans = [q for q in questions if q["answerable"]][: limit // 2]
        una = [q for q in questions if not q["answerable"]][: limit - len(ans)]
        questions = ans + una
    save_json(col.dir / "benchmark.json", {"test": questions})


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--only", default="trec-covid,pubmedqa,handbook", help="comma-separated collections to build")
    p.add_argument("--folder", default=None, help="index this folder of PDF / text / Markdown files instead")
    p.add_argument("--name", default=None, help="collection name for --folder")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    cfg = config_from_args(args)

    if args.folder:
        name = args.name or "mydocs"
        docs = read_folder(args.folder)
        if not docs:
            raise SystemExit("no readable documents (.pdf .txt .md .html) in %s" % args.folder)
        col = Collection.build(name, docs, cfg, dense_models=dense_models(cfg, extra=False))
        print("collection %r: %d documents, %d passages -> %s" % (name, len(docs), len(col.passages), col.dir))
        return

    builders = {"trec-covid": build_trec_covid, "pubmedqa": build_pubmedqa, "handbook": build_handbook}
    for name in [n.strip() for n in args.only.split(",") if n.strip()]:
        if name not in builders:
            raise SystemExit("unknown collection %r" % name)
        builders[name](cfg, args.force)


if __name__ == "__main__":
    main()
