"""Ask one question from the command line.

    python scripts/ask.py "What is the protocol for a missed insulin dose?" --collection handbook
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

from ragsearch.collection import Collection
from ragsearch.generate import answer_closed_book, answer_rag
from ragsearch.retriever import Retriever
from ragsearch.runtime import base_parser, config_from_args, get_llms, require_llm


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("question")
    p.add_argument("--collection", default="handbook")
    p.add_argument("--mode", default=None, choices=["bm25", "dense", "hybrid"])
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--rerank", action="store_true")
    p.add_argument("--kg", action="store_true")
    p.add_argument("--closed-book", action="store_true", help="also print the closed-book answer")
    p.add_argument("--retrieve-only", action="store_true")
    args = p.parse_args()
    cfg = config_from_args(args)
    col = Collection.load(args.collection, cfg)
    res = Retriever(col, cfg).search(args.question, mode=args.mode, top_k=args.top_k, rerank=args.rerank, use_kg=args.kg)
    if not args.retrieve_only:
        gen, _ = get_llms(cfg)
        require_llm(gen)
        ans = answer_rag(args.question, res, gen, signal=cfg.abstention.get("signal"), threshold=cfg.abstention.get("threshold"))
        print("\nANSWER (%s, %s, top-%d):\n  %s" % (gen.model, res.mode, len(res.passages), ans.text))
        if ans.invalid_citations or ans.n_uncited_sentences:
            print("  [citation check] invalid: %s, uncited sentences: %d" % (ans.invalid_citations, ans.n_uncited_sentences))
        if args.closed_book:
            print("\nCLOSED-BOOK:\n  %s" % answer_closed_book(args.question, gen).text)
    print("\nEVIDENCE:")
    for i, (psg, hit) in enumerate(zip(res.passages, res.hits), start=1):
        print("  [%d] (%s; BM25 rank %s, dense rank %s) %s\n      %s" % (
            i, hit.found_by, hit.bm25_rank, hit.dense_rank, psg.location(), psg.text[:240].replace("\n", " ")))


if __name__ == "__main__":
    main()
