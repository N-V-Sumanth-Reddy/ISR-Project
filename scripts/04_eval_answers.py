"""Answer evaluation: closed-book LLM vs. the RAG system.

PubMedQA: ~100 answerable test questions + ~30 whose source abstract is withheld from the
collection. Reports the pairwise LLM-judge comparison against the ground-truth answer,
yes/no/maybe accuracy, ALCE-style citation quality, RAGAs-style faithfulness, abstention
accuracy and over-refusal, and a failure categorisation. top-k and the abstention
threshold are tuned on the dev questions only.

Handbook: the hand-written demo questions, using the settings tuned on PubMedQA.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

from ragsearch.collection import Collection
from ragsearch.eval import answers
from ragsearch.generate import YESNO_STYLE
from ragsearch.runtime import (Timer, base_parser, config_from_args, get_llms, load_json, make_entails,
                               require_llm, results_dir, save_json)


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--only", default="pubmedqa,handbook")
    p.add_argument("--no-judge", action="store_true")
    p.add_argument("--no-faithfulness", action="store_true")
    p.add_argument("--no-citation", action="store_true")
    args = p.parse_args()
    cfg = config_from_args(args)
    gen, judge = get_llms(cfg)
    require_llm(gen, judge)
    print("generator: %s | judge: %s" % (gen.model, judge.model))
    entails = None if args.no_citation else make_entails(cfg, judge)
    names = [n.strip() for n in args.only.split(",") if n.strip()]
    common = dict(do_judge=not args.no_judge, do_citation=not args.no_citation,
                  do_faithfulness=not args.no_faithfulness, entails=entails)

    if "pubmedqa" in names and Collection.exists("pubmedqa", cfg):
        col = Collection.load("pubmedqa", cfg)
        b = load_json(col.dir / "benchmark.json")
        bench = {"name": "pubmedqa", "collection": col, "test": b["test"], "dev": b["dev"], "style": YESNO_STYLE}
        with Timer("answer evaluation: pubmedqa"):
            summary = answers.run(bench, cfg, gen, judge, results_dir(cfg, "answers", "pubmedqa"), **common)
        print_summary(summary)

    if "handbook" in names and Collection.exists("handbook", cfg):
        col = Collection.load("handbook", cfg)
        b = load_json(col.dir / "benchmark.json")
        transfer = load_json(results_dir(cfg, "answers", "pubmedqa") / "dev_tuning.json")
        bench = {"name": "handbook", "collection": col, "test": b["test"], "dev": None, "style": ""}
        with Timer("answer evaluation: handbook"):
            summary = answers.run(bench, cfg, gen, judge, results_dir(cfg, "answers", "handbook"), transfer=transfer, **common)
        print_summary(summary)

    save_json(results_dir(cfg, "answers") / "llm_usage.json", {
        "generator": {"model": gen.model, "calls": gen.calls, "cache_hits": gen.cache_hits, "seconds": gen.total_seconds},
        "judge": {"model": judge.model, "calls": judge.calls, "cache_hits": judge.cache_hits, "seconds": judge.total_seconds}})


def print_summary(s) -> None:
    print("\n[%s] top-k=%s, policy=%s" % (s["bench"], s["top_k"], s["primary_policy"]))
    if "judge" in s:
        j = s["judge"]
        print("  pairwise judge: RAG wins %d, closed-book wins %d, ties %d (n=%d, sign test p=%.4f)" % (
            j["rag_wins"], j["closed_book_wins"], j["ties"], j["n"], j["sign_test_p"]))
    if "decision_accuracy" in s:
        d = s["decision_accuracy"]
        print("  yes/no/maybe accuracy: closed-book %.3f, RAG %.3f (coverage %.2f)" % (
            d["closed_book"]["accuracy"], d["rag"]["accuracy"], d["rag"]["coverage"]))
    if "citation" in s:
        c = s["citation"]
        print("  citation precision %s, recall %s" % (c["citation_precision_micro"], c["citation_recall_micro"]))
    if "faithfulness" in s:
        print("  faithfulness %s" % s["faithfulness"]["mean"])
    a = s["abstention"][s["primary_policy"]]
    print("  abstention accuracy %s, over-refusal %s" % (a["abstention_accuracy"], a["over_refusal_rate"]))


if __name__ == "__main__":
    main()
