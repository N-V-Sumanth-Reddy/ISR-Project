"""Check the NLI model used for citation precision / recall against the judge LLM.

The NLI cross-encoder is trained on sentence-length premises. Given a whole ~200-word passage
it labels nearly every statement "neutral", even one the passage states almost verbatim. This
script takes every cited (statement, passage) pair from the answer evaluation and compares
the NLI decision -- on the whole passage and on windows of 1-4 consecutive sentences -- with
the judge LLM's decision, so the choice of `eval.citation_nli_window` is documented.

Run after scripts/04_eval_answers.py. Writes results/answers/citation_nli_validation.json.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import json
from typing import Dict, List

from ragsearch.eval.citation import is_bare_label, llm_entailment
from ragsearch.eval.metrics import cohen_kappa
from ragsearch.rerank import NLIEntailment, premise_windows
from ragsearch.runtime import base_parser, config_from_args, get_llms, results_dir, save_json

WINDOWS = [0, 1, 2, 3, 4]          # 0 = whole passage


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--benches", default="pubmedqa,handbook")
    args = p.parse_args()
    cfg = config_from_args(args)
    _, judge = get_llms(cfg)
    llm_entails = llm_entailment(judge)
    nli = NLIEntailment(cfg.eval.citation_nli_model, cfg.dense.get("device", "auto"))

    pairs: List[Dict] = []
    for bench in args.benches.split(","):
        path = results_dir(cfg, "answers", bench) / "records.jsonl"
        if not path.exists():
            continue
        for line in path.open(encoding="utf-8"):
            r = json.loads(line)
            if not r.get("citation"):
                continue
            by_n = {q["n"]: q for q in r["passages"]}
            for s in r["citation"]["statements"]:
                if is_bare_label(s["text"]):
                    continue
                for n in s["citations"]:
                    if n in by_n:
                        pairs.append({"bench": bench, "statement": s["text"], "passage": by_n[n]["text"]})
    if not pairs:
        raise SystemExit("no cited statements found; run scripts/04_eval_answers.py first")

    llm = [bool(llm_entails(x["passage"], x["statement"])) for x in pairs]
    n = len(pairs)
    out = {"n_pairs": n, "nli_model": cfg.eval.citation_nli_model, "judge_model": judge.model,
           "configured_window": int(cfg.eval.get("citation_nli_window", 0) or 0),
           "judge_entailed_rate": sum(llm) / n,
           "per_bench": {b: sum(1 for x in pairs if x["bench"] == b) for b in dict.fromkeys(x["bench"] for x in pairs)},
           "windows": {}}
    print("%d cited (statement, passage) pairs; judge LLM (%s) finds %.1f%% entailed" % (n, judge.model, 100 * out["judge_entailed_rate"]))
    print("%-18s %9s %10s %7s" % ("NLI premise", "entailed", "agreement", "kappa"))
    for w in WINDOWS:
        dec = [any(nli.entails_batch([(win, x["statement"]) for win in premise_windows(x["passage"], w)])) for x in pairs]
        agree = sum(a == b for a, b in zip(dec, llm)) / n
        kappa = cohen_kappa(["y" if d else "n" for d in dec], ["y" if d else "n" for d in llm])
        out["windows"][str(w)] = {"entailed_rate": sum(dec) / n, "agreement": agree, "kappa": kappa,
                                  "nli_yes_when_judge_yes": sum(a and b for a, b in zip(dec, llm)) / max(1, sum(llm)),
                                  "nli_yes_when_judge_no": sum(a and not b for a, b in zip(dec, llm)) / max(1, n - sum(llm))}
        print("%-18s %8.1f%% %9.1f%% %7.3f" % ("whole passage" if w == 0 else "%d-sentence windows" % w,
                                               100 * sum(dec) / n, 100 * agree, kappa))
    save_json(results_dir(cfg, "answers") / "citation_nli_validation.json", out)


if __name__ == "__main__":
    main()
