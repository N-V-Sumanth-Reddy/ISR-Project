"""Score the human validation of the LLM judge.

1. Open results/answers/pubmedqa/judge_handcheck.csv (50 sampled comparisons; the two
   answers are in random order and unlabeled).
2. In the `human_verdict` column write 1, 2 or tie: which answer agrees better with the
   reference answer?
3. Run this script. It reports agreement and Cohen's kappa between the human verdicts and
   the LLM judge, and stores them for the report.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import csv
import json

from ragsearch.eval.metrics import cohen_kappa
from ragsearch.runtime import base_parser, config_from_args, results_dir, save_json


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--bench", default="pubmedqa")
    args = p.parse_args()
    cfg = config_from_args(args)
    d = results_dir(cfg, "answers", args.bench)
    key = json.loads((d / "judge_handcheck_key.json").read_text())
    human, judge = [], []
    with open(d / "judge_handcheck.csv", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            v = (row.get("human_verdict") or "").strip().lower()
            if v not in ("1", "2", "tie"):
                continue
            k = key[row["qid"]]
            human.append("tie" if v == "tie" else k[v])
            judge.append(k["judge"])
    if not human:
        raise SystemExit("no verdicts found: fill in the human_verdict column with 1, 2 or tie")
    agree = sum(h == j for h, j in zip(human, judge)) / len(human)
    out = {"n": len(human), "agreement": agree, "kappa": cohen_kappa(human, judge),
           "human": {v: human.count(v) for v in ("rag", "closed_book", "tie")},
           "judge": {v: judge.count(v) for v in ("rag", "closed_book", "tie")}}
    save_json(d / "judge_handcheck_agreement.json", out)
    print("checked %d judgments: agreement %.1f%%, Cohen's kappa %.2f" % (out["n"], 100 * agree, out["kappa"]))
    print("human:", out["human"], "| judge:", out["judge"])


if __name__ == "__main__":
    main()
