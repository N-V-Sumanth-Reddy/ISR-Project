"""Build results/RESULTS.md, the LaTeX tables and macros in report/generated/, and the figures.

Run it any time; it only reads the files under results/ and never calls a model.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
from pathlib import Path

from ragsearch.config import ROOT, resolve
from ragsearch.reporting import write_all
from ragsearch.runtime import base_parser, config_from_args


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--report-dir", default=None, help="where to write generated/ and figures/ (default: report/)")
    args = p.parse_args()
    cfg = config_from_args(args)
    results = resolve(cfg, "results_dir")
    if args.report_dir:
        report_dir = Path(args.report_dir)
    else:                               # only the standard / full profiles feed the paper
        report_dir = (ROOT / "report") if cfg.get("profile") != "smoke" else (results / "report_preview")
    out = write_all(results, report_dir)
    print("wrote %s" % (results / "RESULTS.md"))
    print("tables: %s" % ", ".join(out["tables"]))
    print("figures: %s" % ", ".join(out["figures"]))
    for c in out["criteria"]:
        print("  [%s] %s -- %s" % (c["outcome"], c["criterion"], c["evidence"]))


if __name__ == "__main__":
    main()
