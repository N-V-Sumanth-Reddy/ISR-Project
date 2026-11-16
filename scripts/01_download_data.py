"""Download the public datasets and the retrieval / NLI models (the only step that needs the internet)."""
from __future__ import annotations

import _bootstrap  # noqa: F401

from ragsearch.config import resolve
from ragsearch.datasets import ensure_pubmedqa, ensure_trec_covid
from ragsearch.runtime import base_parser, config_from_args


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--skip-trec-covid", action="store_true")
    p.add_argument("--skip-models", action="store_true")
    args = p.parse_args()
    cfg = config_from_args(args)
    data_dir = resolve(cfg, "data_dir")

    d = ensure_pubmedqa(data_dir)
    print("PubMedQA ready:", d)
    if not args.skip_trec_covid:
        d = ensure_trec_covid(data_dir)
        print("TREC-COVID ready:", d)

    if args.skip_models:
        return
    from ragsearch.dense import get_encoder
    from ragsearch.rerank import NLIEntailment, get_reranker

    for name in [cfg.dense.model] + list(cfg.dense.get("extra_models") or []):
        if name.lower().startswith("lsa"):               # classical baseline, nothing to download
            continue
        enc = get_encoder(name, cfg.dense.get("device", "auto"), cfg.dense.batch_size, cfg.dense.max_seq_length)
        print("bi-encoder ready: %s (%d-d, device %s)" % (name, enc.dim, enc.device))
    get_reranker(cfg.rerank.model, cfg.dense.get("device", "auto"))
    print("cross-encoder ready:", cfg.rerank.model)
    if cfg.eval.get("citation_backend", "nli") == "nli":
        try:
            NLIEntailment(cfg.eval.citation_nli_model, cfg.dense.get("device", "auto"))
            print("NLI model ready:", cfg.eval.citation_nli_model)
        except Exception as exc:
            print("NLI model could not be loaded (%s); citation checks will use the judge LLM instead." % str(exc)[:200])


if __name__ == "__main__":
    main()
