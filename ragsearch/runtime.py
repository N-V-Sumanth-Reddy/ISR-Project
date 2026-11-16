"""Shared helpers for the command-line scripts."""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .config import Config, deep_merge, load_config, resolve
from .llm import LLM, LLMCache, LLMError, make_llm


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default=None, help="path to config.yaml")
    p.add_argument("--profile", default="standard", help="smoke | standard | full")
    p.add_argument("--set", action="append", default=[], metavar="section.key=value", help="override a config value")
    return p


def config_from_args(args) -> Config:
    cfg = load_config(args.config)
    profiles = cfg.get("profiles", {}) or {}
    if args.profile not in profiles:
        raise SystemExit("unknown profile %r (choose from %s)" % (args.profile, ", ".join(profiles)))
    merged = deep_merge(dict(cfg), profiles[args.profile] or {})
    from .config import apply_overrides

    cfg = Config(apply_overrides(merged, args.set))
    cfg["profile"] = args.profile
    return cfg


def results_dir(cfg: Config, *parts: str) -> Path:
    d = resolve(cfg, "results_dir").joinpath(*parts)
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_json(path: Path, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=str))


def load_json(path: Path, default: Any = None) -> Any:
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def get_llms(cfg: Config) -> Tuple[LLM, LLM]:
    """Generator and judge clients sharing one on-disk cache."""
    cache = LLMCache(str(results_dir(cfg, "cache") / "llm_cache.sqlite"))
    return make_llm(cfg, cfg.llm.model, cache), make_llm(cfg, cfg.llm.get("judge_model") or cfg.llm.model, cache)


def require_llm(*llms: LLM) -> None:
    for llm in llms:
        if not llm.is_up():
            raise SystemExit(
                "\nThe local LLM server is not reachable at %s.\n"
                "Start Ollama (open the Ollama app, or run `ollama serve`) and re-run this command.\n" % getattr(llm, "host", "?"))
        names = llm.available_models()
        short = {n.split(":")[0] for n in names if n}
        if names and llm.model not in names and llm.model not in short and (llm.model + ":latest") not in names:
            raise SystemExit("\nModel %r is not installed in Ollama. Run:  ollama pull %s\n(installed: %s)\n"
                             % (llm.model, llm.model, ", ".join(sorted(n for n in names if n)) or "none"))


def make_entails(cfg: Config, judge: LLM):
    """Entailment function for citation checking: NLI cross-encoder, or the judge LLM."""
    from .eval.citation import llm_entailment

    backend = cfg.eval.get("citation_backend", "nli")
    if backend == "nli":
        try:
            from .rerank import NLIEntailment

            nli, last = None, None
            for device in dict.fromkeys([cfg.dense.get("device", "auto"), "cpu"]):
                try:                                   # load, then run one pair to be sure the device supports the model
                    cand = NLIEntailment(cfg.eval.citation_nli_model, device,
                                         window_sentences=int(cfg.eval.get("citation_nli_window", 0) or 0))
                    cand.entails("The clinic opens at nine.", "The clinic opens in the morning.")
                    nli = cand
                    break
                except Exception as exc:
                    last = exc
            if nli is None:
                raise last if last is not None else RuntimeError("NLI model could not be loaded")

            def nli_entails(premise: str, hypothesis: str) -> bool:
                return nli.entails(premise, hypothesis)

            nli_entails.backend_name = "nli:" + cfg.eval.citation_nli_model   # type: ignore[attr-defined]
            nli_entails.nli_window = nli.window_sentences                     # type: ignore[attr-defined]
            return nli_entails
        except Exception as exc:
            print("  (NLI model unavailable, using the judge LLM for entailment: %s)" % str(exc)[:160], flush=True)
    fn = llm_entailment(judge)
    fn.backend_name = "llm:" + judge.model                                   # type: ignore[attr-defined]
    return fn


def environment() -> Dict[str, Any]:
    info: Dict[str, Any] = {"python": sys.version.split()[0], "platform": platform.platform(),
                            "machine": platform.machine(), "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    from importlib import metadata

    # read versions from package metadata: importing faiss and torch into the same process
    # is exactly what the environment check wants to probe in a subprocess first
    for mod, dist in (("numpy", "numpy"), ("scipy", "scipy"), ("torch", "torch"), ("sentence_transformers", "sentence-transformers"),
                      ("transformers", "transformers"), ("faiss", "faiss-cpu"), ("streamlit", "streamlit")):
        try:
            info[mod] = metadata.version(dist)
        except Exception:
            info[mod] = None
    try:
        import os

        info["cpu_count"] = os.cpu_count()
        if hasattr(os, "sysconf") and "SC_PHYS_PAGES" in os.sysconf_names:
            info["ram_gb"] = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2 ** 30, 1)
    except Exception:
        pass
    try:
        from .dense import pick_device

        info["torch_device"] = pick_device("auto")
    except Exception:
        info["torch_device"] = None
    return info


class Timer:
    def __init__(self, label: str):
        self.label = label

    def __enter__(self):
        self.t0 = time.time()
        print("\n=== %s ===" % self.label, flush=True)
        return self

    def __exit__(self, *exc):
        self.seconds = time.time() - self.t0
        print("--- %s: %.1f s" % (self.label, self.seconds), flush=True)
        return False
