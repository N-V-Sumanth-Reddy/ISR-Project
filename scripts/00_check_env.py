"""Check that this machine can run the pipeline, and record the environment.

Writes config.local.yaml when a machine-specific setting is needed:
  * dense.backend: numpy     if FAISS and PyTorch cannot be loaded together here
  * smaller LLMs             if the machine has too little memory for 7-8B models
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import shutil
import subprocess
import sys

import yaml

from ragsearch.config import ROOT, resolve
from ragsearch.runtime import base_parser, config_from_args, environment, results_dir, save_json

FAISS_PROBE = """
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import numpy as np, torch, faiss
x = np.random.RandomState(0).rand(2000, 384).astype('float32')
idx = faiss.IndexFlatIP(384); idx.add(x)
t = torch.nn.functional.normalize(torch.from_numpy(x[:8]), dim=1)
print(idx.search(t.numpy(), 5)[1].shape)
"""


def main() -> int:
    p = base_parser(__doc__)
    p.add_argument("--pull", action="store_true", help="download missing Ollama models")
    args = p.parse_args()
    cfg = config_from_args(args)
    env = environment()
    local = {}
    problems = []
    print("Python %s on %s (%s), %s cores, %s GB RAM" % (env["python"], env["platform"], env["machine"],
                                                        env.get("cpu_count"), env.get("ram_gb")))
    if sys.version_info < (3, 9):
        problems.append("Python 3.9 or newer is required")
    for mod in ("numpy", "scipy", "torch", "sentence_transformers"):
        if not env.get(mod):
            problems.append("missing Python package: %s (run: pip install -r requirements.txt)" % mod)
    print("PyTorch %s, device: %s | sentence-transformers %s | faiss %s" % (
        env.get("torch"), env.get("torch_device"), env.get("sentence_transformers"), env.get("faiss")))

    # FAISS and PyTorch each bundle OpenMP; on some macOS setups loading both crashes.
    backend = "numpy"
    if env.get("faiss") and env.get("torch"):
        probe = subprocess.run([sys.executable, "-c", FAISS_PROBE], capture_output=True, text=True, timeout=300)
        backend = "faiss" if probe.returncode == 0 else "numpy"
        if probe.returncode != 0:
            print("FAISS + PyTorch probe failed (exit %d); using exact numpy search instead (same results)." % probe.returncode)
            local.setdefault("dense", {})["backend"] = "numpy"
    env["dense_backend"] = backend
    print("dense search backend: %s" % backend)

    # Local LLM
    ram = env.get("ram_gb") or 0
    gen, judge = cfg.llm.model, cfg.llm.get("judge_model") or cfg.llm.model
    if ram and ram < 14 and "llm.model" not in " ".join(args.set):
        gen, judge = "llama3.2:3b", "qwen2.5:3b"
        local["llm"] = {"model": gen, "judge_model": judge}
        print("NOTE: %.0f GB RAM is tight for 7-8B models; using %s (generator) and %s (judge) instead." % (ram, gen, judge))
    env["generator"], env["judge"] = gen, judge
    if cfg.llm.get("backend", "ollama") == "ollama":
        from ragsearch.llm import _HTTP as requests      # local session that ignores proxy settings

        try:
            tags = requests.get(cfg.llm.host.rstrip("/") + "/api/tags", timeout=5).json()
            installed = [m.get("name") for m in tags.get("models", [])]
            env["ollama_models"] = installed
            print("Ollama is running; installed models: %s" % (", ".join(installed) or "none"))
            for model in dict.fromkeys([gen, judge]):
                if model not in installed and model + ":latest" not in installed:
                    if args.pull and shutil.which("ollama"):
                        print("pulling %s (several GB the first time) ..." % model, flush=True)
                        if subprocess.call(["ollama", "pull", model]) != 0:
                            problems.append("could not pull Ollama model %s" % model)
                    elif args.pull:                       # no CLI on PATH: use the HTTP API
                        print("pulling %s through the Ollama API (several GB, no progress bar) ..." % model, flush=True)
                        r = requests.post(cfg.llm.host.rstrip("/") + "/api/pull", json={"model": model, "stream": False}, timeout=7200)
                        if r.status_code != 200:
                            problems.append("could not pull Ollama model %s: %s" % (model, r.text[:200]))
                    else:
                        problems.append("Ollama model %s is not installed (run: ollama pull %s)" % (model, model))
        except Exception as exc:
            problems.append("Ollama is not reachable at %s (%s). Install it from https://ollama.com and start it."
                            % (cfg.llm.host, str(exc)[:80]))

    free_gb = shutil.disk_usage(str(ROOT)).free / 2 ** 30
    env["disk_free_gb"] = round(free_gb, 1)
    if free_gb < 15:
        problems.append("only %.0f GB of free disk space; the models, data and indexes need about 15 GB" % free_gb)

    local_path = ROOT / "config.local.yaml"
    if local:
        existing = yaml.safe_load(local_path.read_text()) if local_path.exists() else {}
        from ragsearch.config import deep_merge

        local_path.write_text(yaml.safe_dump(deep_merge(existing or {}, local), sort_keys=False))
        print("wrote %s: %s" % (local_path.name, local))
    save_json(results_dir(cfg) / "env.json", env)

    if problems:
        print("\nPROBLEMS:")
        for prob in problems:
            print("  - " + prob)
        return 1
    print("\nEnvironment OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
