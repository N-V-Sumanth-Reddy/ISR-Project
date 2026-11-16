"""Fixtures: an offline configuration (LSA encoder, stand-in LLM server, temp folders)."""
from __future__ import annotations

import socket
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from ragsearch.config import Config, apply_overrides, load_config  # noqa: E402


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def fake_server():
    import fake_ollama

    port = _free_port()
    server = fake_ollama.serve(port)
    yield "http://127.0.0.1:%d" % port
    server.shutdown()


@pytest.fixture(scope="session")
def cfg(tmp_path_factory, fake_server) -> Config:
    tmp = tmp_path_factory.mktemp("rag")
    base = load_config(str(ROOT / "config.yaml"))
    base.pop("profiles", None)
    return Config(apply_overrides(dict(base), [
        "paths.index_dir=%s" % (tmp / "indexes"), "paths.results_dir=%s" % (tmp / "results"),
        "paths.data_dir=%s" % (ROOT / "data"), "dense.model=lsa-64", "dense.extra_models=[]",
        "llm.host=%s" % fake_server, "llm.model=llama3.1:8b", "llm.judge_model=qwen2.5:7b",
        "rerank.model=none", "eval.citation_backend=llm",
    ]))


@pytest.fixture(scope="session")
def handbook(cfg):
    from ragsearch.collection import Collection
    from ragsearch.ingest import read_folder

    return Collection.build("handbook", read_folder(str(ROOT / "data" / "handbook" / "docs")), cfg, verbose=False)
