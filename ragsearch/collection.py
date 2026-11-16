"""A document collection: passages + BM25 index + dense indexes (+ optional entity graph)."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from .bm25 import BM25Index
from .chunking import Document, Passage, chunk_documents
from .config import Config, resolve
from .dense import DenseIndex, Encoder, LSAEncoder, encode_in_batches, get_encoder, model_key
from .kg import EntityGraph


def _is_lsa(name: str) -> bool:
    return name.lower().startswith("lsa")


class Collection:
    def __init__(self, name: str, directory: Path, cfg: Config):
        self.name = name
        self.dir = Path(directory)
        self.cfg = cfg
        self.passages: List[Passage] = []
        self.bm25: Optional[BM25Index] = None
        self.dense: Dict[str, DenseIndex] = {}        # model name -> index
        self.kg: Optional[EntityGraph] = None
        self.meta: Dict = {}
        self._encoders: Dict[str, Encoder] = {}
        self._doc_ids: Optional[List[str]] = None

    # -------------------------------------------------------------- accessors
    @property
    def doc_ids(self) -> List[str]:
        if self._doc_ids is None:
            self._doc_ids = [p.doc_id for p in self.passages]
        return self._doc_ids

    @property
    def dense_models(self) -> List[str]:
        return list(self.dense.keys())

    def default_dense_model(self) -> Optional[str]:
        preferred = self.cfg.dense.model
        if preferred in self.dense:
            return preferred
        return next(iter(self.dense), None)

    def encoder(self, model_name: str) -> Encoder:
        if model_name not in self._encoders:
            if _is_lsa(model_name):
                self._encoders[model_name] = LSAEncoder.load(str(self.dir / ("encoder_%s.pkl" % model_key(model_name))))
            else:
                d = self.cfg.dense
                self._encoders[model_name] = get_encoder(model_name, d.get("device", "auto"),
                                                         d.get("batch_size", 64), d.get("max_seq_length", 256))
        return self._encoders[model_name]

    # ------------------------------------------------------------------ build
    @classmethod
    def build(cls, name: str, documents: Iterable[Document], cfg: Config, index_dir: Optional[Path] = None,
              dense_models: Optional[Sequence[str]] = None, build_kg: bool = True,
              chunk_words: Optional[int] = None, overlap_words: Optional[int] = None,
              verbose: bool = True) -> "Collection":
        t0 = time.time()
        directory = Path(index_dir) if index_dir else resolve(cfg, "index_dir") / name
        directory.mkdir(parents=True, exist_ok=True)
        col = cls(name, directory, cfg)
        cw = int(chunk_words or cfg.chunking.chunk_words)
        ow = int(cfg.chunking.overlap_words if overlap_words is None else overlap_words)
        documents = list(documents)
        col.passages = chunk_documents(documents, cw, ow)
        if verbose:
            print("[%s] %d documents -> %d passages (%d words, %d overlap)" % (name, len(documents), len(col.passages), cw, ow), flush=True)
        texts = [p.index_text for p in col.passages]

        col.bm25 = BM25Index(k1=cfg.bm25.k1, b=cfg.bm25.b, stem=cfg.bm25.stem, stopwords=cfg.bm25.stopwords).build(texts)
        if verbose:
            print("[%s] BM25 index: %d terms, avg length %.1f tokens (%.0fs)" % (name, len(col.bm25.vocab), col.bm25.avgdl, time.time() - t0), flush=True)

        col.meta = {"name": name, "n_documents": len(documents), "n_passages": len(col.passages),
                    "chunk_words": cw, "overlap_words": ow, "dense_models": [], "built_at": time.strftime("%Y-%m-%d %H:%M:%S")}
        col._save_core()

        if dense_models is None:
            dense_models = [cfg.dense.model]
        for model_name in dense_models:
            col.add_dense(model_name, texts=texts, verbose=verbose)

        if build_kg and len(col.passages) > 0:
            t1 = time.time()
            k = cfg.get("kg", {})
            col.kg = EntityGraph(k.get("max_entities_per_passage", 12), k.get("min_df", 2), k.get("max_df_ratio", 0.05)).build(texts)
            col.kg.save(str(col.dir))
            col.meta["kg"] = col.kg.stats()
            if verbose:
                print("[%s] entity graph: %d entities, %d passage-entity edges (%.0fs)" % (name, len(col.kg.entities), col.kg.A.nnz, time.time() - t1), flush=True)
        col.meta["build_seconds"] = round(time.time() - t0, 1)
        col._save_meta()
        return col

    def add_dense(self, model_name: str, texts: Optional[List[str]] = None, verbose: bool = True) -> None:
        """Encode all passages with ``model_name`` and store the embeddings."""
        t0 = time.time()
        texts = texts if texts is not None else [p.index_text for p in self.passages]
        path = self.dir / ("dense_%s.npy" % model_key(model_name))
        if _is_lsa(model_name):
            dim = int(model_name.split("-")[1]) if "-" in model_name else 256
            enc = LSAEncoder(dim=dim).fit(texts)
            enc.save(str(self.dir / ("encoder_%s.pkl" % model_key(model_name))))
            self._encoders[model_name] = enc
            emb = enc.encode_passages(texts)
        else:
            enc = self.encoder(model_name)
            if verbose:
                print("[%s] encoding %d passages with %s on %s ..." % (self.name, len(texts), model_name, getattr(enc, "device", "cpu")), flush=True)
            emb = encode_in_batches(enc, texts, show_progress=verbose, checkpoint=str(path) + ".partial.npy")
        np.save(path, emb)
        self.dense[model_name] = DenseIndex(emb, backend=self.cfg.dense.get("backend", "auto"))
        if model_name not in self.meta.setdefault("dense_models", []):
            self.meta["dense_models"].append(model_name)
        self.meta.setdefault("dense_seconds", {})[model_name] = round(time.time() - t0, 1)
        self._save_meta()
        if verbose:
            print("[%s] dense index %s: %d x %d (%.0fs, %s search)" % (self.name, model_name, emb.shape[0], emb.shape[1] if emb.ndim == 2 else 0,
                                                                     time.time() - t0, self.dense[model_name].backend), flush=True)

    # ------------------------------------------------------------------- disk
    def _save_meta(self) -> None:
        (self.dir / "meta.json").write_text(json.dumps(self.meta, indent=2))

    def _save_core(self) -> None:
        with open(self.dir / "passages.jsonl", "w", encoding="utf-8") as fh:
            for p in self.passages:
                fh.write(json.dumps(p.to_dict(), ensure_ascii=False) + "\n")
        self.bm25.save(str(self.dir))
        self._save_meta()

    @classmethod
    def load(cls, name: str, cfg: Config, index_dir: Optional[Path] = None,
             dense_models: Optional[Sequence[str]] = None, load_kg: bool = True) -> "Collection":
        directory = Path(index_dir) if index_dir else resolve(cfg, "index_dir") / name
        if not (directory / "meta.json").exists():
            raise FileNotFoundError("collection %r has not been built (%s)" % (name, directory))
        col = cls(name, directory, cfg)
        col.meta = json.loads((directory / "meta.json").read_text())
        with open(directory / "passages.jsonl", "r", encoding="utf-8") as fh:
            col.passages = [Passage.from_dict(json.loads(line)) for line in fh if line.strip()]
        col.bm25 = BM25Index.load(str(directory))
        wanted = col.meta.get("dense_models", []) if dense_models is None else list(dense_models)
        for model_name in wanted:
            path = directory / ("dense_%s.npy" % model_key(model_name))
            if path.exists():
                col.dense[model_name] = DenseIndex.load(str(path), backend=cfg.dense.get("backend", "auto"))
        if load_kg and (directory / "kg_matrix.npz").exists():
            col.kg = EntityGraph.load(str(directory))
        return col

    @staticmethod
    def exists(name: str, cfg: Config, index_dir: Optional[Path] = None) -> bool:
        directory = Path(index_dir) if index_dir else resolve(cfg, "index_dir") / name
        return (directory / "meta.json").exists()


def list_collections(cfg: Config) -> List[str]:
    root = resolve(cfg, "index_dir")
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if (p / "meta.json").exists())
