"""Cross-encoder re-ranking and NLI entailment (both via sentence-transformers CrossEncoder)."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .dense import pick_device
from .textproc import split_sentences

_CE_CACHE: Dict[Tuple[str, str], "CrossEncoderReranker"] = {}


def load_cross_encoder(name: str, device: str, max_length: int = 512):
    """Load a CrossEncoder and cap its input length at ``max_length`` tokens.

    The cap is applied after loading so it never exceeds what the model supports
    (the attribute is `max_seq_length` in recent sentence-transformers, `max_length` before).
    """
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(name, device=device)
    for attr in ("max_seq_length", "max_length"):
        try:
            current = getattr(model, attr)
        except Exception:
            continue
        if current is None or int(current) > max_length:
            try:
                setattr(model, attr, max_length)
            except Exception:
                pass
        break
    return model


class CrossEncoderReranker:
    def __init__(self, name: str, device: str = "auto", batch_size: int = 32, max_length: int = 512):
        self.name = name
        self.device = pick_device(device)
        self.batch_size = batch_size
        self.model = load_cross_encoder(name, self.device, max_length)

    def score(self, query: str, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros(0, dtype=np.float32)
        pairs = [(query, t) for t in texts]
        scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        return np.asarray(scores, dtype=np.float32).reshape(len(texts), -1)[:, 0]


def get_reranker(name: str, device: str = "auto") -> CrossEncoderReranker:
    if not name or str(name).lower() in ("none", "off", "false"):
        raise RuntimeError("no re-ranker configured (rerank.model is %r)" % name)
    key = (name, device)
    if key not in _CE_CACHE:
        _CE_CACHE[key] = CrossEncoderReranker(name, device=device)
    return _CE_CACHE[key]


class NLIEntailment:
    """premise entails hypothesis?  Uses a 3-way NLI cross-encoder.

    NLI cross-encoders are trained on sentence-length premises (MNLI, SNLI) and label almost any
    passage-length premise "neutral", even when it contains the hypothesis nearly verbatim. With
    ``window_sentences = w > 0`` the premise is therefore split into windows of w consecutive
    sentences (stride 1) and the hypothesis counts as entailed if any window entails it, as in
    SummaC-ZS (Laban et al., 2022). ``w = 0`` scores the whole premise at once.
    """

    def __init__(self, name: str = "cross-encoder/nli-deberta-v3-base", device: str = "auto", batch_size: int = 16,
                 window_sentences: int = 0):
        self.name = name
        self.batch_size = batch_size
        self.window_sentences = int(window_sentences or 0)
        self.model = load_cross_encoder(name, pick_device(device), 512)
        hf_model = getattr(self.model, "model", self.model)
        id2label = getattr(getattr(hf_model, "config", None), "id2label", None) or {}
        labels = {int(i): str(lbl).lower() for i, lbl in id2label.items()}
        self.entail_idx = next((i for i, lbl in labels.items() if lbl.startswith("entail")), 1)
        self._memo: Dict[Tuple[str, str], bool] = {}

    def entails_batch(self, pairs: Sequence[Tuple[str, str]]) -> List[bool]:
        todo = [p for p in dict.fromkeys(pairs) if p not in self._memo]
        if todo:
            logits = np.asarray(self.model.predict(list(todo), batch_size=self.batch_size, show_progress_bar=False))
            logits = logits.reshape(len(todo), -1)
            for pair, row in zip(todo, logits):
                self._memo[pair] = bool(int(np.argmax(row)) == self.entail_idx)
        return [self._memo[p] for p in pairs]

    def entails(self, premise: str, hypothesis: str) -> bool:
        windows = premise_windows(premise, self.window_sentences)
        return any(self.entails_batch([(w, hypothesis) for w in windows]))


def premise_windows(premise: str, window_sentences: int) -> List[str]:
    """Windows of ``window_sentences`` consecutive sentences (stride 1); the whole premise if 0 or short."""
    if not window_sentences or window_sentences <= 0:
        return [premise]
    sents = split_sentences(premise)
    if len(sents) <= window_sentences:
        return [premise]
    return [" ".join(sents[i:i + window_sentences]) for i in range(len(sents) - window_sentences + 1)]
