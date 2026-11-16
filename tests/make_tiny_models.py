"""Create tiny randomly initialised models so the real sentence-transformers code paths
(bi-encoder, cross-encoder, NLI) can be exercised offline. They produce meaningless
scores; they exist only to test the plumbing."""
from __future__ import annotations

import sys
from pathlib import Path


def make(out_dir: str) -> dict:
    import torch
    from transformers import BertConfig, BertForSequenceClassification, BertModel, BertTokenizerFast

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    words = ("the of and to in a is for with on by study patients results were was that this as are from "
             "policy leave travel student employee data lab safety days must may within request approval "
             "covid virus treatment effect risk group control trial health care rate use yes no maybe").split()
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + sorted(set(words)) + list("abcdefghijklmnopqrstuvwxyz0123456789") \
        + ["##" + c for c in "abcdefghijklmnopqrstuvwxyz0123456789"]
    vocab_file = out / "vocab.txt"
    vocab_file.write_text("\n".join(vocab))
    tok = BertTokenizerFast(vocab_file=str(vocab_file), do_lower_case=True)
    cfg = dict(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
               intermediate_size=64, max_position_embeddings=128)
    paths = {}

    bi = out / "tiny-bi-encoder"
    BertModel(BertConfig(**cfg)).save_pretrained(bi)
    tok.save_pretrained(bi)
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.models import Pooling, Transformer
    tr = Transformer(str(bi), max_seq_length=64)
    dim_fn = getattr(tr, "get_embedding_dimension", None) or tr.get_word_embedding_dimension
    st = SentenceTransformer(modules=[tr, Pooling(dim_fn(), pooling_mode="mean")])
    st.save(str(out / "tiny-st"))
    paths["bi"] = str(out / "tiny-st")

    for name, labels in (("tiny-cross-encoder", 1), ("tiny-nli", 3)):
        c = BertConfig(num_labels=labels, **cfg)
        if labels == 3:
            c.id2label = {0: "contradiction", 1: "entailment", 2: "neutral"}
            c.label2id = {v: k for k, v in c.id2label.items()}
        BertForSequenceClassification(c).save_pretrained(out / name)
        tok.save_pretrained(out / name)
        paths["ce" if labels == 1 else "nli"] = str(out / name)
    return paths


if __name__ == "__main__":
    print(make(sys.argv[1] if len(sys.argv) > 1 else "tests/_tiny_models"))
