"""Web UI for the private, citation-grounded RAG search engine.

    streamlit run app/streamlit_app.py

Everything runs on this machine: the indexes are local files, the retrieval models run
in this process, and the LLM is served by Ollama on localhost. The settings panel
switches the retrieval mode, top-k and re-ranking, so the UI doubles as the experimental
interface.
"""
from __future__ import annotations

import html
import json
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The retrieval models were downloaded by scripts/01_download_data.py. In the app we
# forbid the Hugging Face libraries from touching the network at all.
if os.environ.get("RAG_ALLOW_DOWNLOADS") != "1":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import streamlit as st  # noqa: E402

from ragsearch.collection import Collection, list_collections  # noqa: E402
from ragsearch.config import Config, apply_overrides, deep_merge, load_config, resolve  # noqa: E402
from ragsearch.generate import NOT_FOUND, answer_closed_book, answer_rag  # noqa: E402
from ragsearch.ingest import SUPPORTED, read_folder  # noqa: E402
from ragsearch.llm import LLMCache, LLMError, make_llm  # noqa: E402
from ragsearch.retriever import Retriever  # noqa: E402
from ragsearch.textproc import extract_citations  # noqa: E402

st.set_page_config(page_title="Private RAG Search", page_icon="🔎", layout="wide")

BADGE = {"both": ("BM25 + Dense", "#1b7f5c", "#e3f5ee"), "bm25": ("BM25 only", "#1f5fae", "#e3eefb"),
         "dense": ("Dense only", "#b3491c", "#fdebe2"), "kg": ("Entity graph", "#5a48c2", "#ece9fb"), "none": ("-", "#555", "#eee")}

CSS = """
<style>
.block-container {padding-top: 2.2rem; max-width: 1250px;}
.badge {display:inline-block; padding:1px 8px; border-radius:10px; font-size:0.74rem; font-weight:600; margin-right:6px; white-space:nowrap;}
.passage {border:1px solid rgba(128,128,128,0.28); border-radius:8px; padding:10px 12px; margin-bottom:10px;}
.passage.cited {border-left:4px solid #2a78d6;}
.passage .num {font-weight:700; margin-right:6px;}
.passage .src {font-size:0.8rem; opacity:0.75;}
.passage .txt {font-size:0.88rem; margin-top:6px; line-height:1.45;}
.passage .ranks {font-size:0.74rem; opacity:0.7; margin-top:6px;}
.answer {border-radius:8px; padding:14px 16px; background:rgba(42,120,214,0.08); font-size:1.02rem; line-height:1.55;}
.answer.abstain {background:rgba(235,160,0,0.13);}
.answer.closed {background:rgba(128,128,128,0.10);}
.cite {font-weight:700; color:#2a78d6;}
.small {font-size:0.8rem; opacity:0.75;}
</style>
"""


@st.cache_resource(show_spinner=False)
def get_config() -> Config:
    cfg = load_config(os.environ.get("RAGSEARCH_CONFIG"))
    profile = os.environ.get("RAGSEARCH_PROFILE")
    if profile and profile in (cfg.get("profiles") or {}):
        cfg = Config(deep_merge(dict(cfg), cfg["profiles"][profile] or {}))
    extra = [x for x in os.environ.get("RAGSEARCH_SET", "").split(";") if x.strip()]
    return Config(apply_overrides(dict(cfg), extra))


@st.cache_resource(show_spinner="Loading the index ...")
def get_collection(name: str, stamp: float) -> Collection:
    return Collection.load(name, get_config())


@st.cache_resource(show_spinner=False)
def get_llm(model: str):
    # temperature 0 + fixed seed: identical prompts are answered from a local cache,
    # so changing a display setting does not re-run the model
    cfg = get_config()
    cache = LLMCache(str(resolve(cfg, "index_dir") / ".app_llm_cache.sqlite"))
    return make_llm(cfg, model, cache)


def tuned_defaults(cfg: Config) -> dict:
    """top-k and abstention threshold tuned on the dev split, if the evaluation has been run."""
    path = resolve(cfg, "results_dir") / "answers" / "pubmedqa" / "dev_tuning.json"
    if path.exists():
        try:
            t = json.loads(path.read_text())
            return {"top_k": int(t.get("top_k", {}).get("best", cfg.retrieval.top_k)), "signal": t.get("signal"), "threshold": t.get("threshold")}
        except Exception:
            pass
    return {"top_k": int(cfg.retrieval.top_k), "signal": None, "threshold": None}


def render_answer(text: str, n_passages: int, css: str = "answer") -> None:
    safe = html.escape(text)
    for n in sorted(set(extract_citations(text)), reverse=True):
        if 1 <= n <= n_passages:
            safe = safe.replace("[%d]" % n, '<span class="cite">[%d]</span>' % n)
    st.markdown('<div class="%s">%s</div>' % (css, safe), unsafe_allow_html=True)


def render_passage(i: int, passage, hit, cited: bool) -> None:
    label, fg, bg = BADGE[hit.found_by]
    ranks = []
    if hit.bm25_rank is not None:
        ranks.append("BM25 rank %d (score %.2f)" % (hit.bm25_rank, hit.bm25_score))
    if hit.dense_rank is not None:
        ranks.append("dense rank %d (cosine %.3f)" % (hit.dense_rank, hit.dense_score))
    if hit.rerank_score is not None:
        ranks.append("re-ranker %.3f" % hit.rerank_score)
    if hit.kg_rank is not None:
        ranks.append("entity-graph rank %d" % hit.kg_rank)
    st.markdown(
        '<div class="passage%s"><span class="num">[%d]</span>'
        '<span class="badge" style="color:%s;background:%s">%s</span>%s'
        '<span class="src">%s</span><div class="txt">%s</div><div class="ranks">%s</div></div>'
        % (" cited" if cited else "", i, fg, bg, label,
           '<span class="badge" style="color:#2a78d6;background:rgba(42,120,214,0.12)">cited</span>' if cited else "",
           html.escape(passage.location()), html.escape(passage.text), html.escape(" · ".join(ranks))),
        unsafe_allow_html=True)


def add_documents_panel(cfg: Config) -> None:
    with st.sidebar.expander("Add your own documents"):
        st.caption("PDF, text, Markdown or HTML. Files are indexed on this machine and are not uploaded anywhere.")
        files = st.file_uploader("Files", accept_multiple_files=True, type=[s.lstrip(".") for s in sorted(SUPPORTED)], label_visibility="collapsed")
        name = st.text_input("Collection name", value="mydocs")
        if st.button("Build index", disabled=not files):
            with tempfile.TemporaryDirectory() as tmp:
                for f in files:
                    (Path(tmp) / Path(f.name).name).write_bytes(f.getbuffer())
                docs = read_folder(tmp)
                if not docs:
                    st.error("No readable text in these files.")
                    return
                with st.spinner("Chunking and indexing %d documents ..." % len(docs)):
                    col = Collection.build(name.strip() or "mydocs", docs, cfg, verbose=False)
            st.session_state["collection_stamp"] = time.time()
            st.session_state["collection"] = col.name
            st.success("Indexed %d documents (%d passages)." % (len(docs), len(col.passages)))
            st.rerun()


def main() -> None:
    cfg = get_config()
    st.markdown(CSS, unsafe_allow_html=True)
    st.title("Private RAG Search")
    st.caption("Answers come only from your documents, with citations. Retrieval and generation both run on this machine.")

    names = list_collections(cfg)
    add_documents_panel(cfg)
    if not names:
        st.info("No collection has been indexed yet. Run `python scripts/02_build_indexes.py --only handbook`, "
                "or add documents from the sidebar.")
        return

    defaults = tuned_defaults(cfg)
    sb = st.sidebar
    sb.header("Settings")
    preferred = st.session_state.get("collection", "handbook" if "handbook" in names else names[0])
    col_name = sb.selectbox("Collection", names, index=names.index(preferred) if preferred in names else 0)
    col = get_collection(col_name, st.session_state.get("collection_stamp", 0.0))
    sb.caption("%s documents · %s passages" % ("{:,}".format(col.meta.get("n_documents", 0)), "{:,}".format(len(col.passages))))

    mode_label = sb.radio("Retrieval mode", ["Hybrid (BM25 + dense, RRF)", "BM25 only", "Dense only"], index=0)
    mode = {"Hybrid (BM25 + dense, RRF)": "hybrid", "BM25 only": "bm25", "Dense only": "dense"}[mode_label]
    top_k = sb.slider("Passages given to the LLM (top-k)", 1, 10, int(defaults["top_k"]))
    rerank = sb.toggle("Re-rank with the cross-encoder", value=False)
    use_kg = sb.toggle("Entity-graph expansion", value=False, disabled=col.kg is None)
    dense_model = None
    if len(col.dense_models) > 1:
        dense_model = sb.selectbox("Dense encoder", col.dense_models, format_func=lambda m: m.split("/")[-1])

    sb.subheader("Abstention")
    sb.caption("The LLM is always told to reply \"%s\" when the passages do not answer the question." % NOT_FOUND)
    signals = {"dense_top": "top dense cosine similarity", "rrf_top": "top fused RRF score", "bm25_top": "top BM25 score", "ce_top": "top cross-encoder score"}
    # the threshold was tuned on PubMedQA, so it is switched on by default only there
    use_thr = sb.toggle("Also abstain on low retrieval confidence",
                        value=defaults["threshold"] is not None and col_name == "pubmedqa", key="thr_" + col_name)
    signal, threshold = None, None
    if use_thr:
        keys = list(signals)
        signal = sb.selectbox("Signal", keys, index=keys.index(defaults["signal"]) if defaults["signal"] in keys else 0,
                              format_func=lambda k: signals[k])
        threshold = sb.number_input("Abstain below", value=float(defaults["threshold"]) if (defaults["threshold"] is not None and signal == defaults["signal"]) else 0.30,
                                    format="%.4f", step=0.01)
        if defaults["threshold"] is not None and signal == defaults["signal"]:
            sb.caption("Default threshold tuned on the PubMedQA dev questions; other collections may need a different value.")

    sb.subheader("Local LLM")
    base = get_llm(cfg.llm.model)
    installed = [m for m in base.available_models() if m]
    options = installed or [cfg.llm.model]
    model = sb.selectbox("Model (Ollama)", options, index=options.index(cfg.llm.model) if cfg.llm.model in options else 0)
    compare = sb.toggle("Also show the closed-book answer", value=True, help="The same model answering without the documents.")
    if not base.is_up():
        sb.error("The local LLM server at %s is not running. Start Ollama, then reload this page." % cfg.llm.host)

    # ------------------------------------------------------------------- question
    bench_path = col.dir / "benchmark.json"
    examples = []
    if bench_path.exists():
        try:
            qs = json.loads(bench_path.read_text()).get("test", [])
            if isinstance(qs, list):
                examples = [q["question"] for q in qs if q.get("answerable")][:3] + [q["question"] for q in qs if not q.get("answerable")][:1]
        except Exception:
            examples = []
    if examples:
        st.caption("Try one (the last is not covered by this collection):")
        cols = st.columns(len(examples))
        for c, ex in zip(cols, examples):
            if c.button(ex, use_container_width=True):
                st.session_state["question"] = ex
    with st.form("ask", clear_on_submit=False):
        question = st.text_input("Question", key="question", placeholder="Ask a question about the documents in this collection")
        submitted = st.form_submit_button("Search and answer", type="primary")
    if not (submitted or st.session_state.get("question")):
        return
    question = (question or "").strip()
    if not question:
        return

    retriever = Retriever(col, cfg)
    try:
        with st.spinner("Retrieving ..."):
            res = retriever.search(question, mode=mode, top_k=top_k, rerank=rerank, use_kg=use_kg, dense_model=dense_model)
    except Exception as exc:
        st.error("Retrieval failed: %s\n\nIf a model is missing, run `python scripts/01_download_data.py` once while online." % exc)
        return
    if res.mode != mode:
        st.warning("This collection has no dense index, so BM25 was used.")

    llm = get_llm(model)
    answer, closed, err = None, None, None
    try:
        with st.spinner("Generating a grounded answer with %s ..." % model):
            answer = answer_rag(question, res, llm, signal=signal, threshold=threshold, kg=col.kg if use_kg else None)
        if compare:
            with st.spinner("Closed-book answer ..."):
                closed = answer_closed_book(question, llm)
    except LLMError as exc:
        err = str(exc)

    left, right = st.columns([5, 4], gap="large")
    with left:
        st.subheader("Answer")
        if err:
            st.error(err)
        elif answer.abstained:
            render_answer(NOT_FOUND, 0, "answer abstain")
            if answer.abstain_reason == "threshold":
                st.caption("Declined before generation: %s = %.4f is below the threshold %.4f." % (signals.get(answer.signal, answer.signal), answer.signal_value, answer.threshold))
            elif answer.abstain_reason == "llm":
                st.caption("The model judged that the retrieved passages do not contain the answer.")
        else:
            render_answer(answer.text, len(res.passages))
            notes = []
            if answer.invalid_citations:
                notes.append("cites passage number(s) %s that were not retrieved" % ", ".join(map(str, answer.invalid_citations)))
            if answer.n_uncited_sentences:
                notes.append("%d sentence(s) carry no citation" % answer.n_uncited_sentences)
            nums = sorted({n for s in answer.sentences for n in s.unsupported_numbers})
            if nums:
                notes.append("number(s) not found in the cited passages: %s" % ", ".join(nums))
            terms = sorted({t for s in answer.sentences for t in s.unsupported_terms})
            if terms:
                notes.append("term(s) not found in the cited passages: %s" % ", ".join(terms[:6]))
            if notes:
                st.warning("Citation check: " + "; ".join(notes) + ".")
            else:
                st.caption("Citation check passed: every sentence cites a retrieved passage.")
            st.caption("%.1f s · %d prompt tokens · %d passages · %s" % (answer.seconds, answer.prompt_tokens, len(res.passages), model))
        if closed is not None:
            st.subheader("Closed-book answer (no documents)")
            render_answer(closed.text, 0, "answer closed")
            st.caption("The same model from memory alone. It cannot cite a source, and for private documents it can only guess.")

    with right:
        tab_e, tab_d, tab_g = st.tabs(["Evidence", "Retrieval details", "Evidence graph"])
        cited = set(answer.citations) if (answer is not None and not answer.abstained) else set()
        with tab_e:
            if not res.passages:
                st.write("Nothing was retrieved.")
            for i, (p, h) in enumerate(zip(res.passages, res.hits), start=1):
                render_passage(i, p, h, i in cited)
        with tab_d:
            st.dataframe([{"#": h.rank, "found by": BADGE[h.found_by][0], "BM25 rank": h.bm25_rank, "dense rank": h.dense_rank,
                           "score": round(h.score, 4), "source": p.location()} for p, h in zip(res.passages, res.hits)],
                         hide_index=True, use_container_width=True)
            st.caption("Confidence signals: " + " · ".join("%s = %s" % (k.replace("_", " "), ("%.4f" % v) if v is not None else "n/a")
                                                              for k, v in res.signals.items()))
            st.caption("Timings: " + " · ".join("%s %.0f ms" % (k, 1000 * v) for k, v in res.timings.items()))
            s = res.settings
            st.caption("mode %s · top-k %d · fusion %s (k=%d, depth %d) · BM25 k1=%.2f b=%.2f · dense %s" % (
                s["mode"], s["top_k"], s["fusion"], s["rrf_k"], s["depth"], s["k1"], s["b"], (s["dense_model"] or "none").split("/")[-1]))
        with tab_g:
            if col.kg is None or not res.passages:
                st.write("No entity graph for this collection.")
            else:
                st.caption("Retrieved passages (boxes) and the terms they share (ovals). Terms from the question are highlighted.")
                st.graphviz_chart(col.kg.evidence_dot([h.pid for h in res.hits], ["[%d]" % i for i in range(1, len(res.hits) + 1)], question),
                                  use_container_width=True)

    st.divider()
    st.caption("Private by design: indexes are local files, the retrieval models run in this process, and the LLM runs at %s. "
               "No document or query is sent to another machine, and nothing is used for training." % cfg.llm.host)


main()
