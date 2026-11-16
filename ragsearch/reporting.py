"""Turn the result files into RESULTS.md, LaTeX tables, key-number macros and figures.

Everything here is derived mechanically from files under results/. Nothing is typed in by
hand, so the report can never show a number that the pipeline did not produce.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ------------------------------------------------------------------------ formatting


def f3(x: Any, nd: int = 3) -> str:
    return "--" if x is None else ("%." + str(nd) + "f") % float(x)


def pct(x: Any, nd: int = 1) -> str:
    return "--" if x is None else ("%." + str(nd) + "f") % (100.0 * float(x))


def pval(p: Any) -> str:
    if p is None:
        return "--"
    return "<0.001" if p < 0.001 else "%.3f" % p


def peq(p: Any) -> str:
    """'p = 0.031' / 'p < 0.001' for running text."""
    if p is None:
        return "p = --"
    return "p < 0.001" if p < 0.001 else "p = %.3f" % p


def md_cell(s: str) -> str:
    """Plain-text version of a cell that may contain inline LaTeX math."""
    s = re.sub(r"\$([^$]*)\$", lambda m: m.group(1).replace("_{", "_").replace("}", "").replace("_1", "1"), str(s))
    return s


def tex(s: Any) -> str:
    s = str(s)
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("#", r"\#"), ("_", r"\_"), ("$", r"\$")):
        s = s.replace(a, b)
    return s


def nick(model: Optional[str]) -> str:
    """Short display name for a model."""
    if not model:
        return "--"
    name = model.split("/")[-1]
    table = {"all-MiniLM-L6-v2": "MiniLM-L6", "bge-small-en-v1.5": "bge-small", "ms-marco-MiniLM-L-6-v2": "MiniLM cross-encoder",
             "nli-deberta-v3-base": "DeBERTa-v3 NLI"}
    return table.get(name, name)


def load(path: Path) -> Optional[Any]:
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def load_results(results: Path) -> Dict[str, Any]:
    r = Path(results)
    return {
        "env": load(r / "env.json"),
        "trec": load(r / "retrieval" / "trec-covid" / "metrics.json"),
        "pubret": load(r / "retrieval" / "pubmedqa" / "metrics.json"),
        "pub": load(r / "answers" / "pubmedqa" / "summary.json"),
        "pub_tuning": load(r / "answers" / "pubmedqa" / "dev_tuning.json"),
        "hb": load(r / "answers" / "handbook" / "summary.json"),
        "chunk_pub": load(r / "ablations" / "chunk_size_pubmedqa.json"),
        "chunk_trec": load(r / "ablations" / "chunk_size_trec_covid.json"),
        "modes": load(r / "ablations" / "answer_modes_pubmedqa.json"),
        "usage": load(r / "answers" / "llm_usage.json"),
        "handcheck": load(r / "answers" / "pubmedqa" / "judge_handcheck_agreement.json"),
        "nli_validation": load(r / "answers" / "citation_nli_validation.json"),
        "timings": load(r / "timings.json"),
    }


def system_label(s: Dict, latex: bool = False) -> str:
    key = s["key"]
    m = nick(s.get("model")) if s.get("model") else ""
    if key == "bm25_default":
        return "BM25 (default $k_1$=0.9, $b$=0.4)" if latex else "BM25 (default k1=0.9, b=0.4)"
    if key == "bm25":
        return "BM25 (tuned on dev)"
    if key == "bm25_rerank":
        return "BM25 + re-ranker"
    if key.startswith("dense_"):
        return "Dense (%s)" % m
    if key.startswith("hybrid_rerank_"):
        return "Hybrid (BM25 + %s) + re-ranker" % m
    if key == "hybrid_kg":
        return "Hybrid + entity graph"
    if key.startswith("hybrid_"):
        return "Hybrid RRF (BM25 + %s)" % m
    if key == "weighted":
        return "Weighted score fusion (tuned)"
    if key == "dat":
        return "Dynamic alpha (DAT-style)"
    return s.get("label", key)


# ---------------------------------------------------------------------------- tables

class Table:
    def __init__(self, name: str, caption: str, header: Sequence[str], rows: Sequence[Sequence[str]],
                 align: Optional[str] = None, wide: bool = False, note: str = "", bold_max: Sequence[int] = ()):
        self.name, self.caption, self.header, self.rows = name, caption, list(header), [list(r) for r in rows]
        self.align = align or ("l" + "r" * (len(header) - 1))
        self.wide, self.note, self.bold_max = wide, note, tuple(bold_max)

    def _best(self) -> Dict[int, float]:
        best = {}
        for c in self.bold_max:
            vals = []
            for r in self.rows:
                if c >= len(r):
                    continue
                m = re.match(r"^-?\d+(\.\d+)?", r[c])
                if m:
                    vals.append(float(m.group(0)))
            if vals:
                best[c] = max(vals)
        return best

    def markdown(self) -> str:
        best = self._best()
        lines = ["| " + " | ".join(md_cell(h) for h in self.header) + " |", "|" + "|".join(["---"] + ["---:"] * (len(self.header) - 1)) + "|"]
        for r in self.rows:
            if r == ["---"]:
                continue
            cells = []
            for c, cell in enumerate(r):
                m = re.match(r"^-?\d+(\.\d+)?", cell)
                cell = md_cell(cell)
                if c in best and m and abs(float(m.group(0)) - best[c]) < 1e-12:
                    cell = "**%s**" % cell
                cells.append(cell)
            lines.append("| " + " | ".join(cells) + " |")
        out = "\n".join(lines)
        if self.note:
            out += "\n\n*" + self.note + "*"
        return out

    def latex(self) -> str:
        best = self._best()
        env = "table*" if self.wide else "table"
        lines = ["\\begin{%s}[t]" % env, "\\centering\\small", "\\begin{tabular}{%s}" % self.align, "\\toprule",
                 " & ".join(tex_cell(h) for h in self.header) + " \\\\", "\\midrule"]
        for r in self.rows:
            if r and r[0] == "---":
                lines.append("\\midrule")
                continue
            cells = []
            for c, cell in enumerate(r):
                m = re.match(r"^-?\d+(\.\d+)?", cell)
                t = tex_cell(cell)
                if c in best and m and abs(float(m.group(0)) - best[c]) < 1e-12:
                    t = "\\textbf{%s}" % t
                cells.append(t)
            lines.append(" & ".join(cells) + " \\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]
        cap = tex_cell(self.caption) + ((" " + tex_cell(self.note)) if self.note else "")
        lines += ["\\caption{%s}" % cap, "\\label{tab:%s}" % self.name, "\\end{%s}" % env]
        return "\n".join(lines) + "\n"


def tex_cell(s: str) -> str:
    """Escape a cell, but leave inline math ($...$) alone."""
    parts = re.split(r"(\$[^$]*\$)", str(s))
    out = "".join(p if (p.startswith("$") and p.endswith("$") and len(p) > 1) else tex(p) for p in parts)
    for src, dst in (("±", "$\\pm$"), ("≥", "$\\geq$"), ("≤", "$\\leq$"), ("α", "$\\alpha$"), ("Δ", "$\\Delta$"),
                     ("−", "$-$"), ("·", "$\\cdot$"), ("—", "---"), ("–", "--")):
        out = out.replace(src, dst)
    return out


def ci_cell(m: Dict) -> str:
    return "%.3f ± %.3f" % (m["mean"], (m["ci"][1] - m["ci"][0]) / 2.0)


def retrieval_tables(R: Dict) -> List[Table]:
    tables: List[Table] = []
    t = R.get("trec")
    if t:
        rows = []
        prev_group = None
        for s in t["systems"]:
            if prev_group is not None and s["group"] != prev_group:
                rows.append(["---"])
            prev_group = s["group"]
            te, al = s["metrics"]["test"], s["metrics"].get("all", s["metrics"]["test"])
            rows.append([system_label(s, True), ci_cell(te["ndcg@10"]), f3(te["recall@100"]["mean"]),
                         f3(te["capped_recall@100"]["mean"]), f3(al["ndcg@10"]["mean"])])
        tables.append(Table(
            "retrieval_trec", "TREC-COVID retrieval. nDCG@10 (± half-width of the 95%% bootstrap interval), Recall@100 and capped "
            "Recall@100 on the %d test topics; the last column is nDCG@10 on all %d topics for comparison with published BEIR numbers "
            "(the %d dev topics were used for tuning)." % (t["n_test"], t["n_test"] + t["n_dev"], t["n_dev"]),
            ["System", "nDCG@10", "R@100", "R$_{cap}$@100", "nDCG@10 (all)"], rows, wide=True, bold_max=(1, 2, 3, 4)))
        if t.get("significance"):
            key2label = {s["key"]: system_label(s, True) for s in t["systems"]}
            rows = [[key2label.get(x["a"], x["a"]), key2label.get(x["b"], x["b"]), "%+.3f" % x["diff"],
                     "%d / %d" % (x["wins"], x["losses"]), pval(x["p"])] for x in t["significance"]]
            tables.append(Table("retrieval_tests", "Paired comparisons on the TREC-COVID test topics (nDCG@10): mean difference A − B, "
                                "topics won / lost by A, and the two-sided paired randomisation test.",
                                ["A", "B", "Δ nDCG@10", "wins / losses", "p"], rows, align="llrrr", wide=True))
        fus = []
        for row in t.get("rrf_k", []):
            fus.append(["RRF, k=%d" % row["k"], f3(row.get("dev")), f3(row.get("test"))])
        if t.get("alpha_curve"):
            best = next(c for c in t["alpha_curve"] if abs(c["alpha"] - t["alpha_tuned"]) < 1e-9)
            fus.append(["Weighted, α=%.1f (tuned on dev)" % t["alpha_tuned"], f3(best.get("dev")), f3(best.get("test"))])
            a0 = next(c for c in t["alpha_curve"] if c["alpha"] == 0.0)
            a1 = next(c for c in t["alpha_curve"] if c["alpha"] == 1.0)
            fus.append(["Weighted, α=0 (BM25 only)", f3(a0.get("dev")), f3(a0.get("test"))])
            fus.append(["Weighted, α=1 (dense only)", f3(a1.get("dev")), f3(a1.get("test"))])
        dat = next((s for s in t["systems"] if s["key"] == "dat"), None)
        if dat:
            fus.append(["Dynamic α (DAT-style), mean α=%.2f" % t["dat"]["mean_alpha"], f3(dat["metrics"].get("dev", {}).get("ndcg@10", {}).get("mean")),
                        f3(dat["metrics"]["test"]["ndcg@10"]["mean"])])
        for row in t.get("kg_grid", []):
            fus.append(["Hybrid + entity graph, w=%.2f" % row["weight"], f3(row.get("dev")), f3(row.get("test"))])
        if fus:
            tables.append(Table("ablation_fusion", "Fusion ablation on TREC-COVID (nDCG@10, BM25 + %s). α is the dense weight." % nick(t.get("primary_dense_model")),
                                ["Fusion method", "dev", "test"], fus, bold_max=(2,)))
        if t.get("latency_ms_per_query"):
            tables.append(Table("latency_retrieval", "Retrieval latency on the TREC-COVID index (%s passages), mean over 20 queries." % "{:,}".format(t["collection"]["n_passages"]),
                                ["Mode", "ms / query"], [[k, "%.0f" % v] for k, v in t["latency_ms_per_query"].items()]))
    p = R.get("pubret")
    if p:
        rows = []
        for s in p["systems"]:
            te = s["metrics"]["test"]
            rows.append([system_label(s, True), f3(te["recall@1"]["mean"]), f3(te["recall@5"]["mean"]), f3(te["recall@10"]["mean"]), f3(te["mrr@10"]["mean"])])
        tables.append(Table("retrieval_pubmedqa", "PubMedQA known-item retrieval: finding the abstract a question was written about, among "
                            "%s abstracts (%d test questions)." % ("{:,}".format(p["collection"]["n_documents"]), p["n_test"]),
                            ["System", "R@1", "R@5", "R@10", "MRR@10"], rows, wide=True, bold_max=(1, 2, 3, 4)))
    return tables


def answer_tables(R: Dict) -> List[Table]:
    tables: List[Table] = []
    pub, hb = R.get("pub"), R.get("hb")
    benches = [(n, s) for n, s in (("PubMedQA", pub), ("Handbook", hb)) if s]
    if not benches:
        return tables
    rows = []
    for name, s in benches:
        j = s.get("judge")
        if j:
            rows.append([name, str(j["n"]), "%d (%s%%)" % (j["rag_wins"], pct(j["rag_win_rate"])), "%d (%s%%)" % (j["ties"], pct(j["tie_rate"])),
                         "%d (%s%%)" % (j["closed_book_wins"], pct(j["closed_book_win_rate"])), pval(j["sign_test_p"]), pct(j.get("position_consistency"))])
    if rows:
        tables.append(Table("judge", "Pairwise LLM-as-a-judge against the reference answer: RAG vs. the closed-book LLM on answerable test questions. "
                            "A system wins only if the judge prefers it in both presentation orders; p is a two-sided sign test on wins vs. losses; "
                            "the last column is the share of comparisons where both orders agreed.",
                            ["Benchmark", "n", "RAG wins", "ties", "closed-book wins", "p", "order-consistent %"], rows, wide=True))
    if pub and pub.get("decision_accuracy"):
        d = pub["decision_accuracy"]
        rows = [["Majority class", pct(d["majority_baseline"]), "100.0", "--"],
                ["Closed-book LLM", "%s [%s, %s]" % (pct(d["closed_book"]["accuracy"]), pct(d["closed_book"]["ci"][0]), pct(d["closed_book"]["ci"][1])), "100.0", "--"],
                ["RAG (ours)", "%s [%s, %s]" % (pct(d["rag"]["accuracy"]), pct(d["rag"]["ci"][0]), pct(d["rag"]["ci"][1])), pct(d["rag"]["coverage"]),
                 pct(d["rag"]["accuracy_when_answered"])]]
        tables.append(Table("decision", "PubMedQA yes / no / maybe accuracy on the %d answerable test questions (95%% Wilson intervals). A declined question "
                            "counts as wrong; coverage is the share answered. McNemar exact test on discordant pairs: %s (RAG only correct on %d, closed-book only on %d)."
                            % (d["n"], peq(d["mcnemar"]["p"]), d["mcnemar"]["rag_only_correct"], d["mcnemar"]["closed_book_only_correct"]),
                            ["System", "Accuracy %", "Coverage %", "Acc. when answered %"], rows, wide=True))
    rows = []
    def col(s, f):
        return f(s) if s else "--"
    metrics = [
        ("Answers evaluated", lambda s: str(s["post_check"]["n_answers"])),
        ("Citation precision (ALCE-style) %", lambda s: pct((s.get("citation") or {}).get("citation_precision_micro"))),
        ("Citation recall (ALCE-style) %", lambda s: pct((s.get("citation") or {}).get("citation_recall_micro"))),
        ("Faithfulness (RAGAs-style) %", lambda s: pct((s.get("faithfulness") or {}).get("mean"))),
        ("Fully faithful answers %", lambda s: pct((s.get("faithfulness") or {}).get("fully_faithful_rate"))),
        ("Answers citing the gold document %", lambda s: pct(s["post_check"].get("cites_gold_document"))),
        ("Answers with an invalid citation number %", lambda s: pct(s["post_check"]["answers_with_invalid_citation"])),
        ("Sentences without a citation %", lambda s: pct(s["post_check"]["uncited_sentence_rate"])),
        ("Answers with a number absent from cited passages %", lambda s: pct(s["post_check"]["answers_with_unsupported_number"])),
        ("Citations per answer", lambda s: f3(s["post_check"]["mean_citations_per_answer"], 2)),
    ]
    for label, fn in metrics:
        rows.append([label] + [col(s, fn) for _, s in benches])
    backend = (benches[0][1].get("citation") or {}).get("backend", "")
    window = (benches[0][1].get("citation") or {}).get("nli_window")
    tables.append(Table("grounding", "Groundedness of the RAG answers that were not declined. Citation precision / recall use entailment by %s%s; "
                        "a bare Yes / No / Maybe sentence is not scored. "
                        "Faithfulness is the share of atomic claims the judge model finds supported by the retrieved passages." % (
                            nick(backend.split(":", 1)[1]) if backend.startswith("nli:") else "the judge LLM",
                            (" on windows of %d consecutive passage sentences (maximum over windows)" % window)
                            if backend.startswith("nli:") and window else ""),
                        ["Metric"] + [n for n, _ in benches], rows, align="p{%.2f\\linewidth}" % (0.86 - 0.2 * len(benches)) + "r" * len(benches)))
    rows = []
    for name, s in benches:
        a = s["abstention"]
        for policy, label in (("llm", "LLM declines"), ("threshold", "Score threshold"), ("combined", "Both (system)")):
            m = a[policy]
            if policy != "llm" and a.get("threshold_value") is None:
                continue
            rows.append(["%s: %s" % (name, label), "%d / %d (%s%%)" % (m["abstained_unanswerable"], m["n_unanswerable"], pct(m["abstention_accuracy"])),
                         "%d / %d (%s%%)" % (m["abstained_answerable"], m["n_answerable"], pct(m["over_refusal_rate"])), f3(m.get("abstention_f1"), 2)])
        rows.append(["%s: closed-book says it does not know" % name, pct(a.get("closed_book_refusal_unanswerable")) + "%",
                     pct(a.get("closed_book_refusal_answerable")) + "%", "--"])
        rows.append(["---"])
    if rows and rows[-1] == ["---"]:
        rows.pop()
    sig = (pub or benches[0][1])["abstention"]
    tables.append(Table("abstention", "Abstention. Left: out-of-collection questions correctly declined (abstention accuracy). Right: answerable questions "
                        "wrongly declined (over-refusal). The score threshold uses the signal %s with threshold %s, both chosen on the PubMedQA dev questions%s."
                        # the signal name is not escaped here: Table.latex escapes the whole caption
                        % (str(sig.get("signal")), f3(sig.get("threshold_value"), 4), " and transferred unchanged to the handbook" if hb else ""),
                        ["Policy", "Declined out-of-collection", "Declined answerable", "F1"], rows, align="lrrr", wide=True))
    tun = R.get("pub_tuning")
    if tun and tun.get("signals"):
        rows = []
        pub_abst = (pub or {}).get("abstention", {})
        test_auc = pub_abst.get("signal_test_auroc", {})
        test_rates = pub_abst.get("signal_test_at_dev_threshold", {})
        names = {"dense_top": "Top dense cosine similarity", "rrf_top": "Top RRF score (as proposed)", "bm25_top": "Top BM25 score",
                 "ce_top": "Top cross-encoder score", "overlap10": "BM25 / dense top-10 overlap"}
        n_dev_a = n_dev_u = None
        for sig_name, v in tun["signals"].items():
            tr = test_rates.get(sig_name) or {}
            n_dev_a, n_dev_u = v.get("n_answerable", n_dev_a), v.get("n_unanswerable", n_dev_u)
            rows.append([names.get(sig_name, sig_name) + (" *" if sig_name == tun.get("signal") else ""), f3(v.get("auroc")), f3(test_auc.get(sig_name)),
                         pct(v.get("tpr")), pct(v.get("fpr")), pct(tr.get("caught")), pct(tr.get("refused"))])
        n_test = ("%d out-of-collection and %d answerable" % (pub["n_test_unanswerable"], pub["n_test_answerable"])) if pub else "the test"
        n_dev = ("%d out-of-collection and %d answerable" % (n_dev_u, n_dev_a)) if n_dev_a is not None and n_dev_u is not None else "the dev"
        tables.append(Table("signals", "Retrieval-confidence signals for abstention on PubMedQA. AUROC measures how well each signal separates answerable "
                            "from out-of-collection questions. Each signal gets its own threshold, tuned on the dev questions (%s); the last four columns "
                            "give the percentage of out-of-collection questions that threshold catches and of answerable questions it wrongly refuses, on dev "
                            "and when applied unchanged to the held-out test questions (%s). * marks the signal used by the system." % (n_dev, n_test),
                            ["Signal", "AUROC dev", "AUROC test", "caught dev", "refused dev", "caught test", "refused test"],
                            rows, wide=True, bold_max=(1, 2)))
    if tun and tun.get("top_k"):
        rows = []
        for k, v in tun["top_k"]["grid"].items():
            rows.append([("k = %s" % k) + (" *" if int(k) == tun["top_k"]["best"] else ""), pct(v.get("gold_in_top_k")), pct(v.get("answerable_ok")),
                         pct(v.get("unanswerable_ok")), pct(v.get("utility")), "%.0f" % v.get("mean_prompt_tokens", 0), f3(v.get("mean_seconds"), 1)])
        tables.append(Table("ablation_topk", "Number of passages given to the LLM (PubMedQA dev, %d questions). 'Answerable correct' requires an answer with the "
                            "right yes / no / maybe label; 'declined' is the share of out-of-collection questions the LLM refused; utility is the share of all dev "
                            "questions handled correctly. * marks the value used on test." % tun.get("n_dev", 0),
                            ["top-k", "gold in top-k %", "answerable correct %", "declined %", "utility %", "prompt tokens", "s / answer"], rows, wide=True))
    rows = []
    cats: List[str] = []
    for _, s in benches:
        for c in s.get("failures", {}):
            if c not in cats:
                cats.append(c)
    order = ["ok", "citation_problem", "unfaithful", "wrong_conclusion", "no_clear_decision", "judged_worse", "over_refusal", "retrieval_miss",
             "correct_abstention", "false_answer"]
    names = {"ok": "Answerable: correct, cited and supported", "citation_problem": "Answerable: correct, but citation problem",
             "unfaithful": "Answerable: correct, but unsupported claim", "wrong_conclusion": "Answerable: wrong conclusion (gold passage retrieved)",
             "no_clear_decision": "Answerable: no clear yes / no / maybe", "judged_worse": "Answerable: judged worse than closed-book",
             "over_refusal": "Answerable: declined (gold passage retrieved)", "retrieval_miss": "Answerable: gold passage not retrieved",
             "correct_abstention": "Out-of-collection: declined", "false_answer": "Out-of-collection: answered anyway"}
    for c in [c for c in order if c in cats] + [c for c in cats if c not in order]:
        rows.append([names.get(c, c)] + [str(s.get("failures", {}).get(c, 0)) for _, s in benches])
    tables.append(Table("failures", "Outcome of every test question, by category (each question gets exactly one category; see failures.csv for the examples).",
                        ["Outcome"] + [n for n, _ in benches], rows, align="p{%.2f\\linewidth}" % (0.86 - 0.2 * len(benches)) + "r" * len(benches)))
    return tables


def ablation_tables(R: Dict) -> List[Table]:
    tables: List[Table] = []
    c = R.get("chunk_pub")
    if c:
        rows = [["%d (overlap %d)" % (r["chunk_words"], r["overlap_words"]), "{:,}".format(r["n_passages"]),
                 f3(r["systems"]["bm25"]["mrr@10"]), f3(r["systems"]["dense"]["mrr@10"]), f3(r["systems"]["hybrid"]["mrr@10"]),
                 f3(r["systems"]["hybrid"]["recall@1"]), f3(r["systems"]["hybrid"]["recall@5"])] for r in c["rows"]]
        tables.append(Table("ablation_chunk", "Chunk size ablation on PubMedQA known-item retrieval (%d test questions, dense model %s)." % (c["n_queries"], nick(c["dense_model"])),
                            ["Passage words", "passages", "BM25 MRR@10", "Dense MRR@10", "Hybrid MRR@10", "Hybrid R@1", "Hybrid R@5"], rows, wide=True, bold_max=(2, 3, 4, 5, 6)))
    c = R.get("chunk_trec")
    if c:
        rows = [["%d (overlap %d)" % (r["chunk_words"], r["overlap_words"]), "{:,}".format(r["n_passages"]), f3(r["systems"]["bm25"]["ndcg@10"]),
                 f3(r["systems"]["dense"]["ndcg@10"]), f3(r["systems"]["hybrid"]["ndcg@10"]), f3(r["systems"]["hybrid"]["recall@100"])] for r in c["rows"]]
        tables.append(Table("ablation_chunk_trec", "Chunk size ablation on TREC-COVID (%d test topics, dense model %s)." % (c["n_queries"], nick(c["dense_model"])),
                            ["Passage words", "passages", "BM25 nDCG@10", "Dense nDCG@10", "Hybrid nDCG@10", "Hybrid R@100"], rows, wide=True, bold_max=(2, 3, 4, 5)))
    m = R.get("modes")
    if m:
        rows = [[r["mode"], pct(r["gold_in_top_k"]), pct(r["decision_accuracy"]), pct(r["coverage"]), pct(r["abstention_accuracy"]), pct(r["utility"])] for r in m["rows"]]
        tables.append(Table("ablation_modes", "End-to-end effect of the retrieval mode on PubMedQA test answers (generator %s, top-k = %d, LLM abstention only)."
                            % (m["generator"], m["rows"][0]["top_k"]),
                            ["Retrieval mode", "gold in top-k %", "accuracy %", "coverage %", "declined out-of-coll. %", "utility %"], rows, wide=True, bold_max=(1, 2, 5)))
    return tables


# --------------------------------------------------------------------------- findings

def success_criteria(R: Dict) -> List[Dict[str, str]]:
    """The three success criteria stated in the proposal, checked against the results."""
    out = []
    t = R.get("trec")
    if t and t.get("hypothesis"):
        h = t["hypothesis"]
        best_name = "BM25" if h["bm25"] >= h["dense"] else "dense (%s)" % nick(t.get("primary_dense_model"))
        out.append({"criterion": "Hybrid matches or beats the better single retriever on nDCG@10 (TREC-COVID test topics)",
                    "outcome": "met" if h["holds"] else "not met",
                    "evidence": "hybrid %.3f vs. %s %.3f (BM25 %.3f, dense %.3f)" % (h["hybrid"], best_name, h["better_single"], h["bm25"], h["dense"])})
    pub = R.get("pub")
    if pub and pub.get("judge"):
        j = pub["judge"]
        outcome = "met" if j["rag_wins"] > j["n"] / 2.0 else ("partly met" if j["rag_wins"] > j["closed_book_wins"] else "not met")
        out.append({"criterion": "RAG wins most pairwise comparisons against the closed-book LLM (PubMedQA)",
                    "outcome": outcome,
                    "evidence": "RAG wins %d, ties %d, closed-book wins %d of %d (sign test, %s)" % (j["rag_wins"], j["ties"], j["closed_book_wins"], j["n"], peq(j["sign_test_p"]))})
    if pub:
        a = pub["abstention"][pub["primary_policy"]]
        acc, over = a.get("abstention_accuracy"), a.get("over_refusal_rate")
        if acc is not None and over is not None:
            outcome = "met" if (acc > 0.5 and over <= 0.10) else ("partly met" if acc > 0.5 else "not met")
            out.append({"criterion": "Abstention catches most out-of-collection questions without over-refusing (PubMedQA; operationalised as more than half "
                                     "caught and at most 10% of answerable questions refused)",
                        "outcome": outcome,
                        "evidence": "%d of %d out-of-collection questions declined (%s%%); %d of %d answerable questions declined (%s%%)" % (
                            a["abstained_unanswerable"], a["n_unanswerable"], pct(acc), a["abstained_answerable"], a["n_answerable"], pct(over))})
    return out


def findings(R: Dict) -> List[str]:
    """Plain-language statements of what the numbers show. Rule-based, no interpretation
    beyond the comparison itself."""
    out: List[str] = []
    t = R.get("trec")
    if t and t.get("hypothesis"):
        h = t["hypothesis"]
        prim = nick(t.get("primary_dense_model"))
        sig = {(x["a"], x["b"]): x for x in t.get("significance", [])}
        short = (t.get("primary_dense_model") or "").split("/")[-1]
        better = "BM25" if h["bm25"] >= h["dense"] else "the dense retriever"
        other = sig.get(("hybrid_" + short, "bm25" if h["bm25"] >= h["dense"] else "dense_" + short))
        s = ("Retrieval. On the %d TREC-COVID test topics, tuned BM25 reaches nDCG@10 = %.3f, the dense bi-encoder (%s) %.3f, and their "
             "reciprocal rank fusion %.3f. Hybrid is %s %s, the better single retriever, by %.3f" % (
                 t["n_test"], h["bm25"], prim, h["dense"], h["hybrid"], "above" if h["hybrid"] > h["better_single"] else
                 ("level with" if abs(h["hybrid"] - h["better_single"]) < 5e-4 else "below"), better, abs(h["hybrid"] - h["better_single"])))
        if other:
            s += " (paired randomisation test, %s; hybrid better on %d topics, worse on %d)" % (peq(other["p"]), other["wins"], other["losses"])
        s += ". An oracle that picks the better of the two retrievers per topic would reach %.3f, and the two top-10 lists share %.0f%% of their documents on average." % (
            h["oracle_best_of_two"], 100 * h["overlap_at_10_bm25_dense"])
        out.append(s)
        for mname, v in h.get("per_model", {}).items():
            if mname != short:
                out.append("With the %s encoder, dense retrieval reaches nDCG@10 = %.3f and the hybrid %.3f (BM25: %.3f)." % (nick(mname), v["dense"], v["hybrid"], h["bm25"]))
        rr = sig.get(("hybrid_rerank_" + short, "hybrid_" + short))
        if rr:
            out.append("Re-ranking the top 50 fused documents with the cross-encoder changes hybrid nDCG@10 from %.3f to %.3f (%+.3f, %s)." % (
                rr["mean_b"], rr["mean_a"], rr["diff"], peq(rr["p"])))
        kg = sig.get(("hybrid_kg", "hybrid_" + short))
        if kg:
            out.append("Adding the entity-graph ranker changes hybrid nDCG@10 from %.3f to %.3f (%+.3f, %s)." % (kg["mean_b"], kg["mean_a"], kg["diff"], peq(kg["p"])))
        dat = sig.get(("dat", "hybrid_" + short))
        if dat:
            out.append("The DAT-style per-query weighting reaches %.3f against %.3f for fixed RRF (%+.3f, %s)." % (dat["mean_a"], dat["mean_b"], dat["diff"], peq(dat["p"])))
    pub = R.get("pub")
    if pub:
        j = pub.get("judge")
        if j:
            out.append("Answers. On the %d answerable PubMedQA test questions the judge (%s) preferred the RAG answer %d times, the closed-book answer %d times, "
                       "and found no consistent winner %d times (two-sided sign test, %s). When RAG did answer (%d questions) it won %d and lost %d." % (
                           j["n"], pub.get("judge_model"), j["rag_wins"], j["closed_book_wins"], j["ties"], peq(j["sign_test_p"]),
                           j["when_rag_answered"]["n"], j["when_rag_answered"]["rag_wins"], j["when_rag_answered"]["closed_book_wins"]))
        d = pub.get("decision_accuracy")
        if d:
            out.append("On the yes / no / maybe label, the closed-book %s scores %s%% and RAG %s%% (majority-class baseline %s%%); RAG answered %s%% of the questions "
                       "and was right on %s%% of those it answered." % (pub.get("generator_model"), pct(d["closed_book"]["accuracy"]), pct(d["rag"]["accuracy"]),
                                                                       pct(d["majority_baseline"]), pct(d["rag"]["coverage"]), pct(d["rag"]["accuracy_when_answered"])))
        c, f, pc = pub.get("citation"), pub.get("faithfulness"), pub["post_check"]
        if c or f:
            s = "Grounding."
            if c:
                s += " ALCE-style citation precision is %s%% and citation recall %s%% over %d statements and %d citations." % (
                    pct(c["citation_precision_micro"]), pct(c["citation_recall_micro"]), c["n_statements"], c["n_citations"])
            if f:
                s += " RAGAs-style faithfulness averages %s%% over %d claims, and %s%% of answers have every claim supported." % (
                    pct(f["mean"]), f["n_claims"], pct(f["fully_faithful_rate"]))
            s += " The post-check found an out-of-range citation number in %s%% of answers and no citation on %s%% of sentences." % (
                pct(pc["answers_with_invalid_citation"]), pct(pc["uncited_sentence_rate"]))
            out.append(s)
        a = pub["abstention"]
        m, llm_only = a[pub["primary_policy"]], a["llm"]
        s = ("Abstention. The system declined %d of %d out-of-collection questions (%s%%) and wrongly declined %d of %d answerable ones (%s%%)." % (
            m["abstained_unanswerable"], m["n_unanswerable"], pct(m["abstention_accuracy"]), m["abstained_answerable"], m["n_answerable"], pct(m["over_refusal_rate"])))
        if pub["primary_policy"] != "llm":
            s += " The LLM's own refusals account for %d and %d of these; the rest come from the %s threshold." % (
                llm_only["abstained_unanswerable"], llm_only["abstained_answerable"], str(a.get("signal")).replace("_", " "))
        if a.get("closed_book_refusal_unanswerable") is not None:
            s += " The closed-book model said it did not know on %s%% of the out-of-collection questions." % pct(a["closed_book_refusal_unanswerable"])
        out.append(s)
    hb = R.get("hb")
    if hb:
        m = hb["abstention"][hb["primary_policy"]]
        s = "Handbook demo set. Of %d answerable hand-written questions the system answered %d" % (m["n_answerable"], m["n_answerable"] - m["abstained_answerable"])
        if hb.get("judge"):
            s += " (judge: RAG wins %d, ties %d, closed-book wins %d)" % (hb["judge"]["rag_wins"], hb["judge"]["ties"], hb["judge"]["closed_book_wins"])
        s += ", and it declined %d of %d out-of-collection questions." % (m["abstained_unanswerable"], m["n_unanswerable"])
        hb_llm = hb["abstention"].get("llm")
        if hb["primary_policy"] != "llm" and hb_llm:
            s += (" The LLM's own refusals account for %d and %d of these; the rest come from the %s threshold tuned on PubMedQA"
                  " (with the LLM's refusals alone the system would answer %d of the %d answerable questions)." % (
                      hb_llm["abstained_unanswerable"], hb_llm["abstained_answerable"],
                      str(hb["abstention"].get("signal")).replace("_", " "),
                      hb_llm["n_answerable"] - hb_llm["abstained_answerable"], hb_llm["n_answerable"]))
        out.append(s)
    return out


def macros(R: Dict) -> Dict[str, str]:
    """Key numbers as LaTeX macros (letters only in names)."""
    m: Dict[str, str] = {}
    env = R.get("env") or {}
    m["GeneratorModel"] = tex((R.get("pub") or {}).get("generator_model") or env.get("generator") or "the local LLM")
    m["JudgeModel"] = tex((R.get("pub") or {}).get("judge_model") or env.get("judge") or "the judge LLM")
    t = R.get("trec")
    if t:
        m["TrecDocs"] = "{:,}".format(t["collection"]["n_documents"])
        m["TrecPassages"] = "{:,}".format(t["collection"]["n_passages"])
        m["TrecDev"], m["TrecTest"] = str(t["n_dev"]), str(t["n_test"])
        m["BmKone"], m["BmB"] = "%.1f" % t["bm25_tuned"]["k1"], "%.2f" % t["bm25_tuned"]["b"]
        m["DenseModel"] = tex(nick(t.get("primary_dense_model")))
        if t.get("hypothesis"):
            h = t["hypothesis"]
            m["NdcgBm"], m["NdcgDense"], m["NdcgHybrid"] = f3(h["bm25"]), f3(h["dense"]), f3(h["hybrid"])
    pub = R.get("pub")
    if pub:
        m["PubDocs"] = "{:,}".format(pub["collection"]["n_documents"])
        m["PubAnswerable"], m["PubOut"], m["TopK"] = str(pub["n_test_answerable"]), str(pub["n_test_unanswerable"]), str(pub["top_k"])
        if pub.get("judge"):
            j = pub["judge"]
            m["RagWins"], m["CbWins"], m["JudgeTies"] = str(j["rag_wins"]), str(j["closed_book_wins"]), str(j["ties"])
        a = pub["abstention"][pub["primary_policy"]]
        m["AbstainAcc"], m["OverRefusal"] = pct(a.get("abstention_accuracy")), pct(a.get("over_refusal_rate"))
        rates = pub["abstention"].get("signal_test_at_dev_threshold") or {}
        for sig_name, prefix in (("rrf_top", "Rrf"), ("ce_top", "Ce")):
            if sig_name in rates:
                m[prefix + "TestCaught"], m[prefix + "TestRefused"] = pct(rates[sig_name]["caught"]), pct(rates[sig_name]["refused"])
            if sig_name in pub["abstention"].get("signal_test_auroc", {}):
                m[prefix + "TestAuroc"] = f3(pub["abstention"]["signal_test_auroc"][sig_name])
        if pub.get("citation"):
            m["CitePrec"], m["CiteRec"] = pct(pub["citation"]["citation_precision_micro"]), pct(pub["citation"]["citation_recall_micro"])
        if pub.get("faithfulness"):
            m["Faithfulness"] = pct(pub["faithfulness"]["mean"])
    hc = R.get("handcheck")
    if hc:
        m["HandcheckResult"] = "%d hand-checked comparisons (agreement with the judge %s\\%%, Cohen's $\\kappa=%s$)" % (
            hc["n"], pct(hc["agreement"]), f3(hc["kappa"], 2))
    v = R.get("nli_validation")
    if v and "0" in v["windows"] and str(v["configured_window"]) in v["windows"]:
        whole, used = v["windows"]["0"], v["windows"][str(v["configured_window"])]
        m["NliPairs"], m["NliWindow"] = str(v["n_pairs"]), str(v["configured_window"])
        m["NliJudgeRate"] = pct(v["judge_entailed_rate"])
        m["NliWholeRate"], m["NliWholeKappa"] = pct(whole["entailed_rate"]), f3(whole["kappa"], 2)
        m["NliWindowRate"], m["NliWindowKappa"] = pct(used["entailed_rate"]), f3(used["kappa"], 2)
    return m


# ---------------------------------------------------------------------------- figures

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, NEUTRAL = "#2a78d6", "#eb6834", "#1baf7a", "#b9b8b2"


def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.axisbelow": True, "legend.frameon": False, "legend.fontsize": 7.5, "figure.dpi": 150, "savefig.bbox": "tight",
        "pdf.fonttype": 42, "font.family": "DejaVu Sans",
    })
    return plt


def _save(fig, fig_dirs: Sequence[Path], name: str) -> None:
    for d in fig_dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(d) / (name + ".pdf"))
        fig.savefig(Path(d) / (name + ".png"), dpi=200)


def make_figures(R: Dict, results: Path, fig_dirs: Sequence[Path]) -> List[str]:
    plt = _style()
    made: List[str] = []
    t = R.get("trec")
    if t:
        systems = t["systems"]
        labels = [system_label(s) for s in systems]
        vals = [s["metrics"]["test"]["ndcg@10"]["mean"] for s in systems]
        los = [v - s["metrics"]["test"]["ndcg@10"]["ci"][0] for v, s in zip(vals, systems)]
        his = [s["metrics"]["test"]["ndcg@10"]["ci"][1] - v for v, s in zip(vals, systems)]
        fig, ax = plt.subplots(figsize=(6.3, 0.27 * len(systems) + 0.8))
        y = list(range(len(systems)))[::-1]
        ax.barh(y, vals, height=0.55, color=BLUE, xerr=[los, his], error_kw={"ecolor": INK, "elinewidth": 0.7, "capsize": 1.5})
        for yi, v, hi in zip(y, vals, his):
            ax.text(v + hi + 0.012, yi, "%.3f" % v, va="center", ha="left", fontsize=7.5, color=INK)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_xlim(0, min(1.0, max(v + h for v, h in zip(vals, his)) + 0.12))
        ax.set_xlabel("nDCG@10 on the %d test topics (95%% bootstrap interval)" % t["n_test"])
        ax.grid(axis="y", visible=False)
        ax.set_title("TREC-COVID retrieval quality by system", loc="left")
        _save(fig, fig_dirs, "fig_retrieval")
        plt.close(fig)
        made.append("fig_retrieval")

        if t.get("alpha_curve"):
            curve = t["alpha_curve"]
            xs = [c["alpha"] for c in curve]
            fig, ax = plt.subplots(figsize=(3.1, 2.2))
            if "dev" in curve[0]:
                ax.plot(xs, [c["dev"] for c in curve], color=NEUTRAL, linewidth=1.6, marker="o", markersize=3, label="dev topics")
            ax.plot(xs, [c["test"] for c in curve], color=BLUE, linewidth=1.6, marker="o", markersize=3, label="test topics")
            hyb = (t.get("hypothesis") or {}).get("hybrid")
            if hyb is not None:
                ax.axhline(hyb, color=ORANGE, linewidth=1.2, label="RRF, test topics")
            ax.set_xlabel("dense weight α  (0 = BM25 only, 1 = dense only)")
            ax.set_ylabel("nDCG@10")
            ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.3), ncol=3, handlelength=1.4, columnspacing=1.0)
            ax.set_title("Weighted fusion vs. RRF", loc="left")
            _save(fig, fig_dirs, "fig_alpha")
            plt.close(fig)
            made.append("fig_alpha")

        pq = Path(results) / "retrieval" / "trec-covid" / "per_query.csv"
        prim = (t.get("primary_dense_model") or "").split("/")[-1]
        if pq.exists() and prim:
            with open(pq) as fh:
                rows = [r for r in csv.DictReader(fh) if r["split"] == "test"]
            hk, dk = "hybrid_" + prim, "dense_" + prim
            if rows and hk in rows[0]:
                h = t.get("hypothesis") or {}
                single = "bm25" if h.get("bm25", 0) >= h.get("dense", 0) else dk
                diffs = sorted(float(r[hk]) - float(r[single]) for r in rows)
                fig, ax = plt.subplots(figsize=(3.1, 2.2))
                ax.bar(range(len(diffs)), diffs, width=0.7, color=[ORANGE if d < 0 else BLUE for d in diffs])
                ax.axhline(0, color=MUTED, linewidth=0.6)
                ax.set_xticks([])
                ax.set_xlabel("%d test topics, sorted" % len(diffs))
                ax.set_ylabel("Δ nDCG@10")
                ax.grid(axis="x", visible=False)
                ax.set_title("Hybrid minus %s, per topic" % ("BM25" if single == "bm25" else "dense"), loc="left")
                ax.text(0.02, 0.96, "orange: hybrid worse   blue: hybrid better", transform=ax.transAxes, fontsize=6.5, color=MUTED, va="top")
                _save(fig, fig_dirs, "fig_per_topic")
                plt.close(fig)
                made.append("fig_per_topic")

    pub = R.get("pub")
    rec_path = Path(results) / "answers" / "pubmedqa" / "records.jsonl"
    if pub and rec_path.exists() and pub["abstention"].get("signal"):
        sig, thr = pub["abstention"]["signal"], pub["abstention"].get("threshold_value")
        recs = [json.loads(line) for line in open(rec_path, encoding="utf-8") if line.strip()]
        groups = [("answerable", [r["signals"].get(sig) for r in recs if r["answerable"]], BLUE),
                  ("out-of-collection", [r["signals"].get(sig) for r in recs if not r["answerable"]], ORANGE)]
        if all(v is not None for _, vals_, _ in groups for v in vals_):
            import random as _random

            rng = _random.Random(0)
            fig, ax = plt.subplots(figsize=(3.1, 1.7))
            for i, (name, vals_, color) in enumerate(groups):
                ax.scatter(vals_, [i + rng.uniform(-0.22, 0.22) for _ in vals_], s=9, color=color, alpha=0.75, linewidths=0)
            if thr is not None:
                ax.axvline(thr, color=INK, linewidth=0.9)
                ax.text(thr, 1.47, " threshold", fontsize=6.5, color=INK, va="top", ha="left")
            ax.set_yticks([0, 1])
            ax.set_yticklabels(["answerable\n(n=%d)" % len(groups[0][1]), "out-of-collection\n(n=%d)" % len(groups[1][1])])
            ax.set_ylim(-0.5, 1.5)
            ax.grid(axis="y", visible=False)
            ax.set_xlabel(sig.replace("_", " ") + " (PubMedQA test questions)")
            ax.set_title("Retrieval confidence by question type", loc="left")
            _save(fig, fig_dirs, "fig_abstention")
            plt.close(fig)
            made.append("fig_abstention")

    judged = [(n, s) for n, s in (("PubMedQA", pub), ("Handbook", R.get("hb"))) if s and s.get("judge")]
    if judged:
        fig, ax = plt.subplots(figsize=(3.1, 0.55 * len(judged) + 0.9))
        for i, (name, s) in enumerate(judged[::-1]):
            j = s["judge"]
            left = 0.0
            for val, color, lab in ((j["rag_wins"], BLUE, "RAG wins"), (j["ties"], NEUTRAL, "tie"), (j["closed_book_wins"], ORANGE, "closed-book wins")):
                share = val / j["n"]
                ax.barh(i, share, left=left, height=0.5, color=color, edgecolor="white", linewidth=1.5, label=lab if i == 0 else None)
                if share > 0.09:
                    ax.text(left + share / 2, i, str(val), ha="center", va="center", fontsize=7.5, color="white" if color != NEUTRAL else INK)
                left += share
        ax.set_yticks(range(len(judged)))
        ax.set_yticklabels(["%s (n=%d)" % (n, s["judge"]["n"]) for n, s in judged[::-1]])
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
        ax.grid(axis="y", visible=False)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=3, handlelength=1.0, columnspacing=1.0)
        ax.set_title("Pairwise judge: RAG vs. closed-book", loc="left")
        _save(fig, fig_dirs, "fig_judge")
        plt.close(fig)
        made.append("fig_judge")

    c = R.get("chunk_pub")
    if c and len(c["rows"]) > 1:
        xs = [r["chunk_words"] for r in c["rows"]]
        fig, ax = plt.subplots(figsize=(3.1, 2.1))
        for key, color, lab in (("bm25", BLUE, "BM25"), ("dense", ORANGE, "dense"), ("hybrid", AQUA, "hybrid")):
            ys = [r["systems"][key]["mrr@10"] for r in c["rows"]]
            ax.plot(xs, ys, color=color, linewidth=1.6, marker="o", markersize=3.5, label=lab)
        ax.set_xscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels([str(x) for x in xs])
        ax.minorticks_off()
        ax.set_xlabel("passage length (words)")
        ax.set_ylabel("MRR@10")
        ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.3), ncol=3, handlelength=1.4, columnspacing=1.0)
        ax.set_title("Chunk size (PubMedQA retrieval)", loc="left")
        _save(fig, fig_dirs, "fig_chunk")
        plt.close(fig)
        made.append("fig_chunk")
    return made


# ---------------------------------------------------------------------------- writers

def write_all(results: Path, report_dir: Optional[Path]) -> Dict[str, Any]:
    results = Path(results)
    R = load_results(results)
    tables = retrieval_tables(R) + answer_tables(R) + ablation_tables(R)
    crit, facts = success_criteria(R), findings(R)
    fig_dirs = [results / "figures"] + ([Path(report_dir) / "figures"] if report_dir else [])
    figs = make_figures(R, results, fig_dirs)

    md: List[str] = ["# Results", ""]
    env = R.get("env") or {}
    pub = R.get("pub") or {}
    md.append("Generated by `scripts/06_make_report.py` from the files in `%s/`. Every number below was produced by the pipeline on this machine." % results.name)
    md.append("")
    md.append("- Machine: %s, %s cores, %s GB RAM, PyTorch device `%s`" % (env.get("platform", "?"), env.get("cpu_count", "?"), env.get("ram_gb", "?"), env.get("torch_device", "?")))
    md.append("- Generator LLM: `%s` · judge / verifier LLM: `%s` (both local, temperature 0)" % (pub.get("generator_model") or env.get("generator"), pub.get("judge_model") or env.get("judge")))
    t = R.get("trec")
    if t:
        md.append("- TREC-COVID: %s documents, %s passages of ~%s words; %d dev / %d test topics; dense models: %s" % (
            "{:,}".format(t["collection"]["n_documents"]), "{:,}".format(t["collection"]["n_passages"]), t["collection"]["chunk_words"],
            t["n_dev"], t["n_test"], ", ".join(nick(m) for m in t["collection"].get("dense_models", []))))
    if pub:
        md.append("- PubMedQA: %s abstracts indexed; %d answerable + %d out-of-collection test questions; top-k = %d" % (
            "{:,}".format(pub["collection"]["n_documents"]), pub["n_test_answerable"], pub["n_test_unanswerable"], pub["top_k"]))
    md.append("")
    if crit:
        md += ["## Success criteria from the proposal", "", "| Criterion | Outcome | Evidence |", "|---|---|---|"]
        md += ["| %s | **%s** | %s |" % (c["criterion"], c["outcome"], c["evidence"]) for c in crit]
        md.append("")
    if facts:
        md += ["## What the numbers show", ""] + ["- " + s for s in facts] + [""]
    titles = {"retrieval_trec": "Retrieval: TREC-COVID", "retrieval_tests": "Retrieval: paired significance tests", "ablation_fusion": "Ablation: fusion method",
              "latency_retrieval": "Retrieval latency", "retrieval_pubmedqa": "Retrieval: PubMedQA (known item)", "judge": "Answers: pairwise judge",
              "decision": "Answers: yes / no / maybe accuracy", "grounding": "Groundedness: citations and faithfulness", "abstention": "Abstention and over-refusal",
              "signals": "Abstention signals", "ablation_topk": "Ablation: top-k", "failures": "Failure categories", "ablation_chunk": "Ablation: chunk size (PubMedQA)",
              "ablation_chunk_trec": "Ablation: chunk size (TREC-COVID)", "ablation_modes": "Ablation: retrieval mode, end to end"}
    fig_after = {"retrieval_trec": ["fig_retrieval", "fig_per_topic"], "ablation_fusion": ["fig_alpha"], "judge": ["fig_judge"],
                 "signals": ["fig_abstention"], "ablation_chunk": ["fig_chunk"]}
    for tb in tables:
        md += ["## " + titles.get(tb.name, tb.name), "", tb.caption, "", tb.markdown(), ""]
        for f in fig_after.get(tb.name, []):
            if f in figs:
                md += ["![%s](figures/%s.png)" % (f, f), ""]
    hc = R.get("handcheck")
    if hc:
        md += ["## Human validation of the judge", "", "%d judgments hand-checked: agreement with the LLM judge %s%%, Cohen's kappa %s." % (
            hc["n"], pct(hc["agreement"]), f3(hc["kappa"], 2)), ""]
    else:
        md += ["## Human validation of the judge", "", "Not done yet. Fill in the `human_verdict` column of `answers/pubmedqa/judge_handcheck.csv` "
               "(1, 2 or tie) and run `python scripts/score_handcheck.py`.", ""]
    v = R.get("nli_validation")
    if v:
        md += ["## Validation of the citation NLI model", "",
               "%d cited (statement, passage) pairs, NLI (`%s`) against the judge LLM (`%s`, which finds %s%% entailed). "
               "The configured setting is `eval.citation_nli_window: %d` (0 = whole passage)." % (
                   v["n_pairs"], v["nli_model"], v["judge_model"], pct(v["judge_entailed_rate"]), v["configured_window"]), "",
               "| NLI premise | entailed | agreement with judge | Cohen's kappa |", "|---|---:|---:|---:|"]
        for w, s in sorted(v["windows"].items(), key=lambda kv: int(kv[0])):
            md.append("| %s | %s%% | %s%% | %s |" % ("whole passage" if w == "0" else "%s-sentence windows" % w,
                                                   pct(s["entailed_rate"]), pct(s["agreement"]), f3(s["kappa"], 2)))
        md.append("")
    usage = R.get("usage")
    if usage:
        md += ["## Cost", "", "LLM calls in the last answer-evaluation run: generator %d new + %d cached (%.0f s), judge %d new + %d cached (%.0f s)." % (
            usage["generator"]["calls"], usage["generator"]["cache_hits"], usage["generator"]["seconds"],
            usage["judge"]["calls"], usage["judge"]["cache_hits"], usage["judge"]["seconds"]), ""]
    if pub.get("latency"):
        la = pub["latency"]
        md += ["Mean time per question on PubMedQA: retrieval %.2f s, RAG generation %.1f s (%.0f prompt tokens), closed-book generation %.1f s. "
               "Cached calls report the time of the original call." % (la["retrieval_seconds_mean"], la["rag_seconds_mean"], la["rag_prompt_tokens_mean"], la["closed_book_seconds_mean"]), ""]
    md += ["## Caveats", "",
           "- TREC-COVID has 50 topics; with a %s-topic test split the confidence intervals are wide. Treat differences without a small p-value as inconclusive." % (t["n_test"] if t else "small"),
           "- The LLM judge and the faithfulness check are themselves models of about the same size as the generator; the hand-check above is the guard against judge error.",
           "- ALCE-style citation precision says a cited passage supports the sentence, not that the model actually used it (Wallat et al., 2024).",
           "- The out-of-collection questions are PubMedQA questions whose source abstract was withheld; a related abstract can occasionally still contain an answer.",
           "- The handbook is a small fictional collection written for the demo, with hand-written questions.", ""]
    (results / "RESULTS.md").write_text("\n".join(md), encoding="utf-8")

    if report_dir:
        gen = Path(report_dir) / "generated"
        gen.mkdir(parents=True, exist_ok=True)
        for old in gen.glob("*.tex"):
            old.unlink()
        for tb in tables:
            (gen / ("tab_%s.tex" % tb.name)).write_text(tb.latex(), encoding="utf-8")
        lines = ["\\providecommand{\\%s}{%s}" % (k, v) for k, v in macros(R).items()]
        (gen / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
        if facts:
            (gen / "findings.tex").write_text("\n\n".join("\\noindent\\textbf{%s} %s" % (tex_cell(s.split(". ", 1)[0] + "."), tex_cell(s.split(". ", 1)[1]))
                                                             if re.match(r"^[A-Z][a-z]+( [a-z]+){0,2}\. ", s) else tex_cell(s) for s in facts) + "\n", encoding="utf-8")
        if crit:
            rows = [[c["criterion"], c["outcome"], c["evidence"]] for c in crit]
            body = ["\\begin{table*}[t]", "\\centering\\small", "\\begin{tabular}{p{0.40\\textwidth}p{0.09\\textwidth}p{0.43\\textwidth}}", "\\toprule",
                    "Success criterion (proposal) & Outcome & Evidence \\\\", "\\midrule"]
            body += ["%s & \\textbf{%s} & %s \\\\" % (tex_cell(a), tex_cell(b), tex_cell(c_)) for a, b, c_ in rows]
            body += ["\\bottomrule", "\\end{tabular}", "\\caption{The three success criteria stated in the proposal, checked against the results.}",
                     "\\label{tab:criteria}", "\\end{table*}"]
            (gen / "tab_criteria.tex").write_text("\n".join(body) + "\n", encoding="utf-8")
    return {"tables": [tb.name for tb in tables], "figures": figs, "criteria": crit, "findings": facts}
