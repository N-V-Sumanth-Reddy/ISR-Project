# Private, citation-grounded RAG search over local documents

A search engine for document collections that must not leave the machine. It indexes your
PDF / text files locally, retrieves passages with a hybrid of BM25 and dense embeddings, and has
a locally hosted LLM answer **only** from those passages, with inline citations like `[1][2]`.
When the passages do not contain the answer it says "Not found in the knowledge base."

Course project by Sai Nithin Maddi, Sumanth Reddy and Zhiyan Song.

## Quick start (macOS, Apple Silicon)

You need [Ollama](https://ollama.com) installed and Python 3.10 or newer. Then, from this folder:

```bash
./run_all.sh smoke     # ~10-15 min plus downloads: tiny subsets, proves every stage works here
./run_all.sh           # the evaluation described in the proposal (a few hours; resumable)
./run_app.sh           # the web UI at http://localhost:8501
```

`run_all.sh` creates `.venv`, installs the packages, pulls the two LLMs, downloads the public
data and retrieval models, builds the indexes, runs every evaluation and writes the tables and
figures. It is safe to interrupt and restart: indexes are kept and every LLM call is cached.

| Profile | What it runs | Rough time on an M-series laptop |
|---|---|---|
| `smoke` | 4,000 TREC-COVID documents, 12 PubMedQA questions, 8 handbook questions. Numbers are meaningless; it only checks the pipeline. Writes to `results_smoke/`. | 10-15 min |
| `standard` (default) | Full TREC-COVID (171K documents, two encoders), PubMedQA (100 answerable + 30 out-of-collection test questions), handbook, fusion and chunk-size ablations. | 2-5 h, mostly LLM calls |
| `full` | Standard, plus the chunk-size ablation on TREC-COVID and the end-to-end retrieval-mode ablation. Reuses everything `standard` already computed. | several more hours; run overnight |

Disk: about 10 GB for the two LLMs, 1.5 GB for the retrieval models, 2 GB for data and indexes.
The first run needs the internet for those downloads; after that nothing does.

On a machine with less than 14 GB of memory the environment check switches to 3B models
(`llama3.2:3b`, `qwen2.5:3b`) and says so; results then describe those models.

## Where the results are

After a run, open **`results/RESULTS.md`**. It holds every table, the figures, and a verdict on
the three success criteria from the proposal. The same tables are written as LaTeX to
`report/generated/` and the figures to `report/figures/`, so `report/main.tex` compiles into the
final report with the real numbers (upload the `report/` folder to Overleaf, or run
`pdflatex main && bibtex main && pdflatex main && pdflatex main`).

| File | Content |
|---|---|
| `results/RESULTS.md` | all tables and figures, generated from the files below |
| `results/retrieval/trec-covid/metrics.json`, `per_query.csv`, `runs/` | nDCG@10, Recall@100 per system with confidence intervals and paired tests; BM25 grid; fusion curves |
| `results/answers/pubmedqa/summary.json` | judge outcome, label accuracy, citation precision / recall, faithfulness, abstention, latency |
| `results/answers/pubmedqa/records.jsonl` | every test question with the retrieved passages, both answers, the judge's verdicts and all checks |
| `results/answers/pubmedqa/failures.csv` | one row per question with its failure category, for the error analysis |
| `results/answers/pubmedqa/judge_handcheck.csv` | 50 judged comparisons for the human validation (see below) |
| `results/answers/citation_nli_validation.json` | the citation NLI model against the judge LLM, whole passage vs. sentence windows (why `eval.citation_nli_window` is 2) |
| `results/ablations/*.json` | chunk size and retrieval-mode ablations |

### Human validation of the judge (the one manual step)

The proposal promises to validate the LLM judge by hand-checking 50 judgments. Open
`results/answers/pubmedqa/judge_handcheck.csv`; the two answers are unlabeled and in random
order. In the `human_verdict` column write `1`, `2` or `tie` for the answer that agrees better
with the reference answer. Then:

```bash
source .venv/bin/activate
python scripts/score_handcheck.py      # agreement and Cohen's kappa with the judge
python scripts/06_make_report.py       # adds the number to RESULTS.md
```

## The web UI

`./run_app.sh` opens the app on the demo handbook. It shows the answer with its citations, the
exact passages with file and page, and a badge telling whether BM25, dense retrieval or both
found each passage. The settings panel switches the retrieval mode, top-k, re-ranking, the entity
graph and the abstention threshold, and can show the same model's closed-book answer next to the
grounded one. "Add your own documents" in the sidebar indexes uploaded files on the spot.

A 3-4 minute demo, as planned in the proposal:

1. Collection `handbook`, ask *"What is the protocol for a missed insulin dose?"* with the
   closed-book answer switched on. Point at the cited passage and its source.
2. Ask *"What is the monthly parking permit fee for staff?"* (not in the handbook): the system
   declines, while the closed-book model may answer anyway.
3. Ask *"When is premium economy airfare allowed?"* and toggle BM25 / Dense / Hybrid; the badges
   and ranks in the evidence panel change. The answer cites page numbers of the PDF.
4. Open the "Evidence graph" tab, then switch the collection to `pubmedqa` or `trec-covid`.

## Your own documents

```bash
source .venv/bin/activate
python scripts/02_build_indexes.py --folder ~/path/to/docs --name mydocs
python scripts/ask.py "your question" --collection mydocs --closed-book
```

PDF, text, Markdown and HTML are supported. Re-run the build command after changing files;
updating the knowledge base is re-indexing, never retraining.

## How it works

| Stage | Implementation |
|---|---|
| Chunking | sentence-aware passages of at most 200 words with 40 words of overlap; file and page kept (`ragsearch/chunking.py`, `ingest.py`) |
| BM25 | own implementation, Lucene idf, stop words, Porter stemming; `k1`, `b` tuned on dev without re-indexing (`bm25.py`) |
| Dense | `all-MiniLM-L6-v2` (primary) and `bge-small-en-v1.5`, 384-d, exact inner-product search with FAISS (`dense.py`) |
| Fusion | Reciprocal Rank Fusion, k = 60, over the two top-100 lists; also weighted score fusion and a DAT-style per-query weight (`fusion.py`) |
| Re-ranking | optional cross-encoder `ms-marco-MiniLM-L-6-v2` over the top 50 (`rerank.py`) |
| Generation | Ollama, temperature 0; numbered passages in, cited answer or "Not found in the knowledge base." out (`generate.py`, `llm.py`) |
| Abstention | the LLM's own refusal, plus a retrieval-confidence threshold tuned on dev questions (`eval/abstention.py`) |
| Post-check | cited numbers must exist; uncited sentences and unsupported numbers are flagged (`generate.py`) |
| Entity graph | optional passage-term graph used for expansion / re-ranking, an answer check, and the evidence-graph view (`kg.py`) |
| Evaluation | nDCG@10, Recall@100; pairwise LLM judge in both orders; label accuracy; ALCE-style citation precision / recall (NLI on 2-sentence windows of each passage); RAGAs-style faithfulness; abstention and over-refusal (`eval/`) |

Settings live in `config.yaml`. Any of them can be overridden on the command line, for example
`python scripts/04_eval_answers.py --set llm.model=qwen2.5:7b --set llm.judge_model=llama3.1:8b`.

## Data

* **TREC-COVID** (BEIR release) for retrieval: 50 topics, split 15 dev / 35 test by a fixed seed.
* **PubMedQA** (expert-labelled) for answers. 100 answerable test questions; 30 more whose source
  abstract is withheld from the index, so the collection cannot answer them; 40 + 20 dev
  questions for tuning. The searchable corpus is the other 950 abstracts, without their titles
  (the titles are the questions), conclusions (the reference answers) or labels; only the
  evaluation sees those. `tests/test_datasets.py` checks that none of them reaches the index.
* **Handbook**: a fictional policy handbook written for this project (`data/handbook/`), with 25
  answerable and 10 out-of-collection hand-written questions. It replaces the crawl of real
  university policy pages mentioned in the proposal, so the repository can be shared and the
  demo runs offline. `scripts/crawl_pages.py` fetches public pages if you want real ones.

## Privacy

* Indexes are plain local files; the retrieval models run in-process.
* The LLM client only talks to `localhost`. A non-local host is refused unless
  `llm.allow_remote: true` is set on purpose.
* The app sets `HF_HUB_OFFLINE=1`, so the Hugging Face libraries cannot contact the network, and
  Streamlit's usage statistics are switched off (`.streamlit/config.toml`).
* Nothing is fine-tuned on the documents.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest tests -q
```

The tests run offline. BM25 is checked against a direct evaluation of the formula and against
`rank_bm25`, the metrics against `pytrec_eval` (trec_eval), and the whole answer pipeline runs
end to end against `tests/fake_ollama.py`, a stand-in server that is **not** a language model.

## Troubleshooting

* *"The local LLM server is not reachable"*: open the Ollama app (or run `ollama serve`).
* *A model is missing*: `ollama pull llama3.1:8b` and `ollama pull qwen2.5:7b`.
* *TREC-COVID does not download*: the script prints the URL; unzip it into `data/` so that
  `data/trec-covid/corpus.jsonl` exists, then re-run.
* *Out of memory during generation*: use smaller models, e.g.
  `./run_all.sh` after adding `llm: {model: llama3.2:3b, judge_model: qwen2.5:3b}` to
  `config.local.yaml`.
* *Start over*: delete `indexes/` and `results/` (or `indexes_smoke/`, `results_smoke/`).

## Repository layout

```
ragsearch/        the library: chunking, BM25, dense, fusion, re-ranking, KG, generation, eval
scripts/          00_check_env ... 06_make_report, ask.py, score_handcheck.py, crawl_pages.py
app/              Streamlit UI
data/handbook/    demo documents and hand-written questions
report/           ACL-style LaTeX report; generated/ and figures/ are filled by the pipeline
tests/            unit and integration tests
run_all.sh        one-command reproduction     run_app.sh   start the UI
```
