#!/usr/bin/env bash
# One command to reproduce every result in the report.
#
#   ./run_all.sh smoke      # ~10 minutes: tiny subsets, checks that every stage works here
#   ./run_all.sh            # "standard": the evaluation described in the proposal
#   ./run_all.sh full       # standard + the slower ablations (run it overnight)
#
# The run is resumable: indexes are kept, and every LLM call is cached on disk, so
# re-running after an interruption (or going from standard to full) repeats nothing.
set -euo pipefail
cd "$(dirname "$0")"
PROFILE="${1:-standard}"
case "$PROFILE" in smoke|standard|full) ;; *) echo "usage: ./run_all.sh [smoke|standard|full]"; exit 2 ;; esac

# ---------------------------------------------------------------- Python environment
if [ ! -d .venv ]; then
  PY=""
  # skip pre-release interpreters (e.g. Ubuntu 22.04's python3.11 is 3.11.0rc1, on which PyTorch fails to import)
  for cand in python3.12 python3.11 python3.13 python3.10 python3; do
    if command -v "$cand" >/dev/null 2>&1 && \
       "$cand" -c 'import sys; sys.exit(not (sys.version_info >= (3, 10) and sys.version_info.releaselevel == "final"))' 2>/dev/null; then
      PY="$cand"; break
    fi
  done
  [ -n "$PY" ] || { echo "Python 3.10+ (a final release) is required (https://www.python.org/downloads/)"; exit 1; }
  echo "creating .venv with $($PY --version)"
  "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt
# optional: the tokenizer backend of the NLI model; if it cannot be installed here, citation
# checks fall back to the judge LLM
python -m pip install --quiet sentencepiece protobuf || echo "(optional packages sentencepiece / protobuf not installed)"

RESULTS="results"; [ "$PROFILE" = "smoke" ] && RESULTS="results_smoke"
mkdir -p "$RESULTS"
LOG="$RESULTS/run_${PROFILE}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "profile: $PROFILE   log: $LOG   started: $(date)"

# keep a laptop awake for the duration of the run (macOS)
if command -v caffeinate >/dev/null 2>&1; then caffeinate -i -w $$ & fi

# ------------------------------------------------------------------------ local LLM
if ! curl -s --max-time 3 http://localhost:11434/api/tags >/dev/null 2>&1; then
  echo "Ollama is not running; trying to start it ..."
  if [ "$(uname)" = "Darwin" ] && [ -d "/Applications/Ollama.app" ]; then open -a Ollama || true
  elif command -v ollama >/dev/null 2>&1; then (ollama serve >/dev/null 2>&1 &) ; fi
  for _ in $(seq 1 30); do
    curl -s --max-time 2 http://localhost:11434/api/tags >/dev/null 2>&1 && break
    sleep 1
  done
fi

stage() { echo; echo "######## $1  ($(date +%H:%M:%S))"; shift; "$@"; }

stage "1/8 environment check"      python scripts/00_check_env.py      --profile "$PROFILE" --pull
stage "2/8 data and models"        python scripts/01_download_data.py  --profile "$PROFILE"
stage "3/8 indexes"                python scripts/02_build_indexes.py  --profile "$PROFILE"
stage "4/8 retrieval evaluation"   python scripts/03_eval_retrieval.py --profile "$PROFILE" --dat
stage "5/8 answer evaluation"      python scripts/04_eval_answers.py   --profile "$PROFILE"
stage "6/8 citation NLI check"     python scripts/validate_citation_nli.py --profile "$PROFILE"
stage "7/8 ablations"              python scripts/05_ablations.py      --profile "$PROFILE"
stage "8/8 tables and figures"     python scripts/06_make_report.py    --profile "$PROFILE"

echo
echo "Finished: $(date)"
echo "Results:  $RESULTS/RESULTS.md   (tables, figures and the three success criteria)"
if [ "$PROFILE" = "smoke" ]; then
  echo "The smoke run only proves the pipeline works on this machine; its numbers mean nothing."
  echo "Next:     ./run_all.sh        (or ./run_all.sh full overnight)"
else
  echo "Report:   report/generated/ and report/figures/ are filled in; compile report/main.tex"
  echo "Demo:     ./run_app.sh"
fi
