#!/usr/bin/env bash
# Start the web UI at http://localhost:8501
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -d .venv ]; then echo "Run ./run_all.sh smoke first (it creates the Python environment)."; exit 1; fi
# shellcheck disable=SC1091
source .venv/bin/activate
python - <<'PY'
import sys
sys.path.insert(0, ".")
from ragsearch.collection import Collection, list_collections
from ragsearch.config import load_config
cfg = load_config()
if "handbook" not in list_collections(cfg):
    from ragsearch.ingest import read_folder
    print("building the demo handbook index ...")
    Collection.build("handbook", read_folder("data/handbook/docs"), cfg)
    import json, shutil
    qs = json.load(open("data/handbook/questions.json"))
    json.dump({"test": qs}, open("indexes/handbook/benchmark.json", "w"))
PY
exec streamlit run app/streamlit_app.py
