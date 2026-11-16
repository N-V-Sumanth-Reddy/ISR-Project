"""Fetch public web pages or PDFs into a folder, so they can be indexed as a collection.

    python scripts/crawl_pages.py --urls my_urls.txt --out data/policies
    python scripts/02_build_indexes.py --folder data/policies --name policies

my_urls.txt holds one URL per line (lines starting with # are ignored). Only fetch pages
you are allowed to copy; this is the single script in the project that contacts the web
on purpose, and it is never called by the app or the evaluation.
"""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
from urllib.parse import urlparse

import requests


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--urls", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--delay", type=float, default=1.0, help="seconds between requests")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    urls = [u.strip() for u in Path(args.urls).read_text().splitlines() if u.strip() and not u.strip().startswith("#")]
    ok = 0
    for url in urls:
        try:
            r = requests.get(url, timeout=30, headers={"User-Agent": "course-project-rag-demo/1.0"})
            r.raise_for_status()
        except requests.RequestException as exc:
            print("skip %s (%s)" % (url, exc))
            continue
        parsed = urlparse(url)
        stem = re.sub(r"[^A-Za-z0-9]+", "_", (parsed.netloc + parsed.path).strip("/"))[:120] or "page"
        ctype = r.headers.get("content-type", "")
        ext = ".pdf" if ("pdf" in ctype or parsed.path.lower().endswith(".pdf")) else ".html"
        (out / (stem + ext)).write_bytes(r.content)
        ok += 1
        print("saved %s -> %s" % (url, stem + ext))
        time.sleep(args.delay)
    print("%d of %d pages saved to %s" % (ok, len(urls), out))


if __name__ == "__main__":
    main()
