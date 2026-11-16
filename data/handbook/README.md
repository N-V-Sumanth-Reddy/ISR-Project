# Demo "enterprise handbook"

`docs/` holds a small, **fictional** policy handbook for the invented "Brightwater Institute"
(ten Markdown files and two multi-page PDFs). It was written for this project so the demo and the
tests work offline and nothing copyrighted or private is redistributed. No rule in it describes a
real institution, and the clinic protocol is not medical guidance.

`questions.json` holds 35 hand-written questions: 25 that the handbook answers (with a reference
answer and the file that contains it) and 10 that it does not (three of them are deliberate
near-misses, e.g. the *international* lodging limit when only domestic limits are given).

To use real documents instead, either drop your own PDF / text / Markdown files into a folder and
index it (`python scripts/02_build_indexes.py --folder path/to/docs --name mydocs`), or fetch public
pages with `python scripts/crawl_pages.py --urls my_urls.txt --out data/mydocs` first.
