"""A tiny stand-in for the Ollama HTTP API (/api/tags, /api/chat) used by the tests.

It is NOT a language model. It answers with simple word-overlap heuristics so that every
stage of the pipeline (generation, judging, claim extraction, verification) receives
well-formed replies offline. Numbers produced with it are meaningless and must never be
reported as results.

    python tests/fake_ollama.py 11500      # then: --set llm.host=http://localhost:11500
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODELS = ["llama3.1:8b", "qwen2.5:7b", "llama3.2:3b", "qwen2.5:3b"]
STOP = set("the a an of and or to in is are was were for on with by does do did what how when which who that "
           "this it be as at from can i my we you there their than if not no yes".split())


def _words(text: str):
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP and len(w) > 2}


def _first_sentence(text: str) -> str:
    return re.split(r"(?<=[.!?])\s+", text.strip())[0][:300]


def reply(messages, json_mode: bool) -> str:
    system = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
    user = messages[-1]["content"]
    h = int(hashlib.md5(user.encode("utf-8")).hexdigest(), 16)

    if "numbered passages" in system:                                   # grounded generation
        question = user.split("Question:")[-1].split("\n")[0]
        blocks = re.findall(r"\[(\d+)\](?: \([^)]*\))? (.*?)(?=\n\n\[\d+\]|\n\nQuestion:)", user, flags=re.S)
        q = _words(question)
        scored = sorted(((len(q & _words(body)) / max(len(q), 1), int(num), body) for num, body in blocks), reverse=True)
        if not scored or scored[0][0] < 0.5:
            return "Not found in the knowledge base."
        _, num, body = scored[0]
        lead = "Yes, " if "Start your answer with" in user else ""
        extra = " This is consistent with related evidence [%d]." % scored[1][1] if len(scored) > 1 and h % 3 == 0 else ""
        return "%s%s [%d].%s" % (lead, _first_sentence(body).rstrip("."), num, extra)
    if "from your own knowledge" in system:                             # closed book
        if h % 7 == 0:
            return "I don't know."
        return ["Yes, published studies generally support this.", "No, the available evidence does not support this.",
                "Maybe, the evidence is mixed and further studies are needed."][h % 3]
    if "impartial grader" in system:                                    # pairwise judge
        ref = _words(user.split("Reference answer:")[-1].split("Answer A:")[0])
        a = _words(user.split("Answer A:")[-1].split("Answer B:")[0])
        b = _words(user.split("Answer B:")[-1])
        sa, sb = len(ref & a), len(ref & b)
        return json.dumps({"reason": "overlap with the reference", "winner": "A" if sa > sb else "B" if sb > sa else "tie"})
    if "atomic factual claims" in system:
        answer = user.split("Answer:")[-1]
        claims = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if len(s.strip()) > 8]
        return json.dumps({"claims": claims[:5]})
    if "verify claims" in system:
        ctx, claims = user.split("Claims:")[0], user.split("Claims:")[-1]
        cw = _words(ctx)
        out = []
        for line in claims.strip().splitlines():
            m = re.match(r"\s*(\d+)\.\s*(.*)", line)
            if m:
                w = _words(m.group(2))
                out.append({"claim": int(m.group(1)), "supported": len(w & cw) >= 0.6 * max(len(w), 1)})
        return json.dumps({"verdicts": out})
    if "supports a statement" in system:
        p, s = _words(user.split("Statement:")[0]), _words(user.split("Statement:")[-1])
        return json.dumps({"supported": len(p & s) >= 0.6 * max(len(s), 1)})
    if "what the answer concludes" in system:
        ans = user.split("Answer:")[-1].strip().lower()
        return json.dumps({"decision": "yes" if ans.startswith("yes") else "no" if ans.startswith("no") else "maybe"})
    if "0-5 scale" in system:
        q, p = _words(user.split("Passage:")[0]), _words(user.split("Passage:")[-1])
        return json.dumps({"score": min(5, int(6 * len(q & p) / max(len(q), 1)))})
    return "{}" if json_mode else "OK."


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # silence
        pass

    def _send(self, obj, status: int = 200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/tags":
            self._send({"models": [{"name": m, "model": m} for m in MODELS]})
        else:
            self._send({"error": "not found"}, 404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        payload = json.loads(self.rfile.read(length) or b"{}")
        if self.path != "/api/chat":
            self._send({"error": "not found"}, 404)
            return
        if payload.get("model") not in MODELS:
            self._send({"error": "model %r not found" % payload.get("model")}, 404)
            return
        text = reply(payload.get("messages", []), payload.get("format") == "json")
        prompt = sum(len(m["content"].split()) for m in payload.get("messages", []))
        self._send({"model": payload["model"], "message": {"role": "assistant", "content": text}, "done": True,
                    "prompt_eval_count": int(prompt * 1.3), "eval_count": int(len(text.split()) * 1.3),
                    "total_duration": 1000000})


def serve(port: int = 11500) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 11500
    print("fake Ollama on http://127.0.0.1:%d (test stand-in, not a language model)" % port, flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
