#!/usr/bin/env python3
"""Quick matched prefill/decode probe for the Qwen3.8-Flash-Next recipes.

Mirrors the campaign methodology at probe scale:
  prefill: one cold random-token request at 8192 tokens, server-reported duration;
  decode : 512 forced-output tokens at C1 and C8, aggregate rate over the all-active window.
Usage: ./bench-quick.py [endpoint] [model] [repeats]
"""
import concurrent.futures as cf
import json
import statistics
import sys
import time
import urllib.request

ENDPOINT = sys.argv[1] if len(sys.argv) > 1 else "http://maxwell:8000/v1"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "Qwen3.8-Flash-Next"
REPEATS = int(sys.argv[3]) if len(sys.argv) > 3 else 3

def post(body, timeout=3600):
    req = urllib.request.Request(
        f"{ENDPOINT}/chat/completions",
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)

def prefill(n=8192):
    # Deterministic random token IDs; >=block-size so the pool cannot reuse blocks.
    ids = [(i * 2654435761) % 150000 + 1000 for i in range(n)]
    t0 = time.monotonic()
    r = post_chat(ids)
    dt = time.monotonic() - t0
    usage = r.get("usage", {})
    tok = usage.get("prompt_tokens", n)
    return tok / dt if dt else float("nan")

def post_chat(ids):
    # vLLM accepts prompt_token_ids on /v1/completions; use it directly.
    req = urllib.request.Request(
        f"{ENDPOINT}/completions",
        data=json.dumps({
            "model": MODEL, "max_tokens": 1, "temperature": 0,
            "prompt": ids,
        }).encode(),
        headers={"content-type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.load(r)

def decode_cell(concurrency, ctx=0):
    warm = [(i * 40503) % 150000 + 1000 for i in range(min(ctx, 4096))] if ctx else []
    def one(i):
        req = {
            "model": MODEL, "max_tokens": 512, "temperature": 0,
            "ignore_eos": True,
            "messages": [{"role": "user", "content": "Count from 1 upward forever."}],
        }
        if warm:
            req["messages"] = [{"role": "user", "content": " ".join(str(x) for x in warm) + " Now count from 1 upward forever."}]
        r = post(req)
        return r["usage"].get("completion_tokens", 512)
    with cf.ThreadPoolExecutor(concurrency) as ex:
        t0 = time.monotonic()
        outs = list(ex.map(one, range(concurrency)))
        dt = time.monotonic() - t0
    return sum(outs) / dt

def main():
    results = {}
    results["prefill-8192"] = []
    for _ in range(REPEATS):
        results["prefill-8192"].append(prefill())
    for label, c in (("decode-ctx0-c1", 1), ("decode-ctx0-c8", 8),
                     ("decode-ctx8k-c1", 1), ("decode-ctx8k-c8", 8)):
        ctx = 8192 if "8k" in label else 0
        results[label] = [decode_cell(c, ctx) for _ in range(REPEATS)]
    out = {k: {"mean": round(statistics.mean(v), 1),
               "stdev": round(statistics.stdev(v), 1) if len(v) > 1 else 0.0,
               "samples": [round(x, 1) for x in v]} for k, v in results.items()}
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
