#!/usr/bin/env python3
"""Quick matched prefill/decode probe for the Qwen3.8-Flash-Next recipes.

Methodology (mirrors the campaign at probe scale):
  prefill: N cold requests of 8192 fresh random token IDs (salted per repeat, prefix cache
           reset before each); rate = DELTA of server counters
           prompt_tokens_by_source{source=processed} over wall — cache hits are excluded
           by construction and visible in the `cached` field.
  decode : 512 forced-output tokens on an exact-length random token-ID prompt (cold prefix
           per sample); C1 and C8, aggregate over the all-decoding window.
Usage: ./bench-quick.py [endpoint] [model] [repeats]
"""
import concurrent.futures as cf
import json
import re
import statistics
import sys
import time
import urllib.request

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://maxwell:8000/v1").rsplit("/v1", 1)[0]
ENDPOINT = BASE + "/v1"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "Qwen3.8-Flash-Next"
REPEATS = int(sys.argv[3]) if len(sys.argv) > 3 else 3

def post(path, body, timeout=3600):
    req = urllib.request.Request(ENDPOINT + path, data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)

def counters():
    req = urllib.request.Request(BASE + "/metrics")
    with urllib.request.urlopen(req, timeout=30) as r:
        body = r.read().decode()
    out = {"processed": 0.0, "cached": 0.0}  # sources: local_compute | local_cache_hit
    for line in body.splitlines():
        m = re.match(r'vllm:prompt_tokens_by_source_total\{[^}]*source="local_compute"[^}]*\}\s+(\S+)', line)
        if m: out["processed"] = float(m.group(1))
        m = re.match(r'vllm:prompt_tokens_by_source_total\{[^}]*source="local_cache_hit"[^}]*\}\s+(\S+)', line)
        if m: out["cached"] = float(m.group(1))
    return out

_SALT0 = time.time_ns() % 100000  # never-measured salts; this server build has NO cache-reset endpoint

def reset_prefix_cache():
    pass  # verified 404 on eugr nightly-20260925: coldness must come from unique salts

def ids(n, salt):
    s = _SALT0 + salt * 7919
    return [((i + s) * 2654435761) % 150000 + 1000 for i in range(n)]

def prefill(n=8192, salt=1):
    reset_prefix_cache()
    c0 = counters()
    t0 = time.monotonic()
    post("/completions", {"model": MODEL, "prompt": ids(n, salt), "max_tokens": 1, "temperature": 0})
    dt = time.monotonic() - t0
    c1 = counters()
    processed = c1["processed"] - c0["processed"]
    cached = c1["cached"] - c0["cached"]
    return {"rate": (processed / dt if dt else 0.0), "processed": processed, "cached": cached}

def decode_cell(concurrency, ctx=0, salt=1):
    warm = ids(ctx, salt) if ctx else []
    reset_prefix_cache()
    if warm:
        # PRIME the shared context OUTSIDE the timed window (one 1-token request; the prime
        # pays the cold prefill, measured requests hit it from prefix cache);
        # measured requests then reuse it from prefix cache and the window measures
        # decode only — matching the campaign's decode-at-context semantics.
        post("/completions", {"model": MODEL, "prompt": warm, "max_tokens": 1, "temperature": 0})
    def one(i):
        r = post("/completions", {
            "model": MODEL, "prompt": warm + ids(32, salt * 31 + i),
            "max_tokens": 512, "temperature": 0, "ignore_eos": True,
        })
        return r["usage"].get("completion_tokens", 512)
    with cf.ThreadPoolExecutor(concurrency) as ex:
        t0 = time.monotonic()
        outs = list(ex.map(one, range(concurrency)))
        dt = time.monotonic() - t0
    return sum(outs) / dt

def main():
    out = {}
    pf = []
    for it in range(REPEATS):
        pf.append(prefill(salt=it + 1))
    out["prefill-8192"] = {
        "mean": round(statistics.mean(x["rate"] for x in pf), 1),
        "stdev": round(statistics.stdev((x["rate"] for x in pf), ), 1) if REPEATS > 1 else 0.0,
        "samples": [round(x["rate"], 1) for x in pf],
        "cached_tokens_per_request": [x["cached"] for x in pf],
    }
    for label, c, ctx in (("decode-ctx0-c1", 1, 0), ("decode-ctx0-c8", 8, 0),
                          ("decode-ctx8k-c1", 1, 8192), ("decode-ctx8k-c8", 8, 8192)):
        s = [round(decode_cell(c, ctx, salt=k + hash(label) % 1000), 1) for k in range(REPEATS)]
        out[label] = {"mean": round(statistics.mean(s), 1),
                      "stdev": round(statistics.stdev(s), 1) if REPEATS > 1 else 0.0,
                      "samples": s}
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
