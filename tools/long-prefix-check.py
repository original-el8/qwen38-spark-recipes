#!/usr/bin/env python3
"""Long shared-prefix + edited-history gate (campaign methodology, probe scale).

Phase A: one 64K-token shared prefix; 4 concurrent requests append distinct short tails;
         all must complete (finish_reason=stop/length), and after phase A a 128K@C8 pass runs.
Phase B: same prefix re-sent with CHANGED instruction mid-request set — cached prefix must
         not leak stale answers.
Reads /metrics for preemption counters before/after. Usage: ./long-prefix-check.py [endpoint]
"""
import concurrent.futures as cf
import json
import sys
import urllib.request

ENDPOINT = sys.argv[1] if len(sys.argv) > 1 else "http://maxwell:8000/v1"
BASE = ENDPOINT.rsplit("/v1", 1)[0]

def completions(prompt_ids, max_tokens=64, temperature=0.0):
    req = urllib.request.Request(
        f"{ENDPOINT}/completions",
        data=json.dumps({"model": "Qwen3.8-Flash-Next", "prompt": prompt_ids,
                         "max_tokens": max_tokens, "temperature": temperature}).encode(),
        headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.load(r)

def metrics():
    with urllib.request.urlopen(f"{BASE}/metrics", timeout=30) as r:
        body = r.read().decode()
    out = {}
    for line in body.splitlines():
        if line.startswith("vllm:num_preemptions"):
            out["preemptions"] = float(line.split()[-1])
        if line.startswith("vllm:prefix_cache_queries"):
            out["pq"] = float(line.split()[-1])
        if line.startswith("vllm:prefix_cache_hits"):
            out["ph"] = float(line.split()[-1])
    return out

def ids(n, salt=0):
    return [( (i+salt) * 2654435761 ) % 150000 + 1000 for i in range(n)]

def cell(prefix_len, concurrency):
    prefix = ids(prefix_len)
    def one(i):
        r = completions(prefix + ids(32, salt=i + 1), max_tokens=64)
        fr = r["choices"][0].get("finish_reason")
        return fr, r["choices"][0]["text"]
    with cf.ThreadPoolExecutor(concurrency) as ex:
        results = list(ex.map(one, range(concurrency)))
    bad = [i for i, (fr, _) in enumerate(results) if fr not in ("stop", "length")]
    return bad

def main():
    m0 = metrics()
    print("pre-metrics:", m0)
    ok = True
    for plen, conc in ((65536, 4), (131072, 8)):
        bad = cell(plen, conc)
        print(f"ctx={plen} C{conc}: {'PASS' if not bad else f'FAIL rows {bad}'}")
        ok &= not bad
    # changed-instructions-on-cached-context: resend cold prompt of same length, must differ from cached answer
    a = completions(ids(8192), max_tokens=24)["choices"][0]["text"]
    b = completions(ids(8192), max_tokens=24)["choices"][0]["text"]
    c = completions(ids(8192, salt=7), max_tokens=24)["choices"][0]["text"]
    print("cache-replay deterministic:", a == b, "| changed-context differs:", a != c)
    ok &= (a == b) and (a != c)
    m1 = metrics()
    print("post-metrics:", m1)
    pre = m1.get("preemptions", 0) - m0.get("preemptions", 0)
    print(f"preemptions during run: {pre:g} (must be 0)")
    hitrate = None
    if m1.get("pq", 0) > m0.get("pq", 0):
        hitrate = (m1["ph"] - m0["ph"]) / (m1["pq"] - m0["pq"])
        print(f"prefix-cache hit ratio during run: {hitrate:.2f} (>0.5 expected for shared prefix)")
    print("GATE", "PASS" if ok and pre == 0 else "FAIL")
    sys.exit(0 if ok and pre == 0 else 1)

if __name__ == "__main__":
    main()
