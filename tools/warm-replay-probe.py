#!/usr/bin/env python3
"""Method-matched warm-replay decode probe (the ONLY cell comparable to sparkring
`aggregate_decode_window_tps` / warm-wall realistic numbers, and to the campaign's
75.9/87.5± replay cells — LDB's dataset aggregate mixes scout/cold/warmup into one wall
number and must NEVER be quoted against those).

Method (prefix-cached replays, author-equivalent):
  1. Build ONE realistic multi-turn prose prompt of ~CTX tokens (seeded synthetic corpus —
     cache keys are token hashes, not semantics, so length/entropy is what must match).
  2. Replay the SAME request M+1 times: temp=0, fixed max_tokens (forced output so decode
     length is deterministic), C=1. Run 1 = warmup (fills prefix cache).
  3. Report per-run wall tok/s, STEADY = mean of runs 2..M+1 (warm regime), and mean
     cached/prompt token share (proof the prefix cache actually hit).
Usage: warm-replay-probe.py [endpoint] [model] [ctx_tokens] [replays] [max_tokens]
"""
import json
import statistics
import sys
import time
import urllib.request

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://maxwell:8000/v1").rsplit("/v1", 1)[0] + "/v1"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "Qwen3.8-Flash-Next"
CTX = int(sys.argv[3]) if len(sys.argv) > 3 else 8000
REPLAYS = int(sys.argv[4]) if len(sys.argv) > 4 else 5
MAXTOK = int(sys.argv[5]) if len(sys.argv) > 5 else 512

_SENT = ("The committee reviewed the quarterly throughput report and noted that the "
         "observed drift correlated with scheduling changes introduced earlier in the "
         "quarter. ")
def build_prompt(n_tokens: int) -> str:
    chars, words, i = 0, [], 0
    while chars < n_tokens * 6.17:  # ~6.17 chars/token calibration
        w = (_SENT * 3).split()
        extra = w[(i * 7) % len(w):] + [f"item{i}-{(i * 2654435761) % 9973}"] * 3
        words.extend(extra)
        chars += sum(len(t) + 1 for t in extra)
        i += 1
    return ("You are a careful analyst. Background:\n" + " ".join(words) +
            "\n\nQuestion: Summarize the key findings in exactly two paragraphs.")


def one(req_ids: str):
    body = {"model": MODEL, "messages": [{"role": "user", "content": req_ids}],
            "temperature": 0, "max_tokens": MAXTOK, "stream": True,
            "stream_options": {"include_usage": True}}
    t0 = time.perf_counter()
    out = 0
    cached = prompt_tok = 0
    with urllib.request.urlopen(urllib.request.Request(
            BASE + "/chat/completions", data=json.dumps(body).encode()), timeout=1800) as r:
        for line in r:
            if not line.startswith(b"data: "):
                continue
            chunk = line[6:].strip()
            if chunk == b"[DONE]":
                break
            d = json.loads(chunk)
            if d.get("usage"):
                out = d["usage"].get("completion_tokens", 0)
                prompt_tok = d["usage"].get("prompt_tokens", 0)
                cached = (d["usage"].get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    return {"wall_s": time.perf_counter() - t0, "out": out,
            "tps": out / (time.perf_counter() - t0), "cached": cached, "prompt": prompt_tok}


def main():
    prompt = build_prompt(CTX)
    runs = [one(prompt) for _ in range(REPLAYS + 1)]
    warm = runs[1:]
    print(json.dumps({
        "method": "warm-replay (run1 warmup; steady=mean of replays; temp0 forced output)",
        "ctx_tokens_target": CTX, "max_tokens": MAXTOK, "concurrency": 1,
        "run_tps": [round(r["tps"], 2) for r in runs],
        "steady_warm_wall_tps": round(statistics.mean(r["tps"] for r in warm), 2),
        "steady_stdev": round(statistics.stdev((r["tps"] for r in warm), statistics.mean(r["tps"] for r in warm)), 2) if len(warm) > 1 else None,
        "cold_run1_tps": round(runs[0]["tps"], 2),
        "avg_prompt_tokens": round(statistics.mean(r["prompt"] for r in warm)),
        "avg_cached_share": round(statistics.mean(r["cached"] / max(r["prompt"], 1) for r in warm), 3),
    }, indent=2))


if __name__ == "__main__":
    main()
