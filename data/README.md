# Machine-readable results schema

Every number quoted in a `README.md` or `DETAILS.md` in this repository is one line in
[`results.jsonl`](results.jsonl). Fields:

| field | meaning |
|---|---|
| `id` | stable identifier: `<topology>-<config>-<workload>-<shape>` |
| `topology` | `tp2` (2 Sparks) or `tp4` (4 Sparks) |
| `image_id` | full Docker image ID (`sha256:...`) that produced the number |
| `kv_cache_dtype` | `bfloat16` or `fp8_e4m3` |
| `kv_cache_gib_per_rank` | KV pool budget per GPU, GiB |
| `workload` | `prefill` \| `decode` \| `ab-comparison` \| `component-qsa-*` \| `concurrent-stress-*` |
| `context` | nominal input length in tokens (0 for cold short decode) |
| `concurrency` | simultaneous requests (`C1`, `C8`); aggregate rate for C>1 |
| `tokens_per_s_mean` / `tokens_per_s_stdev` | rate and sample standard deviation |
| `control_tokens_per_s_mean` | matched-arm value for `ab-comparison` rows |
| `change_pct` | recorded verdict percentage for `ab-comparison` rows |
| `repeats` | measured repeats after warmup (always ≥3 unless noted) |
| `mtp_acceptance` | mean draft acceptance fraction, when recorded |
| `source` | file inside the fleet campaign tree or this repo holding the raw samples |
| `published_as` | upstream PR/commit that carries the same evidence, when published |

## Measurement method (catID-style, `llm-inference-bench` lineage)

- **Prefill**: one cold request of random token IDs (nominal 8,192 / 65,536 / 131,074 tokens
  incl. 2 template tokens), prefix cache flushed, zero cached input tokens; timing is the
  server-reported prefill duration.
- **Decode**: fixed 512 forced output tokens, temperature 0, cold prefix caches. C8 rates are
  aggregate tokens over the interval where all requests are decoding.
- Every cell has ≥3 warm repeats; all rank containers were re-inspected after each matrix
  (running state, expected image ID, zero restarts, no OOM, zero preemptions).

## Provenance

The raw sample files, metric snapshots, and rank-health captures for these rows live in the
fleet under `/home/jasonc/spark_vllm/benchmark-results/qwen38-*` (operator host, not
published). The `source` field names the summary document each row was transcribed from.
