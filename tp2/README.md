# TP2 recipe — Qwen3.8-Flash-Next on two DGX Sparks

Serves the same checkpoint from **maxwell (rank 0) + ampere (rank 1)** at
`http://maxwell:8000/v1`. Same context (262,144), same MTP3, FP8 KV at **16 GiB per rank →
1,918,359 tokens** (effective block 2,848), with ≥22 GiB MemAvailable left on every host.

## Default serving config (2026-09-26): [`overlay/`](overlay/) — Path B

`overlay/` mirrors the campaign TP2 profile (`qwen38-qsa-selection-tp2-20260917-{0,1}`,
maxwell rank0 + ampere rank1, image ID `953b00ee`, KV 16 GiB/rank, MTP3). Note the profile
ships `VLLM_QWEN3_8_HC_PREFILL_MODE=off` at TP2 (token-row sharding targets ≥3 ranks) with
coalescing on. Validation + rollback tooling: [`../tools/tp2-campaign-validate.sh`](../tools/tp2-campaign-validate.sh)
(manifest + RoCEnante marker proof, smoke, matched bench, TP4-overlay restore).
Measured cells: `tp2-overlay-*` rows in [`../data/results.jsonl`](../data/results.jsonl).
Stock files in this directory remain the fallback, same switch semantics as TP4.

## Files

`serve.sh`, `compose.yml`, `maxwell.env`, `ampere.env`, `config.json`, `profile.json` — same
contract as [`../tp4/`](../tp4/README.md) with exactly these differences:

| item | tp4 | tp2 | why |
|---|---|---|---|
| hosts | 4 | maxwell + ampere | faraday/hertz stay free for other workloads |
| `TENSOR_PARALLEL_SIZE` / `NNODES` | 4 / 4 | 2 / 2 | topology |
| `KV_CACHE_MEMORY_BYTES` | 45,097,156,608 | 17,179,869,184 | weights need 2× the per-rank share; smaller KV pool |
| HC sharding | `HC_TP=1` | **off/replicated** | HC workspace sharding is admitted only at TP4; running it at TP2 fails startup |

Everything else (RoCE caps, capture sizes, batched tokens 8,192, MTP3, block 16, cache
namespaces, start order worker→coordinator) is identical to tp4.

The env files here ship pre-configured for the tp2 overrides. Stage all files on **maxwell
and ampere only** at `/home/jasonc/spark_vllm/deployments/qwen38-karmic-tp2-DATE/`, create
cache dirs `~/.cache/qwen38-karmic-tp2-DATE-{vllm,triton,flashinfer,b12x}`, then bring up
ampere first, then maxwell:

```bash
D=qwen38-karmic-tp2-DATE
ssh ampere  "cd /home/jasonc/spark_vllm/deployments/$D && docker compose --env-file ampere.env -p $D up -d"
ssh maxwell "cd /home/jasonc/spark_vllm/deployments/$D && docker compose --env-file maxwell.env -p $D up -d"
```

## Measured (campaign overlay image `953b00ee…`, FP8 KV 16 GiB/rank, 3 repeats)

| workload | value | results.jsonl ids |
|---|---:|---|
| prefill 8K | 3,349.9 tok/s | `tp2-qsa-selection-prefill-8k` |
| prefill 64K | 3,058.2 tok/s | `tp2-qsa-selection-prefill-64k` |
| prefill 128K | 2,828.7 tok/s | `tp2-qsa-selection-prefill-128k` |
| decode C1 cold | 58.2 tok/s | `tp2-qsa-selection-decode-ctx0-c1` |
| decode C8 cold | 231.9 tok/s aggregate | `tp2-qsa-selection-decode-ctx0-c8` |
| decode C1 @8K | 55.8 tok/s | `tp2-qsa-selection-decode-ctx8k-c1` |
| decode C8 @8K | 226.6 tok/s aggregate | `tp2-qsa-selection-decode-ctx8k-c8` |

26/26 exact shared-prefix and history-edit checks; min MemAvailable 22.27 GiB (maxwell) /
25.25 GiB (ampere). Stock-base re-validation 2026-09-25 (image `64d4c3e0`, this dir's files verbatim, maxwell+ampere):
smoke 4/4; probe cells `tp2-stock-karmic-*` in `../data/results.jsonl` — cold 8K prefill
3,269.6 ± 12.2 tok/s (0.90× TP4-stock under this probe), decode C1 47.7, C8 154.0.

## When to use tp2

- 4 Sparks are TP4-or-nothing for this recipe; TP2 exists to free faraday+hertz for other
  models. Prefill is ~76–83% of tp4 rate and decode ~70% of tp4 C1 — the second NIC pair is
  not the bottleneck; cross-rank collectives per layer are.
- Historical note: an older **no-PLE** BF16-KV tp2 profile (51 GiB KV, 2048 batched tokens,
  batch-capture [4..64]) is also qualified (3.55M-token pool, 16/16 concurrent 13K stress, no
  OOM; larger KV pool measured 10.8% faster wall-clock, `tp2-no-ple-kv50-vs-kv36-stress`).
  The FP8 with-PLE recipe above supersedes it; the no-PLE ablation view removes 266 tensors /
  28.9 GB (`create-no-ple-view.py` pattern, kept in the fleet tree).
