# TP2 recipe — Qwen3.8-Flash-Next on two DGX Sparks

Same checkpoint, same serving contract as [`../tp4/`](../tp4/README.md), on two Sparks:
rank 0 (`maxwell` example) + rank 1 (`ampere` example). 262,144-token context, MTP3, FP8
KV at **16 GiB per rank → 1,918,359 tokens** (effective block 2,848), ≥22 GiB MemAvailable
left on every host under load.

## Results

Current default (stock `eugr/spark-vllm-b12x:nightly-20260925`, this dir's files verbatim):
cold 8K prefill **3,269.6 ± 12.2 tok/s** (0.90× the TP4-stock rate under the same probe),
decode C1 47.7 / C8 154.0; smoke 4/4 — rows `tp2-stock-karmic-*` in
[`../data/results.jsonl`](../data/results.jsonl).

Historical campaign-image reference (`953b00ee…`, FP8 KV 16 GiB/rank, 3 repeats) —
retained as `tp2-qsa-selection-*` rows, NOT current-config numbers:

| workload | value |
|---|---:|
| prefill 8K / 64K / 128K | 3,349.9 / 3,058.2 / 2,828.7 tok/s |
| decode C1 cold / @8K | 58.2 / 55.8 tok/s |
| decode C8 cold / @8K | 231.9 / 226.6 tok/s aggregate |

Engine-flag A/B on TP2 (LM-head/MTP flags; HC sharding is TP4-only so it never applied):
**neutral in every cell** (prefill 3,391→3,340; cold C1 54.3→62.7 acceptance-noise; primed
cells flat within CI) — rows `tp2-overlay-flags7-*`. Nothing on TP2 justifies an overlay.

## When to use TP2

Four Sparks run this recipe as TP4 or not at all; TP2 exists to free the other two Sparks
for other workloads. Prefill is ~76–83% of the TP4 rate, decode ~70% of TP4 C1 — the NIC
pair is not the bottleneck; cross-rank collectives per layer are. (An older no-PLE BF16-KV
TP2 profile — 3.55M-token pool, 16/16 concurrent 13K stress, no OOM — is superseded by this
one; ablation view kept in `create-no-ple-view.py`.)

## Files and diffs from TP4

`serve.sh`, `compose.yml`, `maxwell.env`, `ampere.env`, `config.json` — identical contract
except:

| item | tp4 | tp2 | why |
|---|---|---|---|
| hosts | 4 | 2 | topology |
| `TENSOR_PARALLEL_SIZE` / `NNODES` | 4 / 4 | 2 / 2 | topology |
| `KV_CACHE_MEMORY_BYTES` | 45,097,156,608 | 17,179,869,184 | weights need 2× the per-rank share |
| HC sharding | `HC_TP=1` | **off/replicated** | HC workspace sharding is admitted only at TP4; enabling it at TP2 fails startup |

Everything else (RoCE caps, capture sizes, 8,192 batched tokens, MTP3, cache namespaces,
worker→coordinator order, same image ID everywhere) is identical to tp4.

## Bring-up

Stage all files on **both** hosts locally (`$DEPLOY_DIR`), create fresh cache dirs
`$CACHE_ROOT-{vllm,triton,flashinfer,b12x}`, worker first, coordinator second:

```bash
ssh $WORKER      "cd $DEPLOY_DIR && docker compose --env-file $WORKER.env -p $PROJECT up -d"
ssh $COORDINATOR "cd $DEPLOY_DIR && docker compose --env-file $COORDINATOR.env -p $PROJECT up -d"
```

Readiness = live `GET /v1/models`, then `../tools/smoke.sh`. Validation/rollback example:
[`../tools/tp2-campaign-validate.sh`](../tools/tp2-campaign-validate.sh).

Behavior record (campaign stack): 26/26 exact shared-prefix and history-edit checks; min
MemAvailable 22.27 / 25.25 GiB. See [`../tp4/README.md`](../tp4/README.md) pitfalls — all
apply to TP2 except HC-sharding-specific ones.
