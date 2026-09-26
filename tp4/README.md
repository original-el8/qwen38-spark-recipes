# TP4 recipe — Qwen3.8-Flash-Next on four DGX Sparks

Serves `Qwen3.8-Flash-Next` (and alias `qwen38-flash-next-nvfp4`) at
`http://maxwell:8000/v1` with 262,144-token context, 16 concurrent sequences, and an
FP8-E4M3 KV pool of **42 GiB per rank → 5,486,463 tokens** (effective hybrid block 1,424
tokens).

## Default serving config (2026-09-26): [`overlay/`](overlay/) — Path B

`overlay/` mirrors the live fleet deployment exactly (`qwen38-qsa-selection-20260917-*`,
image `spark-vllm:qwen38-qsa-selection-76061de4-b12xd2d5368d-sm121-image-r1`, ID `953b00ee`;
built from [`../build/README.md`](../build/README.md)). Promoted as serving default after the
matched A/B (`../COMPARISON.md`): cold prefill 4,894±10 (8K) / 4,712 (16K) / 4,556 (32K) /
4,304 tok/s (64K), +34% over the stock base; HC prefill sharding on (`HC_PREFILL_MODE=shard`),
coalesce + QSA fusion, KV 42 GiB/rank. Bring-up/stop order identical to below, using
`--env-file <host>.env -p qwen38-qsa-selection-<date>`; promotion with gates:
[`../tools/cutover-campaign.sh`](../tools/cutover-campaign.sh).

The files in **this directory** remain the **stock fallback** (goal's base-image recipe,
`eugr/spark-vllm-b12x` nightly, rebuilt from Docker Hub with no fleet-local bits; faster
primed-context decode). One `compose down` + `compose up` per host switches between them;
NEVER without port-release gating (see Pitfalls — the 2026-09-26 EADDRINUSE false-regression).

## Files

| file | role |
|---|---|
| `serve.sh` | in-container launch (rank-aware; `NODE_RANK` 0 = coordinator, 1–3 headless) |
| `compose.yml` | one service, parameterized by `<host>.env`; run once per host |
| `maxwell.env` `ampere.env` `faraday.env` `hertz.env` | rank/fabric identity + per-profile caps |
| `config.json` | checkpoint config override mounted at `/model/config.json` (`index_share_for_mtp_iteration=false`) |
| `profile.json` | machine-readable manifest of the whole runtime contract |
| `promotion-evidence.md` `qsa-selection-evidence.md` | raw campaign reports the numbers came from |

## Bring-up

Prereqs (all four hosts): image present with matching image ID; weights at
`/home/jasonc/models/Qwen3.8-Flash-Next-NVFP4`; `/dev/infiniband` devices; these dirs:

```bash
mkdir -p ~/.cache/qwen38-karmic-tp4-<date>-{vllm,triton,flashinfer,b12x}
```

Start **workers, then coordinator**; stop **coordinator, then workers**:

```bash
D=qwen38-karmic-tp4-<date>
for h in ampere faraday hertz; do
  ssh $h "cd /home/jasonc/spark_vllm/deployments/$D && docker compose --env-file $h.env -p $D up -d"
done
ssh maxwell "cd /home/jasonc/spark_vllm/deployments/$D && docker compose --env-file maxwell.env -p $D up -d"
```

Readiness = coordinator log line `Application startup complete`
(`VLLM_ENGINE_READY_TIMEOUT_S=3600`; a cold-cache start compiles/tritons for ~10+ min;
warm-cache restart is minutes). Then run `tools/smoke.sh` and `tools/bench-quick.py`.

## Memory contract

| quantity | value |
|---|---|
| KV budget / rank (`KV_CACHE_MEMORY_BYTES`) | 45,097,156,608 B (42 GiB) |
| GPU utilization | 0.80 (of GB10 unified memory as CUDA device) |
| KV pool exposed | 5,486,463 tokens, effective block 1,424 |
| MemAvailable floor under load | ≥16 GiB on the limiting rank (measured min 16.05 GiB, maxwell) |
| KV pool vs context limit | pool is aggregate capacity; per-request limit stays 262,144 |

## Measured (campaign overlay image, FP8 KV, healthy arm)

| workload | value | results.jsonl ids |
|---|---:|---|
| prefill 8K | 4,421.5 tok/s | `tp4-qsa-selection-prefill-8k` |
| prefill 64K | 4,276.9 tok/s | `tp4-qsa-selection-prefill-64k` |
| prefill 128K | 3,834.5 tok/s | `tp4-qsa-selection-prefill-128k` |
| decode C1 cold | 81.9 tok/s (MTP accept 53.5%) | `tp4-qsa-selection-decode-ctx0-c1` |
| decode C8 cold | 336.2 tok/s aggregate | `tp4-qsa-selection-decode-ctx0-c8` |
| decode C1 @8K | 82.2 tok/s | `tp4-qsa-selection-decode-ctx8k-c1` |
| decode C8 @8K | 327.9 tok/s aggregate | `tp4-qsa-selection-decode-ctx8k-c8` |

Stock-base (karmic-nightly) probe cells and the A/B verdict against this table are in
[`../COMPARISON.md`](../COMPARISON.md) (`tp4-stock-karmic-*` rows: 3,639 cold 8K prefill,
primed-context decode stronger than the overlay).

## KV dtype decision (FP8 vs BF16 — deliberate, not default)

Measured head-to-head, same campaign, same hardware (`data/results.jsonl`):

| cell | BF16 28 GiB | FP8 42 GiB (qsa-selection) | FP8 vs BF16 |
|---|---:|---:|---|
| prefill 8K | 4,422.6 | 4,421.5 | −0.03% |
| prefill 64K | 4,190.5 | 4,276.9 | **+2.1%** |
| prefill 128K | 3,684.8 | 3,834.5 | **+4.1%** |
| decode C1 cold | 85.6 | 81.9 | −4.3% |
| decode C8 cold | 343.6 | 336.2 | −2.2% |
| KV capacity/rank | 2,024,110 tok | 5,486,463 tok | **2.71×** |

Two honest asterisks: (a) the promotion-day FP8 arm measured far slower (3,319 / 3,856 / 2,713;
rows `tp4-fp8-kv42-*`) with unexplained variance never attributed — only the later
steady-state arm above is representative; (b) FP8 KV carries the open LAVD quality issue
(1 repeated-word loop + client timeout in 20 long-reasoning cases; MTP-off returned 7 exact +
1 near with no loop in 8 requests, unattributable).

**Decision: FP8 stays the published default** — parity-to-faster throughput at long context,
2.71× concurrency headroom — with the loop risk disclosed; switch to the BF16 variant
(pitfall 7) for long-reasoning-heavy or single-stream-latency-critical traffic.

## Verification gates passed (fleet, 2026-09-17/18 campaign + this re-validation)

Re-validation on the **stock base image 2026-09-25** (`tools/smoke.sh` +
`tools/long-prefix-check.py`, cold per-run prefixes): 4/4 behavior gates; 64K@C4 and 128K@C8
shared-prefix + edited-tail cells PASS; changed-instructions-on-cached-context PASS; zero
preemptions (exact `_total` counters); prefix-cache hit ratio 0.82–0.99 on shared prefixes;
all four ranks `running restarts=0 oom=false`; MemAvailable 17 GiB (floor ≥16).
Campaign-era deep validation on the overlay stack: 33/33 bounded behavior cases (arithmetic,
tool call, prefix reuse, 8-/16-concurrent JSON retrieval at ~8K/64K/128K, changed instructions
on cached context, vision); 26/26 long shared-prefix/history-edit checks. Full text:
`qsa-selection-evidence.md`, `promotion-evidence.md`.

## Pitfalls (each cost real time in the campaign)

1. **Never share cache namespaces between profiles/dtypes.** FP8 vs BF16 KV select different
   tuned kernels; a stale namespace silently changes kernels and invalidates A/B. The compose
   mounts four per-profile cache dirs; create them fresh for a new profile.
2. `NCCL_IB_MERGE_NICS=0` on this cabling; merging the two HCA functions breaks QP setup.
3. `HC_TP` sharding (or `HC_PREFILL_MODE=shard` on the campaign image) is **TP4-only** —
   TP2 must run replicated/off (see the tp2 recipe).
4. The image may bake GLM-era env defaults; `serve.sh` unsets them. Keep those unsets.
5. Big first-prefill variance on a fresh coordinator is a known flake (see root README honesty
   box): re-measure a suspicious slow arm before drawing conclusions.
6. FP8 KV quality limitation (LAVD loop) is documented in the root README — keep the BF16
   fallback profile if your workload is long-reasoning-heavy.
7. BF16-KV alternative: with `KV_CACHE_MEMORY_BYTES=30064771072` and no `--kv-cache-dtype`
   override the same files reproduce the faster 28 GiB BF16 baseline (4,422.6 / 4,190.5 /
   3,684.8 prefill; 85.6 C1) at 2,024,110-token capacity — set
   `LOAD_FORMAT=instanttensor`, drop `kv_cache_dtype`, keep everything else.
8. Stage the deployment directory (compose/serve/env/config) to **every** rank's own disk,
   including the coordinator's — `docker compose` runs locally per host; peers being staged is
   not enough. `tools/cutover-tp4.sh` now aborts before touching the fleet if any rank lacks
   the files (this exact miss failed a real cutover once).

See `DETAILS.md` for every flag and env var with its why.
