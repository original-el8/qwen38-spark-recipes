# TP4 recipe — Qwen3.8-Flash-Next on four DGX Sparks

Serves `Qwen3.8-Flash-Next` (alias `qwen38-flash-next-nvfp4`) at
`http://<coordinator>:8000/v1`: 262,144-token context, 16 concurrent sequences, FP8-E4M3
KV pool of **42 GiB per rank → 5,486,463 tokens** (effective hybrid block 1,424).
Example fleet: coordinator `maxwell` + workers `ampere`, `faraday`, `hertz` — substitute
your own hosts; only rank order matters.

## What you get (default profile)

Stock `eugr/spark-vllm-b12x:nightly-20260925` + recipe defaults (KV block-size 32; b12x
backends inside `--speculative-config`, applied by `serve.sh`). No overlay, no local build.

| | default profile | decode-max profile (optional) |
|---|---|---|
| prefill cold 8K / 16K / 128K | ~4,4 / **4,704** / 3,825 tok/s | 3,3–3,4xx (−23%) |
| decode C1 cold | 57.3–60.9 tok/s (steps/s 31.0–31.7) | **64.8 tok/s** ctx0 (steps/s 37.0, +18%) |
| vs author r37 reference | prefill +24…+39%; realistic-prompt decode parity | prefill ≈author; decode lead |
| qualification | shipped default | EXPERIMENTAL (mainline b12x lacks eugr's 17 fork-ahead GB10/Spark commits) |

Row IDs + full matrices: [`../data/results.jsonl`](../data/results.jsonl)
(`tp4-ldb-*`, `tp4-ldb-block32-*`, `tp4-ldb-promoted-*`); scoreboard:
[`../SPARKRING.md`](../SPARKRING.md). The retired campaign image's matched numbers
(4,894 / 4,712 / 4,556 / 4,304 cold prefill ≈ parity with this default; 336.2 C8 decode)
stay in `results.jsonl` as `tp4-overlay-*` / `tp4-qsa-selection-*` historical rows — see
[`../COMPARISON.md`](../COMPARISON.md) for what that comparison taught us.

```bash
# one service per host; workers first, coordinator last; stop in reverse:
cd $DEPLOY_DIR   # staged copy of these files on EVERY host (pitfall 8)
docker compose --env-file <host>.env -p <project> up -d
docker compose --env-file <host>.env -p <project> down   # name-safe stop
# optional decode-max profile (build its image first — header of the overlay):
docker compose --env-file <host>.env -p <project> -f overlay.decode-max.yml up -d
```

- `overlay.decode-max.yml` = image + env delta over the same base/weights: a
  sha256-pinned **public** b12x wheel + engine-flag envs; build recipe in its header.
- `serve.sh` is ONE env-gated script for all variants: `MTP_BACKENDS_IN_SPEC=0` opts out
  of b12x spec backends; `ASYNC_SCHED=0`, `ATTENTION_BACKEND`,
  `RECURRENT_CHECKPOINT_POLICY`, `PROFILER_CONFIG` unset ⇒ default behavior.
- Switching variants on a live fleet: full `down` + port-release wait + `up` — NEVER a
  targeted `stop <service>` (compose project-name collision → EADDRINUSE that looks like a
  perf regression; gated example: [`../tools/cutover-campaign.sh`](../tools/cutover-campaign.sh)).

## Files

| file | role |
|---|---|
| `serve.sh` | ONE env-gated launcher for both variants (rank-aware; `NODE_RANK` 0 = coordinator, 1–3 headless) |
| `compose.yml` | one service, parameterized by `<host>.env`; run once per host, per variant |
| `overlay.decode-max.yml` | the ONE optional overlay (~20 lines); default needs no overlay |
| `maxwell.env` `ampere.env` `faraday.env` `hertz.env` | rank/fabric identity + KV budget + paths (example identities — rename freely) |
| `config.json` | checkpoint config override mounted at `/model/config.json` (`index_share_for_mtp_iteration=false`) |
| `recipe.json` | machine-readable contract for the stock base variant |
| `promotion-evidence.md` `qsa-selection-evidence.md` | raw campaign reports behind the historical rows |

## Bring-up

Prereqs (all four hosts): the base image present with **identical image ID**
(`docker image inspect -f '{{.Id}}'`, not tags); weights at `$MODEL_DIR` (root README step 2);
`/dev/infiniband` devices; per-profile cache dirs:

```bash
mkdir -p $CACHE_ROOT-{vllm,triton,flashinfer,b12x}   # names from <host>.env; fresh per profile
```

```bash
for h in $WORKERS; do
  ssh $h "cd $DEPLOY_DIR && docker compose --env-file $h.env -p $PROJECT up -d"
done
ssh $COORDINATOR "cd $DEPLOY_DIR && docker compose --env-file $COORDINATOR.env -p $PROJECT up -d"
```

Readiness = live API, not log grepping (`Application startup complete` false-passes on
reused containers): `until curl -sf http://<coordinator>:8000/v1/models; do sleep 15; done`
(`VLLM_ENGINE_READY_TIMEOUT_S=3600`; cold-cache start compiles/tritons ~10+ min, warm-cache
restart minutes). Then `tools/smoke.sh` and `tools/bench-quick.py` before trusting anything.

## Memory contract

| quantity | value |
|---|---|
| KV budget / rank (`KV_CACHE_MEMORY_BYTES`) | 45,097,156,608 B (42 GiB) |
| GPU utilization | 0.80 (of GB10 unified memory as CUDA device) |
| KV pool exposed | 5,486,463 tokens, effective block 1,424 |
| MemAvailable floor under load | ≥16 GiB on the limiting rank (measured min 16.05 GiB) — violating it OOM-kills ranks (exit 137) during long prefill |
| KV pool vs context limit | pool is aggregate capacity; per-request limit stays 262,144 |

## KV dtype decision (FP8 default — deliberate, measured)

Head-to-head, same hardware/campaign ([`results.jsonl`](../data/results.jsonl)):

| cell | BF16 28 GiB | FP8 42 GiB | FP8 vs BF16 |
|---|---:|---:|---|
| prefill 8K | 4,422.6 | 4,421.5 | −0.03% |
| prefill 64K | 4,190.5 | 4,276.9 | **+2.1%** |
| prefill 128K | 3,684.8 | 3,834.5 | **+4.1%** |
| decode C1 cold | 85.6 | 81.9 | −4.3% |
| decode C8 cold | 343.6 | 336.2 | −2.2% |
| KV capacity/rank | 2,024,110 tok | 5,486,463 tok | **2.71×** |

Honest asterisks: (a) the promotion-day FP8 arm measured far slower (rows `tp4-fp8-kv42-*`)
with unexplained variance never attributed — only the steady-state arm above is
representative; (b) FP8 KV carries the open quality caveat (1 repeated-word loop + client
timeout in 20 long-reasoning cases; MTP-off: no loop in 8 requests, unattributable — root
README honesty box).

**FP8 stays the published default** (parity-to-faster at long context, 2.71× concurrency
headroom) with the risk disclosed; for long-reasoning-heavy or single-stream-latency-critical
traffic use the BF16 variant: `KV_CACHE_MEMORY_BYTES=30064771072`, `LOAD_FORMAT=instanttensor`,
drop `--kv-cache-dtype`, everything else unchanged → 2,024,110-token pool at the BF16 rates above.

## Verification gates passed

Stock base image (current default), `tools/smoke.sh` + `tools/long-prefix-check.py`, cold
prefixes: 4/4 behavior gates; 64K@C4 and 128K@C8 shared-prefix + edited-tail PASS;
changed-instructions-on-cached-context PASS; zero preemptions (exact `_total` counters);
prefix-cache hit ratio 0.82–0.99; all ranks `restarts=0 oom=false`; MemAvailable 17 GiB.
Campaign-era deep validation (overlay stack): 33/33 behavior cases (arithmetic, tool calls,
prefix reuse, 8/16-concurrent JSON retrieval at 8K/64K/128K, vision); 26/26 long shared-prefix
checks — `qsa-selection-evidence.md`, `promotion-evidence.md`.

## Pitfalls (each cost real time)

1. **Never share cache namespaces between profiles/dtypes.** FP8 vs BF16 KV select different
   tuned kernels; a stale namespace silently changes kernels and invalidates A/B. Create the
   four cache dirs fresh per profile.
2. `NCCL_IB_MERGE_NICS=0` on dual-HCA cabling; merging the two HCA functions breaks QP setup.
3. `HC_TP` sharding is **TP4-only** — TP2 must run replicated/off (see [`../tp2/`](../tp2/README.md)).
4. The image may bake GLM-era env defaults; `serve.sh` unsets them. Keep those unsets.
5. Big first-prefill variance on a fresh coordinator is a known flake (one-off 2× MoE-kernel
   stall seen once, cause unresolved): re-measure any suspiciously slow arm.
6. Decode rates depend on MTP acceptance (cold C1 ranges 57–82 tok/s across regimes with the
   SAME config): read `steps/s` and acceptance, not tok/s alone.
7. Stage deployment files to **every** rank's own disk, including the coordinator's —
   `docker compose` runs locally per host; peers being staged is not enough (`tools/cutover-tp4.sh`
   aborts fleet-wide if any rank lacks the files; this exact miss once failed a real cutover).

See `DETAILS.md` for every flag and env var with its why.
