# Qwen3.8-Flash-Next on 4× NVIDIA DGX Spark

Reproducible vLLM serving recipes for **Qwen3.8-Flash-Next (NVFP4)** on a four-node
NVIDIA DGX Spark cluster with a dual-NIC RoCE fabric. Two topologies ship here:

| recipe | hosts | what it is |
|---|---|---|
| [`tp4/`](tp4/README.md) | maxwell + ampere + faraday + hertz | full fleet, 262,144-token context, FP8 KV, 5.49M-token KV pool — one `compose.yml`, `-f overlay.yml` switches stock base → campaign overlay (serving default) |
| [`tp2/`](tp2/README.md) | maxwell + ampere | two-Spark pair, FP8 KV, 1.92M-token KV pool, ≥22 GiB host headroom left — same compose + overlay-override pattern |

Every number below links to a row in [`data/results.jsonl`](data/results.jsonl)
(schema in [`data/README.md`](data/README.md)).

## Results

Live re-validation on the stock base (image ID `sha256:64d4c3e0…`) 2026-09-25: TP4 serving,
4/4 behavior smoke, all-rank health; matched probe cells in `data/results.jsonl`
(`tp4-stock-karmic-*`). Full campaign matrices (BF16/FP8, tp4/tp2, HC/MoE/MTP/RoCE ablations)
in [`data/results.jsonl`](data/results.jsonl) + [`data/campaign-matrix.json`](data/campaign-matrix.json).
Stock-vs-overlay A/B verdict: [`COMPARISON.md`](COMPARISON.md) — overlay wins cold prefill
(+34.2%) and cold decode (+15-16%); stock wins primed-context decode (+13.6/+25.7%).

## Hardware

Four NVIDIA DGX Spark systems (GB10, 121 GiB unified memory each, `sm_121a` /
CUDA capability 12.1, Ubuntu 24.04 arm64, driver 580.x), each with **two ConnectX NICs** on a
dedicated switch:

| host | rank (tp4) | fabric addr (NIC 1 / NIC 2) | household LAN |
|---|---|---|---|
| `maxwell` (leader, rank 0, endpoint host) | 0 | 10.200.0.12 / 10.200.1.12 | 192.168.1.213 |
| `ampere` | 1 | 10.200.0.14 / 10.200.1.14 | 192.168.1.215 |
| `faraday` | 2 | 10.200.0.11 / 10.200.1.11 | 192.168.1.212 |
| `hertz` | 3 | 10.200.0.13 / 10.200.1.13 | 192.168.1.214 |

Fabric facts that shape the recipes (measured, see `tp4/DETAILS.md`):

- RoCE v2 **GID index 3**; HCAs `rocep1s0f0,roceP2p1s0f0`; both NIC functions are in one
  `10.200.0.0/23` addressing scope → `NCCL_IB_MERGE_NICS=0` is **required** (the two cabled
  functions need separate QP setup).
- Cross-node **RoCEnante** (b12x one-shot RoCE all-reduce/all-gather) replaces NCCL inside the
  decode step under size caps (2 MiB all-reduce / 16 MiB all-gather; NCCL above). Standalone it
  is ~3× faster than NCCL at decode-relevant sizes (measured in b12x PR #295 evidence).
- TP4 cross-node is only sane because of RoCEnante; PCIe all-reduce is single-node only.

## Model

`/home/jasonc/models/Qwen3.8-Flash-Next-NVFP4` (99 GB, 36 safetensors shards), identical bytes
on all four hosts:

- source: `Qwen/Qwen3.8-Flash-Next` @ revision `de4b8e4d43b917e7706784d8bb445c9af86a3540`;
- export template: `local-inference-lab/Qwen3.8-Flash-Next-NVFP4` @ `ada4da32a583a78aa47299f45a70603c950490b8`;
- mixed quantization served with `--quantization modelopt_mixed`: NVFP4 routed experts +
  PLE embedding, MXFP8 attention projections (`hf_quant_config.json` groups);
- this fleet's copy adds quatrain **input-scale calibration**
  (`export-manifest.json`, sha256 `8a0b93599e3edb4ab25357e8af16cf1ac2c6a61354fcec9d6aa50ee9fbf94397`,
  kind `quatrain_qwen38_input_scales_with_unobserved_fallback`);
- architecture `Qwen3_8FlashNextForConditionalGeneration` — on the campaign branch this is
  `vllm/models/qwen3_8_flash_next/`, on `dev/karmic-kraken` it maps to
  `vllm/models/qwen4_exp/` (same checkpoint; the registry entry is an alias). It is a hybrid:
  gated-deltanet (GDN) recurrent layers + attention layers + PLE/Engram + MTP drafter head.

The deployments mount one **config override** over the checkpoint
([`tp4/config.json`](tp4/config.json), sha256
`33d4220bd93677d0f566a93f53ee91e86970240d9c904a39faeb881946f08079`): it differs from the
in-checkpoint `config.json` only by `"index_share_for_mtp_iteration": false`.

## Images

Serving default: the **campaign overlay** (built from pinned sources, reproducible via
`build/`). Stock fallback base: **`eugr/spark-vllm-b12x`** nightlies from Docker Hub
(vLLM `dev/karmic-kraken` + b12x `master`) — the original goal's mandated base, still
published verbatim. Full provenance, digests, and the rebuild recipe:
[`build/README.md`](build/README.md).

## Quick start (tp4, either image variant)

```bash
# 1) pick base or overlay, on every host — base is a pull, overlay is a build (build/README.md)
docker pull eugr/spark-vllm-b12x:nightly-20260925

# 2) stage tp4/{compose.yml,overlay.yml,serve.sh,config.json,<host>.env} into $DEPLOY_DIR
#    on each host; create cache dirs $(dirname): compose binds $CACHE_ROOT-{vllm,triton,...}

# 3) workers first (ampere, faraday, hertz), then coordinator (maxwell) — drop `-f overlay.yml`
#    for the stock base:
ssh ampere  'cd $DEPLOY_DIR && docker compose --env-file ampere.env  -p myproj -f compose.yml -f overlay.yml up -d'
ssh faraday 'cd $DEPLOY_DIR && docker compose --env-file faraday.env -p myproj -f compose.yml -f overlay.yml up -d'
ssh hertz   'cd $DEPLOY_DIR && docker compose --env-file hertz.env   -p myproj -f compose.yml -f overlay.yml up -d'
ssh maxwell 'cd $DEPLOY_DIR && docker compose --env-file maxwell.env -p myproj -f compose.yml -f overlay.yml up -d'

# 4) readiness = live API (log grepping false-passes on reused containers):
until curl -sf http://maxwell:8000/v1/models >/dev/null; do sleep 15; done   # cold compile ~10 min
./tools/smoke.sh
```

Stop order is the reverse (coordinator first). Startup/stop order is not optional: rank-0
serves the API and holds the rendezvous; workers headlessly join `MASTER_ADDR:29507`.
Switching image variants: `down` on all hosts, wait for :8000 release, `up` — see
`tools/cutover-campaign.sh` for the gated version.

## Fleet state (2026-09-26)

**Serving: Path B campaign overlay** (`spark-vllm:qwen38-qsa-selection-76061de4-b12xd2d5368d-sm121-image-r1`,
ID `953b00ee`) as `qwen38-qsa-selection-20260917-{0..3}`, API `maxwell:8000`. Promoted after
matched A/B showed +34% cold prefill; confirmed live with image-manifest + RoCEnante markers
and bench (cold prefill 4,894±10 @8K, 4,712 @16K, 4,556 @32K, 4,304 @64K; cold per-run).
Rollback to stock = same files WITHOUT `-f overlay.yml` (project `qwen38-karmic-tp4-20260925`
deployment dirs still on-host as the legacy stock stack). Promotion tool: `tools/cutover-campaign.sh`
(port-release gate + capture-wait + performance gate + auto-rollback). Cutover lesson recorded:
`compose stop <service>` with a guessed service name silently no-ops — the 2026-09-26 false
"overlay regression" was stock still bound to :8000 (EADDRINUSE on the overlay coordinator);
always `compose down` and gate on port release.

## Repo map

| path | what |
|---|---|
| [`tp4/`](tp4/README.md) | four-Spark recipe: README, DETAILS (every env var + flag), runnable files |
| [`tp2/`](tp2/README.md) | two-Spark recipe (maxwell+ampere) |
| [`build/`](build/README.md) | base-image provenance (Docker Hub), overlay lineage, rebuild recipe |
| [`tools/`](tools/) | `smoke.sh` (behavior gates), `bench-quick.py` (matched probe), `cutover-tp4.sh` (fleet stop/start/rollback orchestration) |
| [`data/`](data/README.md) | every quoted number as [`results.jsonl`](data/results.jsonl); per-variant ablation records in [`campaign-matrix.json`](data/campaign-matrix.json) |
| [`SPARKRING.md`](SPARKRING.md) | applicability of Fujitsu sparkring profile levers to this fleet (verified envs, test order, retained negatives) |
| `COMPARISON.md` | stock karmic-nightly vs campaign overlay image, matched probe cells (`tools/bench-quick.py` run on both stacks) |
| `tools/campaign-ab-probe.sh` | one-command A/B: campaign profile up → API-gate → probe → stock restore |
| `llms.txt` | agent entry point |

## Reading the numbers

- Prefill = one cold random-token request (8K/64K/128K), flushed prefix cache, server-reported
  duration. Decode = forced 512-token outputs at temperature 0, cold caches; C8 is aggregate
  over the all-decoding window. ≥3 warm repeats per cell; all ranks re-inspected after every
  matrix. Full method: [`data/README.md`](data/README.md).
- MTP acceptance moves decode rates independently of kernels: decode rows that changed
  acceptance also record `mtp_acceptance` — read both before crediting a change.
- Memory floors: budgets target **≥16 GiB Linux MemAvailable** on the limiting rank
  (measured under load, not `MemFree`). A KV pool that violates this floor OOM-kills ranks
  with exit 137 under long prefill.

## Known quality limits (honesty box)

- FP8 KV + MTP3 showed one repeated-word loop ending in client timeout in a 20-case LAVD run
  (16 exact / 3 near / 1 loop); FP8 + MTP-off returned 7 exact + 1 near with no loop in 8
  requests, but tuning and scheduling differences prevent attribution. FP8 KV was promoted
  **by explicit operator choice** after that disclosure: parity-to-faster long-context
  prefill and 2.71× KV capacity, at a ~4% single-stream decode cost and the unresolved loop.
  BF16-KV cells remain available in [`data/results.jsonl`](data/results.jsonl).
- T0 same-prompt replay behavior is **not reliably characterized by this repo's tests**: runs
  disagreed (identical when unseeded in one run, divergent in others; seeded results equally
  inconsistent; the isolated manual check produced trivial empty-text outputs). Treat bitwise
  replay reproducibility as UNVERIFIED in either direction. `tools/long-prefix-check.py`
  reports replay agreement but deliberately does not gate on it.
- A one-off TP4 startup slowdown (2× MoE-kernel latency on the coordinator only, same graph,
  same clocks) was seen once and cleared on restart without any change; cause unresolved.
  Recipe advice: re-run before trusting a slow first arm — the campaign did.
