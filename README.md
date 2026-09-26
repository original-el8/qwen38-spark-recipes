# Qwen3.8-Flash-Next on 4× NVIDIA DGX Spark — results and reproduction

What this repo is: the exact serving configuration behind the numbers below, and how to
reproduce them on any four-node (or two-node) DGX Spark fleet with a dual-NIC RoCE fabric.
Nothing here requires fleet-local images or unpublished sources — the default config builds
entirely from public artifacts. Every quoted number links to a row in
[`data/results.jsonl`](data/results.jsonl) (schema + method in [`data/README.md`](data/README.md)).

Model: **Qwen3.8-Flash-Next NVFP4 (QAD revision)**, 262,144-token context, FP8-E4M3 KV,
MTP3 speculative decoding, hybrid GDN+attention+PLE architecture, V2 model runner.

## Results — default profile

Base: `eugr/spark-vllm-b12x:nightly-20260925` (public Docker Hub) + recipe defaults
(KV block-size 32, b12x backends inside the speculative-config). No overlays, no local build.

| cell | value | method / rows |
|---|---:|---|
| prefill 8K / 16K / 32K / 64K / 128K (tok/s, cold) | **3,558 / 3,575 / 3,435 / 3,233 / 2,841** steady state | llm_decode_bench fast matrix ×2 full repeats, near-identical: `tp4-ldb-promoted-prefill-*`. An intermittent HIGH tuning cluster (4,4–4,7xx; `tp4-ldb-block32-*`) does NOT reproduce on fresh-container state — attribution unresolved, see honesty box |
| decode C1 cold (tok/s, ctx0 / 64K) | **63.3–64.3** | same runs, `tp4-ldb-promoted-decode-*-c1`; MTP-normalized steps/s 36.1, accept 1.75–1.83 |
| decode C8 (tok/s, ctx0 / 64K) | **249.2 / 243.3** | `tp4-ldb-promoted-decode-*-c8`; campaign image's 336.2 was a higher-accept historical cell (`tp4-qsa-selection-decode-ctx0-c8`) |
| decode C1 realistic prompts (tok/s, ctx0 / 64K) | 58.3 / 57.9 — LDB mixed-regime cell, NOT regime-comparable (see scoreboard) | `tp4-ldb-promoted-realistic-*`; the comparable cell comes from `tools/warm-replay-probe.py` on the final config |
| KV capacity | **5,486,463 tokens** (42 GiB FP8 KV/rank ×4, effective block 1,424) | `tp4/compose.yml` budgets, verified at startup |
| TP2 pair (maxwell-class hosts ×2) | same context; 1,918,359-token KV pool; ≥22 GiB MemAvailable kept | [`tp2/`](tp2/README.md) |

## Results — decode-max profile (optional, experimental)

Same base + a sha256-pinned **public** b12x mainline wheel + engine-flag envs
([`tp4/overlay.decode-max.yml`](tp4/overlay.decode-max.yml)). Buildable by anyone; see the
overlay header for the two-step recipe. **Status: ranking PENDING** — all its numbers are
from pre-promotion tuning state; same-state re-measure scheduled (this stack's tuning-state
shifts are proven ±17% on decode — the same evidentiary bar applies here).
| cell | value | note |
|---|---:|---|
| decode C1 cold | 64.8 tok/s ctx0, steps/s 37.0 (pre-promotion state) | default steps at 36.1 in current state (31.0 when this overlay was cut); 37.0 NOT re-verified in current state — edge unknown until same-state rerun |
| prefill | 3,3–3,4xx (pre-promotion state) | mainline b12x autotune drifted on this vLLM pairing; also unverified in current state |
| status | smoke-passed (arithmetic, tool-calls, vision) | NOT behavior-qualified: mainline b12x lacks eugr's 17 fork-ahead GB10/Spark commits — experimental |

## Against the reference stack (Fujitsu sparkring, same model + hardware class)

Their raw record [`r37-shared-tp4.json`](https://github.com/FujitsuPolycom/sparkring) is the
comparison anchor; our earlier apparent 5–25% "gap" was a metric artifact (their
`aggregate_decode_window_tps` warm/TTFT-excluded numbers compared against cold wall rates).
Like-for-like (full table + reasoning: [`SPARKRING.md`](SPARKRING.md)):

| cell | author r37 | this recipe (current steady state) | verdict |
|---|---:|---:|---|
| prefill cold 16K | 3,394–3,813 | **3,575** (high-cluster 4,5–4,7xx intermittent, unattributed) | parity, +0…5% (high cluster: +19…+39%) |
| C1 decode cold wall (ctx0) | 53.5 | **63.3–64.3** (steps/s 36.1) | **+18…+20%** (their fixture carries TTFT; weak-comparable) |
| C1 decode, realistic-fixture regime | 81.9 wall / 83.7–85.4 window (prefix-cached replays) | 58.3 LDB-dataset cell = **mixed regime (scout+cold+warm replays folded into one wall aggregate) — NOT comparable** to their window or our historical replay cells | method-matched warm-replay probe against the final config is the only cell that will answer this |
| C8 decode | cold wall 96.8–99.0 / warm wall 255–259.5 | **249.2** (ctx0 cold) | big lead vs their cold wall; ≈parity vs warm wall (−3%) |

Env-level levers A/B'd against the default (one variable per run, same instrument):
keep = b12x backends in speculative-config (accept 1.82→1.89), block-size 32 (author's
value; best cell +3.7% but cluster-suspect — kept as config-parity, effect unproven);
retired = async-scheduling off, size-based dispatch overrides,
`NCCL_CROSS_NIC=1` (all flat). Full table + the b12x pairing law (which wheels even start
with which vLLM): [`SPARKRING.md`](SPARKRING.md).

## Reproduce it

```bash
# 1) every rank: public base image
docker pull eugr/spark-vllm-b12x:nightly-20260925

# 2) every rank: weights — public checkpoint at the pinned QAD revision
hf download local-inference-lab/Qwen3.8-Flash-Next-NVFP4 \
  --revision 629bc3218833a38b475b719f34aa571666f4a03e --local-dir /srv/models/Qwen3.8-Flash-Next-NVFP4
#    (verify with the author's own SHA256SUMS from the sparkring profile dir; this repo's
#     deployment verified byte-identity — the QAD revision, not main, is what performs)

# 3) stage tp4/{compose.yml,serve.sh,config.json,<host>.env} per host; fill <host>.env
#    (fabric addrs, cache paths, KV budget). Cache dirs: compose binds
#    $CACHE_ROOT-{vllm,triton,flashinfer,b12x} — keep them per-image-profile, never shared.

# 4) start workers first, coordinator last; stop in reverse:
ssh worker1 'cd $DEPLOY_DIR && docker compose --env-file <host>.env -p myproj -f compose.yml up -d'
ssh coordinator 'cd $DEPLOY_DIR && docker compose --env-file coordinator.env -p myproj -f compose.yml up -d'

# 5) readiness = live API (log grepping false-passes on reused containers):
until curl -sf http://coordinator:8000/v1/models >/dev/null; do sleep 15; done   # ~6-10 min
./tools/smoke.sh        # behavior gates: arithmetic, typed tool-calls, vision
```

Bench with the same instrument we used: `tools/ldb-fast-matrix.sh` (llm_decode_bench
gated cycle) or `tools/bench-quick.py` (realistic-prompt matched probe). TP2 pair:
identical structure, [`tp2/`](tp2/README.md).

## Hardware prerequisites

Four GB10 Spark systems (121 GiB unified memory, `sm_121a`, Ubuntu 24.04 arm64, driver
580.x), **two ConnectX NICs each** on a dedicated IP routable fabric (this fleet: switched,
dual-port, 10.200.0.0/23 + 10.200.1.0/23 — substitute your own addressing in `<host>.env`),
`/dev/infiniband` in containers, ≥16 GiB MemAvailable on the limiting rank under load
(KV budget is a safety setting, not a capacity knob — violating the floor OOM-kills ranks
with exit 137 during long prefill).

## Provenance and invariants

- Base image: vLLM `dev/karmic-kraken` + b12x pin, built nightly by
  [`eugr/spark-vllm-docker`](https://github.com/eugr/spark-vllm-docker); it carries the
  HC-shard/coalesce/QSA/MoE-padding work this repo measured during its 09-16/17 campaign —
  the campaign's interim overlay images are RETIRED and were never public
  ([`COMPARISON.md`](COMPARISON.md), [`build/README.md`](build/README.md)).
- All ranks must run the SAME image ID (`docker image inspect -f '{{.Id}}'`, not tags).
- `NCCL_IB_MERGE_NICS=0`, `NCCL_IB_GID_INDEX=3`, HC TP-shard only at TP4 (`HC_TP=0` at TP2).
- Never pin `VLLM_USE_V2_MODEL_RUNNER=0` on this model: legacy runner lacks the
  `image_token_index` config field and crashes engine init.
- `serve.sh` env gates: `MTP_BACKENDS_IN_SPEC=0` (opt out of b12x spec backends),
  `ASYNC_SCHED=0`, `ATTENTION_BACKEND`, `RECURRENT_CHECKPOINT_POLICY`, `PROFILER_CONFIG`.

## Honesty box (quality limits, measured)

- FP8 KV + MTP3 produced one repeated-word loop ending in client timeout in a 20-case run
  (16 exact / 3 near / 1 loop); FP8 + MTP-off showed no loop in 8 requests but differs in
  tuning/scheduling, so attribution is open. FP8 KV was promoted by explicit operator
  choice: parity-or-better long-context prefill and 2.71× KV capacity for a ~4%
  single-stream decode cost. BF16-KV cells remain in [`data/results.jsonl`](data/results.jsonl).
- T0 same-prompt replay reproducibility is UNVERIFIED in either direction (runs disagreed;
  see `tools/long-prefix-check.py`, which deliberately does not gate on it).
- A one-off coordinator startup slowdown (2× MoE-kernel latency, same graph/clocks) was
  seen once and cleared on restart, cause unresolved: re-run any suspiciously slow first
  arm before recording it.
- The LDB prefill instrument is **cluster-bimodal** on this stack: ~3,55x vs ~4,4–4,7xx,
  same image ID, byte-equal cache inventories (verified). Leading hypothesis: the shipped
  default promotes TWO levers (block32 × b12x-spec-backends) each validated **singly** in
  campaign A/Bs — the combination was never measured as one arm; it interacts
  super-additively on decode (31→36.1 steps/s, beyond either alone) while costing
  prefill. Single-variable re-attribution arm (`MTP_BACKENDS_IN_SPEC=0`, block32 kept)
  is queued; until it lands, headline tables quote the LOW (steady, two-repeat) cluster
  and high-cluster rows are upper bound only.
- Method: decode rates depend on MTP acceptance — every decode row records `steps/s` and
  `mtp_acceptance_length` (tok/s ÷ accept = engine steps/s, acceptance-independent); read
  both before crediting any change.

## Repo map

| path | what |
|---|---|
| [`tp4/`](tp4/README.md) | four-Spark recipe: files + operational pitfalls (every lesson that cost us a cycle) |
| [`tp2/`](tp2/README.md) | two-Spark recipe |
| [`SPARKRING.md`](SPARKRING.md) | reference-stack scoreboard, env-lever verdicts, b12x pairing law |
| [`COMPARISON.md`](COMPARISON.md) | historical stock-vs-campaign A/B (superseded, kept as methodology evidence) |
| [`build/README.md`](build/README.md) | image provenance; what is and is NOT reproducible |
| [`data/`](data/README.md) | every number as [`results.jsonl`](data/results.jsonl) rows, incl. the author's r37 reference rows (wall + window + TTFT fields) |
| [`tools/`](tools/) | `smoke.sh`, `bench-quick.py`, `ldb-fast-matrix.sh`, cutover/rollback orchestration |
| `llms.txt` | agent entry point |
