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
| prefill 8K / 16K / 32K / 64K / 128K (tok/s, cold) | 4,35x–4,4 / up to **4,704** / 4,437 / 4,25x / 3,825 | llm_decode_bench fast matrix, `tp4-ldb-block32-*`, `tp4-ldb-promoted-*` |
| decode C1 cold (tok/s, ctx0 / 64K) | **57.3–60.9** | same, `tp4-ldb-*` decode rows; MTP-normalized steps/s 31.0–31.7, accept 1.8–1.9 |
| decode C1 realistic prompts (tok/s) | 75.9±24.9 measured on stock `64d4c3e0`; current-config cell `tp4-bq-promoted-*` pending | `tools/bench-quick.py` matched probe |
| decode C8 (tok/s, ctx0) | 336.2 (campaign image `953b00ee`); current-image cell pending | `tp4-qsa-selection-decode-ctx0-c8`, `tp4-ldb-promoted-c8` |
| KV capacity | **5,486,463 tokens** (42 GiB FP8 KV/rank ×4, effective block 1,424) | `tp4/compose.yml` budgets, verified at startup |
| TP2 pair (maxwell-class hosts ×2) | same context; 1,918,359-token KV pool; ≥22 GiB MemAvailable kept | [`tp2/`](tp2/README.md) |

## Results — decode-max profile (optional, experimental)

Same base + a sha256-pinned **public** b12x mainline wheel + engine-flag envs
([`tp4/overlay.decode-max.yml`](tp4/overlay.decode-max.yml)). Buildable by anyone; see the
overlay header for the two-step recipe.

| cell | value | note |
|---|---:|---|
| decode C1 cold | **64.8 tok/s** ctx0 (58.4–60.9 range across cells) | steps/s 37.0 vs default 31.0 (**+18%**), accept 1.65–1.83 |
| prefill | 3,3–3,4xx | −23% vs default: mainline b12x autotune drifts on this vLLM pairing |
| status | smoke-passed (arithmetic, tool-calls, vision) | NOT behavior-qualified: mainline b12x lacks eugr's 17 fork-ahead GB10/Spark commits — experimental |

## Against the reference stack (Fujitsu sparkring, same model + hardware class)

Their raw record [`r37-shared-tp4.json`](https://github.com/FujitsuPolycom/sparkring) is the
comparison anchor; our earlier apparent 5–25% "gap" was a metric artifact (their
`aggregate_decode_window_tps` warm/TTFT-excluded numbers compared against cold wall rates).
Like-for-like (full table + reasoning: [`SPARKRING.md`](SPARKRING.md)):

| cell | author r37 | this recipe | verdict |
|---|---:|---:|---|
| prefill cold 16K | 3,394–3,813 | 4,4–4,7xx | **+24…+39%** |
| C1 decode, realistic-fixture regime | 81.9 wall / 83.7–85.4 window | 87.5±10.7 (historical image); current-config pending | parity on current evidence |
| C8 decode | 255–259.5 warm wall | 336.2 (campaign image); current pending | lead (image-labelled) |

Env-level levers we A/B'd against the default (one variable per run, same instrument):
keep = block-size 32 (16K prefill +3.7%), b12x backends in speculative-config (accept
1.82→1.89); retired = async-scheduling off, size-based dispatch overrides,
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
