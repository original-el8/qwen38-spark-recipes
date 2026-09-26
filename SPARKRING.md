# SparkRing (Fujitsu) profile levers vs this fleet — applicability analysis

Source: [`FujitsuPolycom/sparkring`](https://github.com/FujitsuPolycom/sparkring) `profiles/`
(26 profiles) + `performance/records/` + `spark_transport/nccl/`, pulled 2026-09-26.
They serve the **same model** (`qwen38-flash-next-qad-tp4` = Qwen3.8-Flash-Next NVFP4,
262,144 ctx, 16 seqs, 8192 batched tokens, MTP3, FP8 KV, b12x backends) on 4 Sparks in a
**switchless direct cycle**. We are **switched** — their ring-specific pins are N/A for us;
engine-side levers are not. Their own published QAD TP4 prefill (~3,785–3,813 tok/s @16K) is
**below our campaign overlay** (measured 4,885 @8K, `COMPARISON.md`), so this is a menu of
extra toggles, not a recipe to adopt wholesale.

## APPLIES — cheap env/flag A/Bs against the Path B overlay (test one variable per run)

| Lever | Theirs (verbatim, `profiles/qwen38-flash-next-qad-tp4/config.json`) | Ours | Their reported effect |
|---|---|---|---|
| NVFP4 MTP proposal head | `VLLM_MTP_NVFP4_LM_HEAD=1`, `VLLM_LM_HEAD_A16=1` | not set | GLM MTP3 profile records +8.2% C1 raw decode; same mechanism class as our MTP_COMPACT |
| MTP overlap | `VLLM_QWEN3_8_FLASH_NEXT_OVERLAP=1` | not set (stock has `MTP_COMPACT=1` only) | shipped-on in their profile; no isolated delta published |
| Size-based collective dispatch | `QWEN_DISPATCH_MODE=both`, `QWEN_DISPATCH_AR_BYTES=20480`, `QWEN_DISPATCH_TRACE=0` | image defaults only (`VLLM_ROCE_ALLREDUCE_MAX_SIZE=2MB`, `ALLGATHER_MAX_SIZE=16MB` — numerically the same ceilings) | policy A/B record selected `mode=both`; ours may already be equivalent — check `QWEN_DISPATCH_TRACE=1` before believing either way |
| KV block size | `--block-size 32` | 16 | no delta published; interacts with FP8-KV page layout + prefix-cache hit granularity — measure |
| VRAM ceiling | `--gpu-memory-utilization 0.85` | 0.80 | capacity, not speed; watch MemAvailable floor ≥16 GiB |
| Allocator | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | unset | no delta published |
| Thread/affinity hygiene | `OMP_NUM_THREADS=16`, `NCCL_IGNORE_CPU_AFFINITY=1` | unset | their *negative* record says OMP=1 + pinning did nothing; 16 is their shipped value |
| Cross-NIC | `NCCL_CROSS_NIC=1` | `0` | topology-dependent — switched fabric means this deserves its own A/B, could go either way |
| Spec-config backends | speculative-config carries `moe_backend=b12x`, `attention_backend=B12X` (plain `mtp:3` shorthand skips them) | plain mtp:3 | their claim; unquantified — plausible MTP-draft-path speedup |
| LM-head/runner bits | `VLLM_SSM_CONV_STATE_LAYOUT=DS`, `VLLM_USE_V2_MODEL_RUNNER` (release components) | unset | no delta published |
| Async scheduling | they do **not** pass `--async-scheduling` | we do | untested axis: run our overlay once with it off |

## APPLIES — heavier, source-bound (needs their engine stack, not just envs)

- **Patched-NCCL dual-PCI-domain routing** (`spark_transport/nccl/DUAL_PCI_DOMAIN.md`):
  patches NCCL **2.30.7 source only**, `LD_PRELOAD` + `VLLM_NCCL_SO_PATH` to the rebuilt lib,
  then `NCCL_IB_PRESERVE_PCI_DOMAIN=1` (their QAD profile ships `EXTENDED_IPV4_GIDS=0`;
  the four-GID `=1` mode is documented separately). Contributor-measured on a **direct
  cycle**: 84/168 MiB all-reduce 2× faster, +5.0–6.8% prefill, decode unestablished.
  Our NCCL 2.28.3 → flags inert (confirmed by their doc). Would need the rebuild in-image.
- **SparkCache persistent KV** (`profiles/qwen38-flash-next-qad-tp4/sparkcache.json`):
  their connector + two native `.so`s, lease contract source-bound to their vLLM patch
  snapshot, 4 GiB/rank NVMe budget. TTFT records: 50,768-tok prefill **18.23 s → 1.56 s**
  restored (TP2), 13.7K media 10.17 → 1.66 s; taxes 0–0.6% prefill, 0.4–1.3% C1 decode.
  Massive for repeated long-prefix chat; engine-version-bound, not adoptable by env alone.
- **HC shard/coalesce + PLE + paired QSA** (their PR refs LIL vllm 779, b12x 386/387):
  our Path B overlay already carries equivalent user-side work (`VLLM_QWEN3_8_HC_PREFILL_MODE=shard`,
  `PREFILL_COALESCE`, QSA fusion) — ALREADY HAVE, and our measured +34% prefill exceeds their
  published results.

## N/A for this fleet

- `NCCL_SWITCHLESS_RING_ONLY=1`, `NCCL_ALGO=Ring`, `NCCL_MIN/MAX_NCHANNELS=4`, 4-HCA
  `NCCL_IB_HCA==…:1` list, `B12X_ROCE_OPPOSITE_PATHS=2` — switchless ring/pair wiring.
- SIRCL (their A/B: 2.4–3% **slower** than patched NCCL anyway).
- LD_PRELOAD of their patched NCCL without the matching rebuilt 2.30.7 lib.

## Retained negatives (their records — don't chase)

batch 11,392; OMP=1 + CPU pinning; recurrent-state fusion; 8-channel NCCL; SIRCL promotion;
"assume transport speedup ⇒ application speedup" (their explicit caution).

## Reading their A/B records correctly

`r37-shared-tp4.json` "≈0 delta" is **overlay-on vs overlay-on** (baseline `beeb3225` carries
`prefill_overlay_manifest_sha256`) — it is *not* evidence that HC sharding is worthless; no
overlay on/off record exists in their repo. Our own matched A/B (`COMPARISON.md`, +34% cold
prefill) remains the on/off evidence.

## STATUS 2026-09-26: operator env set applied

The seven operator engine flags (`VLLM_MXFP8_LM_HEAD`, `VLLM_LM_HEAD_A16`,
`VLLM_MTP_NVFP4_LM_HEAD`, `VLLM_B12X_MOE_FP4_LAYER_MAX_INPUT_SCALE=w13`,
`FLASH_NEXT_OVERLAP=1`, `MTP_COMPACT=1`, `GDN_SPEC_DECODE_METADATA_FASTPATH=1`) are pinned in
`tp4/overlay.yml` + `tp2/overlay.yml` and live on the fleet. Matched probe (both topologies):
prefill neutral, TP4 primed-C8 +8.4%, everything else inside acceptance-noise. `OVERLAP=0`
in the old composes was confirmed template carry-forward, **not** an A/B verdict — no
campaign record ever evaluated it and the original runners defaulted it ON.

## Test order on this fleet (one variable per run, bench-quick v3 + long-prefix gate)

1. `VLLM_MTP_NVFP4_LM_HEAD=1` + `VLLM_LM_HEAD_A16=1` (decode C1/C8 + acceptance)
2. `VLLM_QWEN3_8_FLASH_NEXT_OVERLAP=1`
3. `--block-size 32` (prefill scale + prefix-cache hits + KV capacity re-pin)
4. async-scheduling OFF vs ON
5. `NCCL_CROSS_NIC=1`, spec-config backends
6. allocator/OMP hygiene as free wins if 1–5 are neutral
