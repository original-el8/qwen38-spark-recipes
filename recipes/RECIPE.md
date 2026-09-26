# Qwen3.8-Flash-Next TP4 on 4x DGX Spark — validated recipe (r6c)

Base: `eugr/spark-vllm-b12x:nightly-20260925`. No forks, no wheel replacements.

## Build (per rank, ~25 s)
1. Extract nightly's vllm tree: `docker create` + `docker cp /usr/local/lib/python3.12/dist-packages/vllm`.
2. True three-way merge of the port files. Inputs per file:
   `ours` = nightly tree, `base` = local-inference-lab/vllm@9e5d1793fa, `theirs` = PR-825 head 8f1257c1 (plus PR-386's four `b12x/sequence/ple` files copied verbatim into `b12x/sequence/ple/`).
   `git merge-file ours base theirs`. The 2026-09-26 run: 6 of 7 modified files merged clean; GDN `qwen_gdn_linear_attn.py` had ONE real conflict (both sides edited the checkpoint store) — resolved by taking NIGHTLY (its fused runner already exports checkpoints via `checkpoint_export=True`; the PR's manual triton store block would double-store).
   After merge: delete any `autotune=False` kwarg lines in `ple_layer.py` (stale API, nightly removed the kwarg).
3. GDN coalesce gate (port adaptation): in `qwen_gdn_linear_attn.py:get_kv_cache_spec`, when `not self.prefill_checkpoint_blocks` AND `envs.VLLM_QWEN3_8_PREFILL_COALESCE`, return `replace(spec, num_prefill_checkpoint_blocks=1)` so GDN and PLE groups declare the uniform value the nightly `mamba_hybrid` validator requires.
4. `docker build -f Dockerfile.r6` (bakes `verify6.py` gauntlet: aborts if `export_checkpoint`, HC knobs, or `autotune`-free state are missing).

## HC token-row-sharded prefill (the prefill fix, 2026-09-26)
The reference author's +34% cold-prefill came from `VLLM_QWEN3_8_HC_PREFILL_MODE=shard`: prefill token rows are split across the 4 TP ranks (`owner_rows = rows/4`) and exchanged only at complete block boundaries (`all_gather`/`reduce_scatter`) instead of every layer replicating full TP partials.

Our nightly tree had the module (`hc_prefill.py` was ported) but **no consumer**, so the flag was inert and prefill sat at 3.2-3.5k. The port is 11 call sites in `vllm/models/qwen4_exp/nvidia/model.py` (`tools/hc-consumer-port.py` applies it):
- `hc_prefill.configure(self, vllm_config, envs.VLLM_QWEN3_8_HC_PREFILL_MODE)` in `Qwen4ExpModel.__init__`, plus the `_register_b12x_row_parallel_collective(..., "hc_block_output", ...)` hook when enabled.
- `defer_hc_reductions` per decoder layer -> `reduce_results=not defer` on linear_attn / self_attn / mlp / MoE, with the deferred all-reduce re-issued as `hc_owner.reduce(...)`.
- `hc_owner.gather/local/reduce` threading through the decoder forward (attn input, MLP input, deepstack, final mixer).
- **The entry gate MUST live in `Qwen4ExpModel.forward`**, not only `Qwen4ExpForCausalLM.forward`: the serving path is `ForConditionalGeneration.forward -> self.language_model.model(...)`, which skips the ForCausalLM wrapper entirely. Gating in the wrong method = silent no-op (exactly the trap hit here: 0 `ELIGIBLE` calls until relocated).

Result (TP4, 16k, cold, block32): **4,561 tok/s / 3.55 s TTFT** (from 3,174-3,477 / 4.65-5.12 s), decode unchanged at 91.4 tok/s.

## Required serving environment (7) — all cells measured with exactly these
```
VLLM_MXFP8_LM_HEAD=1
VLLM_LM_HEAD_A16=1
VLLM_MTP_NVFP4_LM_HEAD=1
VLLM_B12X_MOE_FP4_LAYER_MAX_INPUT_SCALE=w13
VLLM_QWEN3_8_FLASH_NEXT_OVERLAP=1
VLLM_QWEN3_8_FLASH_NEXT_MTP_COMPACT=1
VLLM_GDN_SPEC_DECODE_METADATA_FASTPATH=1
```
Verified live via `docker exec <rank> env` on all four ranks (2026-09-26). A reproduction missing any of these is not measuring this recipe.

## Flags that matter (2026-09-26 evidence, all identity+content-gated, correctness-certified)
| flag | verdict |
|---|---|
| merge quality (three-way vs fuzz-patch) | THE lever: accept 1.84 -> 2.4-2.85 at temp-0; fuzz patches silently reverted nightly drift the merged tree keeps |
| `DRAFT_SAMPLE=probabilistic` | no-op at temp-0 (90.0 vs 91.1); REQUIRED at temp-1 (82.4 vs 57.8, +42%) — fixes the greedy-draft-into-sampled-target acceptance collapse |
| `VLLM_QWEN3_8_PREFILL_COALESCE` | keep **0**. Tested =1 (certified: rank container env + StartedAt audit): boots — GDN gate heals the mamba_hybrid validator — but decode costs ~10% (81.7 tok/s @ accept 2.23 vs 90-97 @ 2.36-2.61); prefill unchanged. No benefit found |
| `VLLM_QWEN3_8_HC_PREFILL_MODE` | LIVE in the campaign lineage, DEAD in ours. The campaign image ships a PYTHONPATH-shadowing vendored vLLM at `/opt/spark-vllm/vllm` whose `models/qwen3_8_flash_next/model.py:481` calls `hc_prefill.configure(self, vllm_config, envs.VLLM_QWEN3_8_HC_PREFILL_MODE)` and threads `hc_prefill.RowOwnership` through 12 forward-path sites; campaign compose sets `shard`. The nightly lineage maps this family to upstream `qwen4_exp`, whose model file has NO hc_prefill consumer - so the flag is inert and the ported `hc_prefill.py` modules are dead weight there. Prefill evidence: campaign-lineage flags-off 16K = 4,378-4,536 tok/s vs nightly-lineage 3,174-3,475. Candidate root cause of the prefill gap; porting the consumer call sites into `qwen4_exp` is the fix |

## Reference numbers (ctx0 C1 temp-0, warm caches)
non-spec 56.6 tok/s -> MTP 90-103 tok/s = 1.6-1.8x. Prefill 16k: 3.3-3.4k tok/s, TTFT ~4.8-5.1 s.
The withdrawn-era 4,354 prefill row does not replicate under identity gates.

## Correctness protocol (required before believing any temp-0 spec number)
`/v1/completions` with `temperature:0, logprobs:5`: every emitted token's logprob must equal its row max (tolerance 1e-4). All three r6c configs: 128/128 exact.
BYTE-level corpus diffing across runs is NOT a valid oracle on this fork: same engine + same config re-captures diverge at 2-373 chars (continuous-batch tie flips near equal logits). First-diff depth must be compared against the same-engine drift band before indicting any config.

## Executable reproduction
```
tools/build-r6-tree.sh                     # tree2/vllm from public sources (checks: CLEAN x6, TREE2-BUILD-READY)
docker build -f recipes/Dockerfile.r6 .    # from a ctx with tree2/ and recipes/ple5/ (bakes verify6 gauntlet)
```
ple5/ ships the four measured b12x PLE files (sha256 ee518db0 510aa7cd 52814e84 789cb018).

## Measurement protocol (what the data-row regimes mean)
- LDB canonical: `--standalone-prefill --concurrency 1 --duration 20 --contexts 0`, explicit `--temperature`; cycles default to the profile's 0.0.
- **cold** = first LDB run against a freshly loaded engine; **warm** = second+. On this fork the cold penalty is ~10-20% of decode; rows are labeled, never averaged across regimes.
- Accept band {2.15–2.85} at temp-0 reflects regime + near-tie tie-flip variance under continuous batching (same-config replicate spread: 87.1–103.5 tok/s); step rate is the stable invariant (35.8–37.4).
