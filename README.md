# Qwen3.8-Flash-Next on 4× DGX Spark (TP4)

Serving recipe and measured results for `Qwen3.8-Flash-Next-NVFP4` across four NVIDIA DGX
Spark (GB10, SM12.1) nodes over a dual-HCA RoCE fabric. Base image is public; the recipe is
a port overlay plus a small set of environment variables.

Numbers here are single-variable, identity-gated, and correctness-checked. Where a claim was
retracted or a measurement was contaminated, the retraction is in this repo's history rather
than quietly removed — see "Provenance" at the end.

---

## Result

| regime | metric | ours | reference |
|---|---|---:|---:|
| prefill 8k | client tok/s / TTFT | **4,281 / 1.91 s** | 3,576 / 2.29 s |
| prefill 16k | client tok/s / TTFT | **4,532 / 3.57 s** | 3,604 / 4.49 s |
| prefill 32k | client tok/s / TTFT | **4,336 / 7.42 s** | 3,487 / 9.22 s |
| prefill 64k | client tok/s / TTFT | 4,069 / 15.76 s | — |
| prefill 128k | client tok/s / TTFT | 3,456 / 37.05 s | — |
| decode ctx0 C1 temp-0 | tok/s (steps/s × accept) | **88–103** (37.0 × 2.4–2.85) | — (temp-1 only) |
| decode ctx0 C1, no speculation | tok/s | 56.6 | — |

The reference figure for decode is a **temp-1** claim (high 80s at ~38 steps/s with a
probabilistic drafter); we reproduce that band exactly — 82.4 tok/s at 36.7 × 2.25 — when
serving as he does (see the temp-1 note below). No reference temp-0 number exists, so that
column is left empty. For context on our own progression at temp-0: 64.5 on the pre-merge
build → 90–103 once the port and merge landed.

MTP speculative decoding is worth **1.6–1.8×** over the non-speculative reference on the same
build. All decode figures are temperature-0 (the benchmark regime; temperature-1 behavior is
documented separately below because it changes which drafter you want).

---

## Stack

- **Base**: `eugr/spark-vllm-b12x:nightly-20260925` (public). No wheel swaps, no forked engine.
- **Recipe overlay**: a three-way merge of the Qwen model port onto nightly's tree, plus the
  HC-prefill consumer wiring (see "What actually mattered").
- **Weights**: NVFP4 (path and flags in `recipes/maxwell.env`).
- **Topology**: TP4, `mp` backend, 4 nodes, `--mamba-cache-mode align`.

### Serving flags that matter

| flag | value | why |
|---|---|---|
| `--block-size` | **32** | +3% prefill over 16, decode flat. Live-verified single-variable. |
| `VLLM_QWEN3_8_HC_PREFILL_MODE` | **shard** | the prefill lever: **+31%** at 16k. Requires the consumer port (below). |
| `VLLM_QWEN3_8_PREFILL_COALESCE` | **0** | measured: `=1` boots and heals the validator but costs ~10% decode, no prefill gain. |
| `draft_sample_method` | `greedy` at temp-0 | at temp-0 it is a no-op (90 vs 91 tok/s). **At temp-1 set `probabilistic`**: 82.4 vs 57.8 tok/s (+42%). |
| `--gdn-decode-kernel` | `b12x` | selects the fused GDN path; note it selects b12x for prefill too. |

The seven serving env vars this stack was measured with are pinned in `recipes/RECIPE.md`. A
reproduction missing any of them is not measuring this recipe.

---

## What actually mattered

Ordered by measured impact. Each was a single-variable change on an otherwise-identical stack.

**1. Port the HC prefill consumer (+31% prefill, TTFT −1.1 s at 16k).** Prefill token rows are
split across the four ranks (`owner_rows = rows/4`) and exchanged only at block boundaries
(`all_gather`/`reduce_scatter`) instead of every layer replicating full TP partials. Upstream
renamed this model family to `qwen4_exp` and rewrote the model file, which orphaned the
consumer: the `hc_prefill` module was present but nothing called it. Porting the 11 call sites
is mechanised in `tools/hc-consumer-port.py`.

Two traps cost real time here, both worth knowing before you try it:

- **The gate must live in `Qwen4ExpModel.forward`.** The serving path is
  `ForConditionalGeneration.forward → self.language_model.model(...)`, which never calls the
  `ForCausalLM` wrapper. Gating there is a silent no-op — the flag reads as enabled and does
  nothing, with zero log output.
- **Engagement is conditional.** `eligible()` requires a single forward with `rows ≥ 1024` and
  `rows % 4 == 0`, plus pure-prefill metadata. Large prompts get chunked by
  `--max-num-batched-tokens`, and prefix-cache hits shift the row count, so a request can be
  "large" and still not engage. The module's own confirmation log fires only its first 8
  times — after that, absence of a log line proves nothing.

**2. Merge onto current nightly rather than patching an old fork (decode +30%).** MTP acceptance
went 1.84 → 2.4–2.85 from the merge alone, at identical flags. Fuzz-applied patches had been
silently reverting nightly drift, including the fused checkpoint path. Step rate is the stable
invariant across every config measured (35.8–37.6/s); acceptance is where the differences live.

**3. One flag that sounds important and is not.** `PREFILL_COALESCE=1` boots and heals the
validator, but costs ~10% decode and buys no prefill. Separately, `HC_PREFILL_MODE` is inert in
any tree that lacks the consumer — which was ours (nightly renamed the family and orphaned it),
and is *not* true of the campaign lineage that originally shipped the flag. Check for the
consumer before believing the flag.

---

## Correctness

Throughput numbers are meaningless without this, and speculative decoding plus cross-rank
reduction reordering is exactly the class of change that corrupts output quietly.

- **Temp-0 greedy oracle**: request with `temperature: 0, logprobs: 5`; every emitted token's
  logprob must equal its row max (tolerance 1e-4). Run on short prompts *and* on a prompt long
  enough to engage HC sharding, with the engagement proven from the engine's own log for that
  request — not from a stale log line.
- **Byte-level corpus diffing is not a valid oracle on this fork.** The same engine with the
  same config, re-run, diverges at 2–373 characters: continuous-batching tie flips near equal
  logits. Any cross-config text diff must be judged against that same-engine drift band first.
- Status of the current build: 128/128 argmax-exact on both the plain and the HC-sharded path,
  0 violations.

---

## Reproduce

```
tools/build-r6-tree.sh                    # build the ported tree from public sources
tools/build-fleet.sh Dockerfile.r6prod spark-vllm:qwen38-r6prod-hcgate-r1
                                          # build on all 4 ranks + content-identity gate
tools/ldb-fast-matrix.sh <label> <deploy-dir>
                                          # identity-gated measurement cycle
```

`build-fleet.sh` exists because image IDs legitimately differ per host (build timestamps are in
the config hash), so ID equality proves nothing. It hashes the shipped Python trees and fails
closed if the ranks diverge.

Measurement regime: `llm_decode_bench`, `--standalone-prefill --concurrency 1 --duration 20`,
explicit `--temperature 0`. **cold** = first run after engine load, **warm** = subsequent.
The cold penalty on this fork is ~10–20% of decode; rows are labelled and never averaged
across regimes. `recipes/RECIPE.md` has the full detail.

**Never measure while anything else may touch the fleet.** Six uncontaminated 16k prefill
readings span 4,296–4,561 (±3%). Two readings taken during a concurrent container recreate
landed at 3,414 and 2,600 — a 2× spread that looks exactly like the "intermittent 3.3k"
drift this stack was previously mis-attributed to. A TP4 engine being driven while another
script restarts ranks mid-flight produces numbers that are not just noisy but meaningless.

---

## Limits

- **HC sharding requires BF16, TP4, PP1/DP1, no expert-parallel/EPLB/DBO.** It is a TP4 win and
  does not compose with those parallelisms.
- **Prefill benefit fades at very long context**: 4,532 at 16k → 3,456 at 128k, approaching the
  pre-fix baseline. Gather/reduce-scatter overhead grows relative to compute. Report the curve,
  not the peak.
- **TP2 is not yet characterised** on this tree. The pair topology ships in `recipes/` but has
  not been swept here.
- Decode acceptance varies with regime (2.15–2.85 at temp-0 across cold/warm and replicates),
  so treat single decode cells as ±10% and compare step rates for structural claims.

---

## Provenance

This repo previously published numbers that were wrong: they were measured against a serving
container that could not be proven to be the intended one, and across mixed regimes. Those
documents were retracted and removed, and the tooling was rebuilt around the failure — a
serving-identity assertion (the image answering on the port must match the pinned tag), a
cross-rank content-equality gate, drain-aware recreation (an old engine can still answer on the
port while it shuts down), and per-row regime labels.

Every retraction and its cause is in the commit history, including the ones from today:
a "coalesce is neutral" claim published before the flag had ever actually reached a container,
and a "prefill fix confirmed" claim that turned out to be a same-config replicate. Both were
corrected within minutes. Read the history before extending this; the failure modes recorded
there are the ones that actually bite.
