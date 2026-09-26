# Qwen3.8-Flash-Next vs the reference stack (Fujitsu SparkRing) — scoreboard and lever evidence

Source: [`FujitsuPolycom/sparkring`](https://github.com/FujitsuPolycom/sparkring)
`profiles/` + `performance/records/` + `runtime/releases/`, pulled 2026-09-26.
Same model, same hardware class: `qwen38-flash-next-qad-tp4` = Qwen3.8-Flash-Next NVFP4 QAD
revision `629bc3218833a38b475b719f34aa571666f4a03e`, 262,144 ctx, 16 seqs, 8192 batched
tokens, MTP3, FP8 KV, b12x backends, 4 Sparks. Their testbed is a switchless direct cycle;
this recipe's example fleet is switched.

## VERDICT 2026-09-26: the "author gap" was largely a measurement artifact — match-or-lead on prefill, cold-wall C1, and C8; the realistic-fixture C1 cell stays OPEN until a method-matched warm-replay probe runs

Their own machine-readable record `performance/records/qwen38-flash-next/r37-shared-tp4.json`
distinguishes two decode metrics that our earlier campaign conflated:

- `aggregate_wall_tps` — includes TTFT (what llm_decode_bench and any cold client measure)
- `aggregate_decode_window_tps` — decode-window only, prefix-warm fixtures, TTFT excluded

| cell (tok/s) | author r37 | this recipe (same methodology) | Δ |
|---|---:|---:|---|
| C1 decode **cold wall** | 53.5 (7.9K-token fixture; TTFT 1.80 s dominates) | **63.3–64.3** ctx0 (steady, 2 repeats; steps/s 36.1) — ctx0 flatters us slightly | **+18…+20%**, weak-comparable |
| C1 decode realistic-prompt regime | 81.9 warm wall / 83.7–85.4 window (prefix-cached replays) | **OPEN**: current LDB-dataset cell 58.3 is mixed-regime (folds scout/cold/warmup into wall) — not comparable; historical replay cells (75.9±24.9, 87.5±10.7) are from retired images | open — settled only by same-method replay probe on the final config |
| C8 decode | cold wall 96.8–99.0 / warm wall 255–259.5 | **249.2** ctx0 cold, steps-tier same as C1 | lead vs their cold wall; ≈parity vs warm wall (−3%) |
| prefill cold 16K | 3,394 (4.828 s / 16,384 tok, QAD-TP4-PREFILL.md) · 3,647–3,813 (r37) | **3,575** steady (×2 repeats); high tuning cluster 4,5–4,7xx intermittent, unattributed | parity +0…5% (high cluster +19…+39%) |
Earlier mid-2026 rows here ("author 4,853@8K, +6.6% gap"; "73–89 tok/s C1") were
warm-window/relayed numbers — superseded by the raw JSON above. `data/results.jsonl` carries
the authoritative author rows as `author-r37-*` (wall + window + TTFT fields preserved).

Weights are NOT a factor: our `/models/Qwen3.8-Flash-Next-NVFP4` verified
**byte-identical to their pinned QAD revision** (`sha256sum -c` of their profile
`SHA256SUMS` passes on maxwell). "main is not a substitute for the QAD revision" —
we are the QAD revision.

## Env-lever table (each arm = one variable on the stock nightly + flags-off base; fast LDB matrix)

| lever | verdict |
|---|---|
| `--block-size 32` (their shipped value) | **KEEP** — 16K prefill 4,704 (+3.7% best cell); decode flat. Now the recipe default |
| b12x backends inside `--speculative-config` | **KEEP** — acceptance 1.82→1.89 (+4%), C1 +6%; plain `mtp:3` shorthand leaves the drafter off b12x. Now default in `serve.sh` (`MTP_BACKENDS_IN_SPEC` opt-out) |
| async-scheduling OFF (they don't pass it) | **RETIRED** — steps/s and accept identical to ON |
| `QWEN_DISPATCH_MODE=both` + `AR_BYTES=20480` | **RETIRED** — steps flat; 16K rides the same high cluster as block32 (unattributed) |
| `NCCL_CROSS_NIC=1` | **RETIRED** — flat/negative; RoCEnante size-based selection bypasses it |
| operator 7-flag suite on stock nightly | neutral in every isolated cell (earlier matched probes) — but see decode-max: the suite **synergizes with newer b12x** |

## The b12x pairing law (source-bound, not env)

eugr's nightly pins a b12x commit on a **diverged fork line** (17 ahead / 13 behind
b12x main): mainline commits from 2026-09-17→18 (#388 decode-scale conversion, #389 MoE
padding, MTP-verifier tuning, #394 QSA winner fusion) are absent from the pin, while vLLM
in the same nightly REQUIRES mainline-new attributes:

- b12x `d5139035` (#394 tip wheel): **startup FAIL** — vLLM wants
  `TuningRequirement.rejected_count`; older mainline lacks it.
- b12x `e39b437b` (2026-09-26 main wheel, sha256 `345dad40…`): **runs**; decode
  steps/s **+9…+18%** (36.7–37.0 with the flag suite vs 31.0 base; C1 64.8 tok/s) but
  prefill autotune drifts to 3,3–3,4xx (−23%) — unrescued by flags or block32.

Consequence: **newer b12x is a trade, not a free win.** We ship stock nightly as default
(prefill parity vs author; C1 cold wall +18…+20%) and publish
`tp4/overlay.decode-max.yml` (b12x `e39b437b` + flag suite + block32) — ranking PENDING:
its 37.0 steps/s was measured pre-promotion tuning state vs the default's current-state
36.1; tuning-state shifts of ±17% are proven on this stack, so same-state re-runs decide
it. Rebuild recipe and wheel provenance are
in that overlay's header. Author's claimed patched-NCCL (+5–6.8% prefill) and SparkCache
(TTFT restore) remain their engine-bound stack — neither is needed for parity, which is
already met or exceeded per the scoreboard.

Numerics caveat: b12x mainline wheels do NOT contain eugr's 17 fork-ahead commits
(GB10/Spark-specific fixes). The decode-max profile passed smoke (arithmetic, tool-calls,
vision) but has NOT had full behavioral/quality qualification — treat it as experimental;
the shipped default carries zero fork-divergence risk.

## Reading their A/B records correctly

`r37-shared-tp4.json` "≈0 delta" is **overlay-on vs overlay-on** (baseline carries
`prefill_overlay_manifest_sha256`) — not evidence HC sharding is worthless. Our own matched
on/off A/B (COMPARISON.md history, +34% cold prefill) remains that evidence; the current
nightly already contains it.

## Retained negatives (their records — don't chase)

batch 11,392; OMP=1 + CPU pinning; recurrent-state fusion; 8-channel NCCL; SIRCL
promotion (2.4–3% slower than patched NCCL); "assume transport speedup ⇒ application
speedup" (their explicit caution). Switchless-only pins (`NCCL_SWITCHLESS_RING_ONLY`,
Ring algo, 4-HCA lists, opposite-path b12x) are N/A on our switched fabric.

## Open question to the reference authors

- Their **fast-matrix LDB `steps/s (accept len)` row** is the only datum that would
  attribute their warm-window decode behavior between steps and acceptance. Not blocking:
  cold-wall parity is already established.
- Recipe invariant discovered the hard way: pinning `VLLM_USE_V2_MODEL_RUNNER=0` crashes
  engine init on this stack (legacy path lacks `Qwen3_8FlashNextConfig.image_token_index`)
  — never pin off.
