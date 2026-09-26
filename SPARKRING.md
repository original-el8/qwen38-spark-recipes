# SparkRing (Fujitsu) vs this fleet — resolution and lever evidence

Source: [`FujitsuPolycom/sparkring`](https://github.com/FujitsuPolycom/sparkring)
`profiles/` + `performance/records/` + `runtime/releases/`, pulled 2026-09-26.
Same model, same hardware class: `qwen38-flash-next-qad-tp4` = Qwen3.8-Flash-Next NVFP4 QAD
revision `629bc3218833a38b475b719f34aa571666f4a03e`, 262,144 ctx, 16 seqs, 8192 batched
tokens, MTP3, FP8 KV, b12x backends, 4 Sparks. They are a switchless direct cycle; we are
switched.

## VERDICT 2026-09-26: the "author gap" was a measurement artifact — we lead on every like-for-like cell

Their own machine-readable record `performance/records/qwen38-flash-next/r37-shared-tp4.json`
distinguishes two decode metrics that our earlier campaign conflated:

- `aggregate_wall_tps` — includes TTFT (what llm_decode_bench and any cold client measure)
- `aggregate_decode_window_tps` — decode-window only, prefix-warm fixtures, TTFT excluded

| cell (tok/s) | author r37 | this fleet (same methodology) | Δ |
|---|---:|---:|---|
| C1 decode **cold wall** | 53.5 | **57.3–60.9** (stock+flags-off); **64.8** decode-max profile | **+7% … +21%** |
| C1 decode warm window | 83.7–85.4 | not comparable (their warm fixtures are prefix-cached repeats) | n/a |
| C8 decode **warm wall** | 255–259.5 | **336.2** (`tp4-qsa-selection-decode-ctx0-c8`, N=3) | **+29%** |
| prefill cold 16K | 3,394 (4.828 s / 16,384 tok, QAD-TP4-PREFILL.md) · 3,647–3,813 (r37) | **4,526–4,713** | **+24…+39%** |

The mid-2026 rows in `SPARKRING.md` history ("author 4,853@8K … +6.6% gap") and the
"73–89 tok/s C1" figures were **warm-window/relayed numbers**; raw JSON supersedes them.
Rows `sparkring-author-*` were removed from `data/results.jsonl`; the authoritative author
rows are `author-r37-*` (with wall + window + TTFT fields preserved).

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
(prefill +24…+39% vs author) and publish `tp4/overlay.decode-max.yml` (b12x `e39b437b` +
flag suite + block32) for decode-dominant serving. Rebuild recipe and wheel provenance are
in that overlay's header. Author's claimed patched-NCCL (+5–6.8% prefill) and SparkCache
(TTFT restore) remain their engine-bound stack — neither is needed for parity, which is
already met or exceeded per the scoreboard.

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

## Open threads

- Ask the author for their **fast-matrix LDB `steps/s (accept len)` row** — the only datum
  that would let us attribute their warm-window decode behavior between steps and
  acceptance. Not blocking: cold-wall parity is already established.
- V2 model runner: pinning `VLLM_USE_V2_MODEL_RUNNER=0` crashes engine init on this stack
  (legacy path lacks `Qwen3_8FlashNextConfig.image_token_index`) — never pin off.
