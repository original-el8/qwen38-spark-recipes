# Stock karmic-nightly vs campaign overlay image — matched probe

> **SUPERSEDED 2026-09-26.** The `eugr/spark-vllm-b12x:nightly-20260925` tag absorbed the
> campaign overlay's content (HC shard/coalesce, QSA selection, MoE padding; rows
> `tp4-stock-karmic-*` match-or-beat `tp4-campaign-overlay-probe-*` cells). The campaign
> image and `overlay.yml` are retired; the published default is the stock nightly
> (now with block-size 32 + b12x spec-config backends as recipe defaults). The
> historical probe below stays as methodology evidence — one tool, both stacks, cold-cache
> by construction.

The question this probe answered: what did serving on the public
`eugr/spark-vllm-b12x` nightly (Path A, then `64d4c3e0`) give up versus the fleet's
campaign overlay (Path B, [`build/README.md`](build/README.md))?

Measured **2026-09-25 on this fleet** with one tool against both stacks, same topology, same
KV budget, same prompt generator — `tools/campaign-ab-probe.sh` replays it in one command
(campaign profile up behind an API-readiness gate → `smoke.sh` → `bench-quick.py` → stock
restore). Probe semantics: cold prefix cache by construction (time-unique token salts;
server-reported `local_compute` counters; zero cache hits verified in both arms); decode cells
prime their 8K context **outside** the timed window. Random-token prompts → absolute values
sit below the campaign's realistic-prompt numbers; **only paired deltas are meaningful.**

| cell (tok/s) | Path A stock nightly `64d4c3e0` | Path B campaign `953b00ee` | campaign vs stock |
|---|---:|---:|---|
| prefill 8192 cold | 3,639.3 ± 9.9 | **4,885.0 ± 41.5** | **+34.2%** |
| decode C1 cold | 75.9 ± 24.9 | **87.5 ± 10.7** | +15.3% |
| decode C8 cold | 221.0 ± 35.2 | **256.4 ± 42.6** | +16.0% |
| decode C1 @8K primed | **82.6 ± 10.5** | 71.4 ± 6.4 | −13.6% |
| decode C8 @8K primed | **265.4 ± 24.4** | 197.1 ± 10.8 | −25.7% |

Rows: `data/results.jsonl` ids `tp4-stock-karmic-*` / `tp4-campaign-overlay-probe-*`.

## Reading it

- **Path B wins the headline cells decisively**: +34% on cold 8K prefill, +15–16% cold decode.
  That gap is the campaign's own merged-and-local work (HC token-ownership prefill, QSA
  selection fusion, MoE padding, plus newer b12x `master` than the nightly was built with at
  campaign time). If long-prefill throughput is your bottleneck, the overlay was
  measurably better — **moot since 2026-09-26**: the current public nightly carries the
  same content. (Historical build contexts + dist archives are retained **fleet-local**
  only — `/home/jasonc/spark_vllm/build-contexts/<tag>/` — and are NOT published; see the
  reproducibility note in [`build/README.md`](build/README.md).)
- **Path A wins primed-context decode** (−13.6% / −25.7% for the overlay). Two plausible
  contributors, not separated by this probe: the stock nightly's newer b12x master kernels,
  and `VLLM_QWEN3_8_FLASH_NEXT_MTP_COMPACT` (default-on stock; absent campaign) changing MTP
  acceptance on long cached prefixes. For chat-serving traffic that mostly decodes on cached
  history, stock is competitive-to-better.
- Both arms passed the same behavior smoke (arithmetic, `qwen3_xml` tool call, typed args).
- Single A/B session, one restart each; the campaign's flake rule applies — re-run before
  making a cluster-purchasing decision on a single cell.

## What the published recipe is, and why

As of 2026-09-26 the recipe ships **one default and one documented profile**:
`tp4/` + `tp2/` root files on the stock nightly (Path A, public hub tag, no fleet-local
bits, block32 + b12x spec backends defaulted in), and
[`tp4/overlay.decode-max.yml`](tp4/overlay.decode-max.yml) — a thin rebuild
(newer b12x wheel over the same nightly) for decode-dominant serving: C1 64.8 tok/s,
prefill ≈author-level. Author-parity claims and the full lever evidence:
[`SPARKRING.md`](SPARKRING.md).
