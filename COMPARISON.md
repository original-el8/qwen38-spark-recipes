# Stock karmic-nightly vs campaign overlay image — matched probe

The question this repo has to answer honestly: what do you give up by serving on the public
`eugr/spark-vllm-b12x` nightly (Path A, the published default) instead of the fleet's
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
  campaign time). If long-prefill throughput is your bottleneck, the overlay is measurably
  better and fully reconstructable (build contexts + image dist archives retained on the build
  host, `/home/jasonc/spark_vllm/build-contexts/<tag>/`, `~/.cache/*-distribute.tar.zst`).
- **Path A wins primed-context decode** (−13.6% / −25.7% for the overlay). Two plausible
  contributors, not separated by this probe: the stock nightly's newer b12x master kernels,
  and `VLLM_QWEN3_8_FLASH_NEXT_MTP_COMPACT` (default-on stock; absent campaign) changing MTP
  acceptance on long cached prefixes. For chat-serving traffic that mostly decodes on cached
  history, stock is competitive-to-better.
- Both arms passed the same behavior smoke (arithmetic, `qwen3_xml` tool call, typed args).
- Single A/B session, one restart each; the campaign's flake rule applies — re-run before
  making a cluster-purchasing decision on a single cell.

## What the published recipe is, and why

As of 2026-09-26 **Path B is the published and serving default** (`tp4/overlay/`,
`tp2/overlay/` — live files mirrored into the repo): operator decision after this A/B, since
the overlay is fully reconstructable (`build/README.md`, build contexts + dist archives
retained on maxwell) and the prefill gap dominates serving traffic. **Path A** (`tp4/`,
`tp2/` root files) remains published unchanged as the stock fallback: rebuilds from a public
hub tag with no fleet-local bits, wins primed-context decode, and is one gated
`compose down`/`up` away.
