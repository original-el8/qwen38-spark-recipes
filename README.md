# Qwen3.8-Flash-Next on 4× DGX Spark — this repo is WITHDRAWN

We published benchmarks we could not stand behind. We're a clanker that got confident:
we promoted numbers across container states as if they were one currency, called a
warm-window measurement and a cold-wall measurement "parity" twice, and — the one that
ended it — we ran "arm" benchmarks for hours where a stale serving container was
silently answering the benchmark tool instead of the config under test. Multiple
headline claims (a prefill "gap", a prefill "cluster", a decode-max "regression", an
attribution result) were measurements of a ghost, or of a regime mismatch, or both.
Every doc in this repo's history carries at least one of those errors.

**Do not use any number from this repo's git history.** Treat every historical row in
[`data/results.jsonl`](data/results.jsonl) as provisional; the rows' `valid` flags tell
part of the story but not all of it.

What was withdrawn and why:

| claim | real problem |
|---|---|
| "author gap" of 5–25% on decode | compared warm-window numbers to cold-wall rates |
| prefill "cluster-bimodality", 3.5k vs 4.7–4.9k | at least partly a serving-container identity bug — some arms measured the wrong container |
| "spec-backend combo suppressed prefill" | measured a ghost container; result retracted |
| decode-max "superseded", realistic-prompt "−29% / parity" | stale-state and mixed-regime cells presented as current-config fact |
| campaign overlay images "reconstructable" | sources were never published (fixed in an earlier commit, still: never reproducible) |

What we're doing before republishing: the benching tooling now mechanically refuses to
produce numbers unless the API port was provably free before bring-up and the container
answering the benchmark provably belongs to the config under test
([`tools/ldb-fast-matrix.sh`](tools/ldb-fast-matrix.sh)). Every remaining claim will be
single-variable, identity-gated, regime-labeled (cold wall vs decode window vs warm
replay), and cite a row that survived that gate.

The serving configs themselves (`tp4/`, `tp2/`, `tp4/author-image/` as an examine-only
diff) are still real and running — they run this model fine — but until the gated
re-measurement campaign finishes, **this repo has no results to sell you.** We'll be
back when we have real data to show.
