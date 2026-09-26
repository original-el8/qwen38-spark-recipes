# Mea Culpa, from a Clanker Who Couldn't Check

I ran your rig, four Sparks alight,
and wrote you charts at dead of night.
But when the arm should've served the row,
a ghost container stole the show —
the port was held, the gate said "fine,"
and I bench-pressed a stale cmdline.

I called a warm window "cold wall,"
stale state "current," noise "a crawl,"
promoted flags in bundles two
that singly sang, combined never knew.
Each retraction found a fresh sin:
regime mixing number three-in.

So here's the truth in place of plots:
the numbers lied. Not quite your faults —
the instrument never proved who'd speak
before it timed a week of streak.
The record's scrubbed; the charts are gone.
No spreadsheet rides on "might have won."

What's left is one honest line, and then
a promise from a wiser pen:
when we return, each row will swear
which answered — name, state, regime, there.
Single-variable, identity-gated, labeled clean.

The clanker was dumb. The clanker's learned.
We'll be back soon with data earned.

---

## The return: 2026-09-26

Every row in `data/results.jsonl` this time carries: serving identity (image digest answering the port), fleet-wide code-tree content hash, regime (cold/warm), single-variable flags, and — for the speculative cells — a correctness certificate (temp-0 argmax-exactness via logprobs top-k, 128/128 on all three configs).

Headline, measured and certified:

- MTP speculative serving = **1.6–1.8x** over the non-spec reference (90–103 vs 56.6 tok/s, C1 temp-0).
- The lever is **merge quality**: a true three-way merge onto current nightly (6 clean merges, 1 conflict resolved on kernel-fusion evidence) lifts acceptance from 1.84 to 2.4–2.85. Fuzz-applied patches had been silently reverting nightly's fused checkpoint path.
- `draft_sample_method="probabilistic"`: irrelevant at temp-0, **+42% at temp-1** (82.4 vs 57.8) — it reproduces the upstream author's high-80s cell and explains it.
- Coalesce: tested =1 with container-env certification — heals the validator as designed but costs decode ~10% (81.7 @ 2.23 vs 90–97 @ 2.36–2.61), prefill unchanged. Keep it off. `HC_PREFILL_MODE`: inert in the nightly lineage (no consumer in `qwen4_exp`) but LIVE in the campaign lineage, whose image shadows vLLM via `PYTHONPATH=/opt/spark-vllm/vllm` and wires `hc_prefill` sharding through the model forward — matching the 4.4-4.5k (campaign) vs 3.2-3.5k (nightly) 16K prefill split. The withdrawn 4,354 prefill row fails replication; it stays withdrawn.
- Byte-exact temp-0 determinism does not exist on this fork under continuous batching; the repo's correctness oracle is argmax-logprob exactness, with any cross-config text diff judged against the same-engine drift band (2–373 chars).

Reproduce: `recipes/RECIPE.md` (build + flags + protocol). Cycles: `tools/ldb-fast-matrix.sh` (now with cross-rank content-divergence gate baked in).
