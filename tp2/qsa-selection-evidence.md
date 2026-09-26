# Qwen QSA stable-selection fusion on DGX Spark

Status: **research-only**. The investigation is complete.

The B12X candidate constructs exact QSA winners with fewer kernel launches. Isolated complete-QSA transactions preserve attention outputs, selected positions, and recurrent state exactly on TP2 and TP4. After an unchanged cached restart, TP4 prefill measures 4,422 / 4,277 / 3,835 tokens per second at 8K / 64K / 128K: +0.1% / +2.2% / +4.0% against the retained matching baseline. TP2 measures 3,350 / 3,058 / 2,829 tokens per second. Both configurations pass all 26 long shared-prefix and edited-history checks.

The result supports the exact selector fusion as a modest long-context prefill improvement. Decode throughput varies with MTP acceptance, so the larger C1 token-rate increases do not establish a corresponding kernel speedup. The first TP4 startup had an intermittent Maxwell execution slowdown that cleared after restart; its cause remains unresolved. Both attempts are retained below. The healthy TP4 candidate remains running, while the selected baseline is unchanged.

## Implementation and identity

The selector retains the existing exact radix top-k and final score/ID sort. It removes an index-remap pass whose output is overwritten, then combines threshold reduction, winner counting, and stable winner emission into one CuTe kernel. Selection uses six GPU stages before the change and three afterward. Score computation, attention arithmetic, carry staging, scratch layout, and the public QSA contract remain unchanged.

Each CUDA block handles one query row. Warps scan contiguous score ranges and retain earlier global group IDs at threshold ties. Row counts, lengths, score stride, and chunk offsets remain runtime inputs; only device identity and the fixed selection budget enter the compile key. Pool-scaled address arithmetic uses 64-bit integers.

The structural idea follows the exact threshold-and-gather selector reviewed in [DwarfStar](https://github.com/antirez/ds4/tree/8db1d1d155cb0400a86a86b9c62d0defb3a6148b). The CuTe implementation preserves B12X ordering and chunk-carry semantics. The review is in `research/ds4-qwen-review-20260917`.

- B12X: `d2d5368d6c8a5cc43d79bfc8a4a5c58db584fc47`, branch `codex/qwen-qsa-selection-20260917`.
- vLLM: `76061de4bff2adc741cb25018ca79991263228be`; unchanged from the selected FP8 baseline.
- Image: `sha256:953b00ee516cac5f295bfb26863d3cfdcb5f950a6461bed76e11a84281468346`.
- Source, archive, program, and GPU provenance: [component-provenance.json](component-provenance.json), [image receipt](image-r1/candidate-build.json), and [distribution receipt](image-distribution-r1/distribution-receipt.json).

The immutable image is verified on all four Sparks. No source overlay is used in serving. The implementation has no additional runtime option.

## Component correctness and timing

Seventeen tests pass inside the immutable image: adversarial stable selection at budgets 512/2048, carry and tie ordering, empty rows, changing row counts and lengths, frozen-resolution graph capture/replay, prepared program retention, and cache addressing past the signed 32-bit product boundary. Six scalar-versus-CuTe transaction replay cases also pass. The independent scalar oracle resolves its own Triton programs; production uses CuTe.

The complete-QSA comparison runs on Ampere with serving stopped. Baseline and candidate use the same inputs, binding, prepared scorer and attention programs. Every case requires bit-exact nonzero output, selected positions and persistent state, plus allocation-free graph replay. Each arm has thirty CUDA-event samples in balanced order. Both captured graphs include state restoration. Prefill uses 7,520 query rows; these are component workloads, not full-model prefill measurements.

| Component case | TP4 latency reduction | TP2 latency reduction |
|---|---:|---:|
| speculative, 4 rows, 8K context | 2.36% | 1.23% |
| throughput, 8 rows, 8K context | 2.76% | 3.36% |
| prefill, 7520 rows, 8K context | 3.27% | 3.33% |
| prefill, 7520 rows, 64K context | 8.86% | 9.25% |
| prefill, 7520 rows, 128K context | 10.76% | 10.85% |

The source used for the TP4 component timings precedes removal of unused helper definitions and addition of explicit tensor-device ownership. The final source is exercised by the TP2 comparison and immutable-image tests. Exact commits and archives for each attempt are retained. [Raw component summaries](component-summary.json).

## Serving controls

TP4 retains FP8 E4M3 KV, 42 GiB of KV memory per rank, MTP3, resident PLE/Engram, InstantTensor loading, batch-token cap 8,192, maximum 16 sequences, context 262,144, HC prefill sharding, coalescing, aligned recurrent checkpoints and retention interval zero. Effective cache blocks remain 1,424 tokens and capacity remains 5,486,463 tokens. Both RoCEnante NICs remain enabled, with 2 MiB all-reduce and 16 MiB all-gather thresholds.

The matching TP4 control is the frozen `qwen38-checkpoint-spacing-20260917/baseline-return-performance` study. Its measurement script and twenty-one prompts match byte-for-byte. Input IDs, zero cache hits, fixed output length, and rank health are checked during requests. No fresh baseline serving matrix is completed or used. [Control verification](retained-baseline-verification.json).

TP2 runs on Maxwell and Ampere with the same image, FP8 KV, MTP3 and resident PLE. It uses the supported replicated HC path and 16 GiB of KV memory per rank, with effective blocks of 2,848 tokens and capacity of 1,918,359 tokens. Historical TP2 results use BF16 KV and an older vLLM image, so they do not isolate this patch. TP2 numbers below are absolute serving measurements; its paired component comparison isolates the selector change.

Prefill uses nominal 8K/64K/128K prompts, each with two template tokens, and the server prefill duration. Decode generates 512 tokens per request and counts tokens in the interval when all requests are decoding. C8 throughput is aggregate. MTP acceptance and every sample remain in [assessment.json](assessment.json).

## Serving measurements

| Workload | Retained TP4 baseline | TP4 initial candidate | TP4 candidate restart | TP2 candidate |
|---|---:|---:|---:|---:|
| prefill-8192 | 4,415.1 | 2,927.0 (2 samples) | 4,421.5 (3 samples) | 3,349.9 (3 samples) |
| prefill-65536 | 4,183.6 | 3,135.9 (2 samples) | 4,276.9 (3 samples) | 3,058.2 (3 samples) |
| prefill-131072 | 3,688.4 | 3,192.5 (2 samples) | 3,834.5 (3 samples) | 2,828.7 (3 samples) |
| decode-ctx0-c1 | 77.3 | 63.0 (2 samples) | 81.9 (3 samples) | 58.2 (3 samples) |
| decode-ctx0-c8 | 331.6 | 235.9 (2 samples) | 336.2 (3 samples) | 231.9 (3 samples) |
| decode-ctx8192-c1 | 75.8 | 59.5 (2 samples) | 82.2 (3 samples) | 55.8 (3 samples) |
| decode-ctx8192-c8 | 318.2 | 260.9 (1 sample) | 327.9 (3 samples) | 226.6 (3 samples) |

Values are tokens per second. The initial TP4 matrix is stopped after the slowdown repeats; incomplete cells are excluded. Its recorded means remain visible and are not substituted with diagnostic timings.

The completed TP4 restart and TP2 matrices each contain three repeats of seven workloads. Every performance request has zero cached input tokens, and all 42 before/after metric snapshots per matrix report zero preemptions. Rank processes remain healthy throughout each completed matrix.

TP4 C1 decode increases by 6.0% on short prompts and 8.4% at 8K, while mean draft acceptance rises from 49.0% to 53.5% and from 48.2% to 54.8%, respectively. The corresponding approximate engine step rates change only +0.4% and +0.3%. C8 token throughput increases by 1.4% and 3.0%. These serving results do not isolate a decode benefit from this patch. The approximate step metric includes batch ramp-up and drain; raw per-repeat acceptance and timing remain in the assessment.

### Initial TP4 slowdown

All 182 saved B12X tuning configurations match the retained baseline and agree across ranks. The slow short-decode trace shows Maxwell taking about 286 microseconds per unchanged MoE microkernel, versus about 139 microseconds on Ampere, with identical launch geometry. In the 8K-context trace, Maxwell returns to about 135 microseconds and the ranks are balanced. The same CUDA graph and node IDs are used. The other ranks spend the additional time in collectives. Stable-selection construction and sorting account for less than 1% of GPU kernel time.

A bounded runtime probe observes steady active SM clocks and zero throttle flags on every rank, with no competing heavy CPU workload on Maxwell. This identifies uneven rank execution as the observed bottleneck; it does not establish its root cause or prove independence from the candidate. The candidate is not qualified by its component speedup alone. [Trace groups](diagnostic-traces-r1/kernel-groups.json), [runtime telemetry](runtime-state-r1), and [early-stop record](candidate-tp4-performance-stopped.json).

The API multimodal warmup takes 253.9 seconds on the initial TP4 startup. A native stack sample shows CPU tensor work during that interval. Benchmark timing starts after successful startup and smoke checks, so this delay is excluded from throughput.

After the TP2 test, the same TP4 image, source and settings are restarted with their populated compilation caches. Two diagnostic 8K probes measure 4,431 and 4,439 tokens per second, then a separate full three-repeat matrix produces the restart results above. All 182 tuning configurations still match the retained control. No kernel edits, host reboot, or clock, power, network or memory configuration changes intervene. Recovery after restart does not identify the original cause.

## Correctness, memory and final state

- `candidate-tp4`: long shared-prefix checks not completed; sampled minimum MemAvailable in GiB: maxwell: 16.03, ampere: 19.23, faraday: 19.62, hertz: 19.62.
- `candidate-tp2`: 26/26 exact shared-prefix/history-edit cases; sampled minimum MemAvailable in GiB: maxwell: 22.27, ampere: 25.25.
- `candidate-tp4-restart`: 26/26 exact shared-prefix/history-edit cases; sampled minimum MemAvailable in GiB: maxwell: 16.05, ampere: 18.85, faraday: 19.74, hertz: 19.63.

Startup smoke checks cover arithmetic, tool calls, prefix reuse, concurrent retrieval, and vision. The long shared-prefix matrix uses 64K at C4 and 128K at C8, including edits near the middle of the history. These bounded checks do not resolve the previously disclosed FP8 long-reasoning limitations.

The TP4 candidate remains healthy at `http://maxwell:8000/v1`, with the immutable image above on all four ranks, zero container restarts or OOM events, and no queued requests at the final snapshot. Minimum sampled host memory available during qualification is 16.05 GiB on Maxwell and at least 18.85 GiB on the other ranks. TP2 retains at least 22.27 GiB on Maxwell and 25.25 GiB on Ampere. Final service identity and rank health are recorded in [final-state/state.json](final-state/state.json). No baseline promotion or PR publication is part of this investigation.

## Historical attempts

The cancelled `baseline-performance` directory contains an unnecessary duplicate control run stopped at the user’s correction; it is excluded from comparisons. Earlier component failures concern invalid harness case names, compiler subprocess entry protection, baseline program preparation, and stale test assumptions. Their logs are retained. The corrected image tests and completed paired transactions are the correctness evidence.
