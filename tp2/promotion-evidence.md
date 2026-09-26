# Qwen FP8 KV baseline with 16 GiB host headroom

Status: qualified for the recorded operational checks and selected as the baseline at the user's explicit request. Full FP8 numerical and long-reasoning equivalence remain unqualified; the known limitations are preserved below.

The deployed profile uses FP8 E4M3 main KV and a 42 GiB per-rank KV budget. It exposes **5,486,463 KV tokens**, compared with 2,024,110 tokens for the preceding BF16/28 GiB profile: **2.71× capacity**. The effective attention block is 1424 tokens. The per-request context limit remains 262,144 tokens, and the request cap remains sixteen. Capacity is aggregate KV storage, not a larger per-request context limit.

In the three-repeat comparison, prefill means are 8.0–26.4% lower and decode means are 3.9–23.4% lower than the preceding BF16 profile. Several cells show substantial variation, and 91 common kernel configurations change during independent tuning. The performance difference cannot be attributed solely to KV precision. The raw samples are retained, including the slower runs.

The immutable native-160 image is reused: `sha256:a2fbec2f2e348631fad52413c61a1f72ed4df62c6d31c13b5cd6aa90a3540f7c`. The vLLM source is `76061de4bff2adc741cb25018ca79991263228be`; B12X is `72baebbda2a200762f37223ef25aa64e8a5cb734`. Runtime settings retain TP4 across Maxwell/Ampere/Faraday/Hertz, MTP3, 8,192 batched tokens, resident PLE, InstantTensor loading, HC prefill sharding, prefill coalescing, and both RoCEnante interfaces with the 2 MiB/16 MiB thresholds. Main KV uses the existing unit descales; the QSA selector cache remains BF16. No source or image change is made for this promotion.

## Available memory

The budget targets approximately 16 GiB of **Linux MemAvailable** on the limiting rank. This includes reclaimable filesystem cache and differs from the MemFree counter. The same KV budget is used on all ranks; coordinator overhead leaves Maxwell with less available memory than the workers.

| Spark | Minimum sampled under checks, GiB | After checks, GiB |
|---|---:|---:|
| maxwell | 16.93 | 16.98 |
| ampere | 19.55 | 19.70 |
| faraday | 19.66 | 19.68 |
| hertz | 19.71 | 19.73 |

Samples are taken approximately every three seconds from readiness through bounded correctness, sixteen concurrent 8K requests, and repeated throughput measurements. They may miss short transients. No swap-out activity occurs during the measured checks. All ranks finish healthy with the expected image, zero restarts, no OOM kills, and no preemptions. Startup samples are retained separately in the same trace and excluded from the serving headroom minimum.

## Operational checks and throughput

The deployment smoke passes arithmetic, tool-call parsing, prefix reuse, eight concurrent requests, and image understanding. Seventeen bounded checks pass exact retrieval at approximately 8K/64K/128K, cache replay, changed instructions using cached context, and concurrent JSON responses. Sixteen additional simultaneous 8K requests return the expected complete JSON arrays and normal stops. Raw requests, responses, counters, and all-rank logs are retained.

| Workload | BF16 / 28 GiB, tokens/s | FP8 / 42 GiB, tokens/s | Change |
|---|---:|---:|---:|
| 8K prefill | 4422.6 ± 38.8 | 3319.7 ± 634.5 | -24.94% |
| 64K prefill | 4190.5 ± 17.1 | 3856.1 ± 497.4 | -7.98% |
| 128K prefill | 3684.8 ± 7.6 | 2713.4 ± 202.0 | -26.36% |
| C1 short decode | 85.6 ± 7.4 | 66.5 ± 14.2 | -22.37% |
| C8 short decode | 343.6 ± 7.2 | 330.3 ± 5.3 | -3.88% |
| C1 decode, 8K context | 83.2 ± 2.8 | 70.1 ± 11.7 | -15.81% |
| C8 decode, 8K context | 329.0 ± 7.6 | 252.0 ± 42.2 | -23.41% |

Values are mean ± sample standard deviation over three measured runs after warmup. C8 rates count aggregate tokens in the shared all-active decode interval. Nominal prefill lengths are 8K, 64K, and 128K; the saved prompts contain 8,194, 65,538, and 131,074 tokens. Decode uses fixed 512-token outputs at short and 8K contexts, with verified cold prefix caches.

The BF16 reference is the frozen same-day native-160 measurement. The arms are not interleaved, and both KV precision and allocation budget change. FP8 namespaces tune independently: 91 of 182 common query configurations differ from the BF16 reference, with all four ranks agreeing on the FP8 selections. [kernel-selection-comparison.json](kernel-selection-comparison.json) retains those differences. MTP acceptance and output paths also vary. These measurements describe the selected serving configuration and do not isolate the cost of quantization. [assessment.json](assessment.json) retains all samples, acceptance fractions, approximate step rates, and memory observations.

## Known quality limits

The preceding FP8/MTP3 LAVD test returned sixteen exact answers, three near answers, and one repeated-word loop ending in a client timeout. Its cause remains unresolved. FP8 with MTP disabled returned seven exact and one near answer without a loop in eight requests; tuning and scheduling differences prevent attribution to MTP. Several numerical comparisons are inconclusive because the BF16 references are themselves unstable.

The user requested FP8 promotion after disclosure of the unresolved loop. This operational validation does not rerun the full LAVD/Estonia suite, establish general numerical equivalence, or relabel that earlier failure as a pass. The earlier [FP8 quality report](../qwen38-fp8-lavd-20260917/README.md), [MTP0 diagnostic](../qwen38-fp8-mtp0-lavd-20260917/README.md), and [numerical investigation](../qwen38-fp8-kv-correctness-20260917/README.md) retain their original results.

## Deployment and recovery

The selected profile is `/home/jasonc/spark_vllm/deployments/qwen38-fp8-kv42-mtp3-20260917/profile.json`. The mutable selection pointer is `deployments/qwen38-flash-next-baseline.json`; `deployments/qwen38-runtime-preferences.json` records FP8 and the 16 GiB available-memory target for future work. Copies of both selections at promotion are frozen with this report.

The preceding BF16 profile, image, and caches remain intact at `/home/jasonc/spark_vllm/deployments/qwen38-moe-native160-mtp3-20260917/profile.json`. The recorded switch helper stops the coordinator before workers and starts workers before the coordinator. [promotion.json](promotion.json) identifies the selected image and explicit quality limitation, [runtime-diff.patch](runtime-diff.patch) records the runtime changes, and [integrity-manifest.json](integrity-manifest.json) freezes the profile and evidence. Mutable baseline pointers and future runtime preferences are not frozen external inputs.
