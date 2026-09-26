# TP4 runtime details — every knob and why

Serving command (from `serve.sh`, expanded for rank 0):

```bash
vllm serve /model \
  --compilation-config '{"pass_config":{"fuse_act_quant":true},"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"],"cudagraph_capture_sizes":[1,2,3,4,5,6,8,10,12,16,20,24,32,40,48,64,80]}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --served-model-name Qwen3.8-Flash-Next qwen38-flash-next-nvfp4 \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 4 --distributed-executor-backend mp --nnodes 4 --node-rank 0 \
  --master-addr 10.200.0.12 --master-port 29507 \
  --gpu-memory-utilization 0.80 --block-size 16 \
  --max-model-len 262144 --max-num-seqs 16 --max-num-batched-tokens 8192 \
  --load-format instanttensor \
  --kv-cache-memory-bytes 45097156608 --kv-cache-dtype fp8_e4m3 \
  --mamba-cache-mode align --dtype bfloat16 --quantization modelopt_mixed \
  --gdn-decode-kernel b12x --mm-encoder-tp-mode data --mm-processor-cache-gb 0 \
  --limit-mm-per-prompt '{"image":1}' \
  --enable-prefix-caching --enable-chunked-prefill --async-scheduling \
  --generation-config vllm --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml --reasoning-parser qwen3 --trust-remote-code
```

## Engine flags

| flag | why |
|---|---|
| `--tensor-parallel-size 4 --nnodes 4 --distributed-executor-backend mp` | one GB10 per host; vLLM multiprocessing headless-rank mode (no Ray) |
| `--speculative-config mtp/3` | checkpoint ships an MTP drafter head; MTP3 won the depth sweep (MTP2/3/4 compared twice; 8,192-token batched budget removes the "suboptimal performance" warning MTP causes at 2,048) |
| `--kv-cache-dtype fp8_e4m3 --kv-cache-memory-bytes 45097156608` | 2.71× KV tokens per GiB vs BF16; 42 GiB/rank sized to keep ≥16 GiB MemAvailable (measured floor 16.05 GiB). Existing unit descales; QSA selector cache stays BF16 |
| `--block-size 16` | requested block; hybrid (GDN+attention) manager inflates the *effective* attention block to 1,424 tokens under FP8 geometry and pads up to ~8.3% — capacity, not a context change |
| `--mamba-cache-mode align` | required by the GDN recurrent layers (SSM state aligned to attention blocks) |
| `--quantization modelopt_mixed` | NVFP4 experts + MXFP8 attention mixed checkpoint; activations BF16 (`--dtype bfloat16`) |
| `--gdn-decode-kernel b12x` | b12x decode kernel for gated-deltanet state updates |
| `--load-format instanttensor` | operator-preferred loader (fast checkpoint→GPU); the b12x loader experiment was deliberately set aside. Stock nightly also supports `b12x` (io_uring + async CUDA copies) and `fastsafetensors` |
| `--max-num-batched-tokens 8192` | MTP + prefill coalescing need the wider chunk; standing operator preference |
| `--enable-prefix-caching --enable-chunked-prefill --async-scheduling` | production serving defaults validated with the recurrent-checkpoint "aligned" policy |
| `--tool-call-parser qwen3_xml --reasoning-parser qwen3 --enable-auto-tool-choice --generation-config vllm` | the model's native XML tool-call format and thinking channel |
| `--mm-encoder-tp-mode data --mm-processor-cache-gb 0 --limit-mm-per-prompt '{"image":1}'` | vision tower replicated per rank, no processor cache; keep multimodal from stealing KV memory |
| `--compilation-config` | `fuse_act_quant` pass + FULL_AND_PIECEWISE CUDA graphs; capture sizes cover every MTP verify/draft batch shape up to 80 (= 16 seqs × (3+1) + margin) |

## Environment (compose.yml)

### Fabric (both RoCE interfaces, NCCL fallback above RoCEnante caps)

| var | value | why |
|---|---|---|
| `GLOO_SOCKET_IFNAME`/`NCCL_SOCKET_IFNAME` | `enP2p1s0f0np0` | fabric NIC, never the LAN iface |
| `NCCL_IB_HCA` | `rocep1s0f0,roceP2p1s0f0` | both ConnectX functions |
| `NCCL_IB_GID_INDEX` | `3` | RoCE v2 GID on this fabric |
| `NCCL_IB_ADDR_FAMILY/RANGE` | `AF_INET`, `10.200.0.0/23` | both NIC subnets are in scope |
| `NCCL_IB_MERGE_NICS` | `0` | dual-function cabling needs per-function QPs (required) |
| `NCCL_NET`/`NCCL_CUMEM_ENABLE`/`NCCL_NVLS_ENABLE`/`NCCL_CROSS_NIC` | `IB`, `0`, `0`, `0` | no SM121 NVLS; explicit IB transport |
| `NCCL_PROTO` | `LL,LL128,Simple` | keep small-message protocols |
| `VLLM_ENABLE_ROCE_ALLREDUCE` | `1` | RoCEnante replaces NCCL in decode collectives ≤ caps |
| `VLLM_ROCE_ALLREDUCE_MAX_SIZE` / `VLLM_ROCE_ALLGATHER_MAX_SIZE` | `2MB` / `16MB` | tuned cutoffs; NCCL above |
| `B12X_ROCE_SPIN_LIMIT` | `50000000` | ~half-minute peer-failure detection |
| `B12X_ROCE_CACHE_DIR` | `/root/.cache/vllm/b12x-roce` | compiled RoCE proxy cache (inside profile cache namespace) |
| `VLLM_HOST_IP` | per-rank fabric IP | worker self-identity in rendezvous |

### Model/runtime

| var | value | why |
|---|---|---|
| `VLLM_QWEN3_8_FLASH_NEXT_HC_TP` | `1` | TP-shards the HyperConnection workspace (stock-karmic equivalent of the campaign `VLLM_QWEN3_8_HC_PREFILL_MODE=shard` + coalescing). Requires `hc_lowrank % tp_size == 0`; **TP4-only** |
| `VLLM_USE_V2_MODEL_RUNNER` | `1` | v2 worker path the fleet runs qualified on the campaign stack |
| `VLLM_SSM_CONV_STATE_LAYOUT` | `DS` | conv-state layout the b12x GDN kernel expects |
| `VLLM_PLE_CPU_OFFLOAD` | `0` | PLE/Engram table stays resident (disk/Grace tier is a GB300 trick, not needed on 4×GB10) |
| `TORCH_CUDA_ARCH_LIST`/`FLASHINFER_CUDA_ARCH_LIST` | `12.1a` | GB10 arch propagation |
| `CUTE_DSL_ARCH` | `sm_121a` | CuTe DSL kernels target |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | fragmentation control under unified memory |
| `VLLM_PLUGINS` | `""` | no loader plugin on stock nightly (b12x loader would set `b12x_loader`) |
| `VLLM_WORKER_MULTIPROC_METHOD` | `spawn` | fork+CUDA is unsafe |
| `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE` | `1` | air-gapped serving; weights are local |
| cache dirs (`VLLM_CACHE_ROOT`, `TRITON_CACHE_DIR`, `FLASHINFER_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR`, `B12X_COMPILE_CACHE_DIR`, `B12X_CUTE_COMPILE_CACHE_DIR`) | container paths, each bind-mounted from a **per-profile host namespace** | rollback + A/B integrity: never reuse another profile's namespaces |

### Container contract (compose.yml)

host networking + IPC, `shm_size: 32g`, `/dev/infiniband` devices, `cap_add: [SYS_PTRACE,
IPC_LOCK]`, `security_opt: [seccomp=unconfined, label=disable]`, ulimits `memlock=-1`,
`nofile=1048576`, `stack=67108864`, weights mounted read-only with the config override on top,
`pull_policy: never` discipline (pre-pull + image-ID check, then no registry access).

## Startup / stop / rollback

- Start: ranks 1→2→3 (headless), then rank 0. Stop: 0 first, then workers.
- Readiness: rank-0 log `Application startup complete`; then `tools/smoke.sh` must pass; then
  inspect **every** rank (state, image ID, restart count, no OOM, no preemption counters).
- Rollback: stop candidate (0→3), then start the previous profile from its own directory with
  its own cache namespaces (both image sets stay installed on all four hosts).
