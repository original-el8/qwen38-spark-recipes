#!/usr/bin/env bash
# Qwen3.8-Flash-NEXT rank entrypoint — ONE script for both image variants.
#   stock base (eugr/spark-vllm-b12x nightly)  : defaults here, no extra env needed
#   overlay (built via ../build/README.md)     : activated purely by env (see overlay.yml)
# Behavior activated by environment (leave unset for stock base):
#   ATTENTION_BACKEND            e.g. B12X (overlay build only; stock image lacks the backend)
#   RECURRENT_CHECKPOINT_POLICY  e.g. aligned (overlay build only; stock CLI lacks the flag)
#   PROFILER_CONFIG              torch-profiler JSON (optional, either image)
#   HC_TP / VLLM_QWEN3_8_* envs   consumed inside the engine; this script only forwards env
set -euo pipefail
case "${NODE_RANK:?set NODE_RANK to 0, 1, 2, or 3}" in
  0|1|2|3) ;;
  *) printf 'NODE_RANK must be 0, 1, 2, or 3; got %s\n' "${NODE_RANK}" >&2; exit 2 ;;
esac
unset VLLM_GLM53_SPLIT_TARGET_BLOCK_SIZE VLLM_GLM53_SPLIT_MAMBA_BLOCK_SIZE
unset GLM53_KDA_PREFILL_BACKEND PREFILL_SCHEDULE_INTERVAL
export CUDA_VISIBLE_DEVICES=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

# engine entry: overlay images ship the venv at /opt/spark-vllm; stock images expose `vllm`
if [ -x /opt/spark-vllm/.venv/bin/python ]; then
  engine=(/opt/spark-vllm/.venv/bin/python -m vllm.entrypoints.cli.main serve /model)
else
  engine=(vllm serve /model)
fi

headless_args=()
if [[ "${NODE_RANK}" != 0 ]]; then
  headless_args+=(--headless)
fi

execution_args=()
if [[ "${QWEN38_ENFORCE_EAGER:-0}" == 1 ]]; then
  execution_args+=(--enforce-eager)
else
  compilation_config=${COMPILATION_CONFIG:-}
  if [[ -z "${compilation_config}" ]]; then
    compilation_config="{\"cudagraph_mode\":\"${QWEN38_CUDAGRAPH_MODE:-FULL_AND_PIECEWISE}\"}"
  fi
  execution_args+=(--compilation-config "${compilation_config}")
fi

speculative_args=()
if [[ "${MTP_TOKENS:-3}" != 0 ]]; then
  # MTP_BACKENDS_IN_SPEC=1: give the DRAFT path the same b12x backends as the target
  # (sparkring author's profile does this; plain mtp:3 shorthand leaves the drafter on
  # generic kernels -> draft logits disagree with the b12x verifier -> low acceptance).
  if [[ "${MTP_BACKENDS_IN_SPEC:-0}" == "1" ]]; then
    speculative_args=(
      --speculative-config
      "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP_TOKENS},\"moe_backend\":\"b12x\",\"attention_backend\":\"B12X\"}"
    )
  else
  speculative_args=(
    --speculative-config
    "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP_TOKENS}}"
  )
  fi
fi

variant_args=()
if [[ -n "${ATTENTION_BACKEND:-}" ]]; then
  variant_args+=(--attention-backend "${ATTENTION_BACKEND}")
fi
if [[ -n "${RECURRENT_CHECKPOINT_POLICY:-}" ]]; then
  variant_args+=(--recurrent-checkpoint-policy "${RECURRENT_CHECKPOINT_POLICY}")
fi
if [[ -n "${PROFILER_CONFIG:-}" ]]; then
  variant_args+=(--profiler-config "${PROFILER_CONFIG}")
fi
if [[ "${ASYNC_SCHED:-1}" != "0" ]]; then
  variant_args+=(--async-scheduling)   # ASYNC_SCHED=0 => off (sparkring ships without;
fi                                     # untested axis for hybrid-GDN step throughput)

exec "${engine[@]}" \
  "${headless_args[@]}" \
  "${execution_args[@]}" \
  "${speculative_args[@]}" \
  --served-model-name Qwen3.8-Flash-Next qwen38-flash-next-nvfp4 \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-4}" \
  --distributed-executor-backend mp \
  --nnodes "${NNODES:-4}" \
  --node-rank "${NODE_RANK}" \
  --master-addr "${MASTER_ADDR}" \
  --master-port "${MASTER_PORT:-29507}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.80}" \
  --block-size "${BLOCK_SIZE:-16}" \
  --max-model-len "${MAX_MODEL_LEN:-262144}" \
  --max-num-seqs "${MAX_NUM_SEQS:-16}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-8192}" \
  --load-format "${LOAD_FORMAT:-instanttensor}" \
  --kv-cache-memory-bytes "${KV_CACHE_MEMORY_BYTES:-45097156608}" \
  --kv-cache-dtype "${KV_DTYPE:-fp8_e4m3}" \
  --mamba-cache-mode align \
  --dtype bfloat16 \
  --quantization modelopt_mixed \
  --gdn-decode-kernel b12x \
  --mm-encoder-tp-mode data \
  --mm-processor-cache-gb 0 \
  --limit-mm-per-prompt '{"image":8,"video":2}' \
  --enable-prefix-caching \
  --moe-backend "${MOE_BACKEND:-b12x}" \
  --linear-backend "${LINEAR_BACKEND:-b12x}" \
  --no-enable-flashinfer-autotune \
  --enable-chunked-prefill \
  --generation-config vllm \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 \
  "${variant_args[@]}" \
  --trust-remote-code
