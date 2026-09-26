#!/usr/bin/env bash
# Qwen3.8-Flash-Next on stock eugr/spark-vllm-b12x nightly (vllm dev/karmic-kraken + b12x master).
# Rank-parallel entrypoint: NODE_RANK selects coordinator vs --headless worker.
set -euo pipefail

case "${NODE_RANK:?set NODE_RANK to 0, 1, 2, or 3}" in
  0|1|2|3) ;;
  *) printf 'NODE_RANK must be 0, 1, 2, or 3; got %s\n' "${NODE_RANK}" >&2; exit 2 ;;
esac

# Defensive: never inherit platform defaults baked by ancestor images.
unset VLLM_GLM53_SPLIT_TARGET_BLOCK_SIZE VLLM_GLM53_SPLIT_MAMBA_BLOCK_SIZE
unset GLM53_KDA_PREFILL_BACKEND PREFILL_SCHEDULE_INTERVAL

export CUDA_VISIBLE_DEVICES=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

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
  speculative_args=(
    --speculative-config
    "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP_TOKENS}}"
  )
fi

exec vllm serve /model \
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
  --limit-mm-per-prompt '{"image":1}' \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --async-scheduling \
  --generation-config vllm \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 \
  --trust-remote-code
