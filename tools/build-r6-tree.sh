#!/usr/bin/env bash
# Reproduce the r6c tree (content hash d11c515af330d4d6 @ 2026-09-26) from public sources.
# Inputs: eugr/spark-vllm-b12x:nightly-20260925 (local image), GitHub API access (gh),
#         ple5/ next to this script (or fetch from local-inference-lab/b12x PR #386 head).
# Run on any host with docker + gh + git. Output: ./tree2/vllm
set -eu
NIGHTLY=${NIGHTLY:-eugr/spark-vllm-b12x:nightly-20260925}
BASE=9e5d1793fa            # 825's dev/karmic-kraken base (2026-09-21)
THEIRS=8f1257c1            # PR-825 head
REPO=local-inference-lab/vllm
MOD=( envs.py
  model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py
  model_executor/models/qwen3_next.py
  models/qwen4_exp/nvidia/ple_attn.py
  models/qwen4_exp/nvidia/ple_layer.py
  v1/core/sched/scheduler.py
  models/qwen4_exp/amd/model.py )
ADD=( models/qwen3_8_flash_next/hc_prefill.py
  models/qwen4_exp/common/hc_prefill.py )

rm -rf tree2 && mkdir tree2
ID=$(docker create "$NIGHTLY"); docker cp "$ID:/usr/local/lib/python3.12/dist-packages/vllm" tree2/vllm; docker rm "$ID" >/dev/null

mkdir -p m3b m3t
for f in "${MOD[@]}"; do
  n=$(echo "$f" | tr / _)
  gh api "repos/$REPO/contents/$f?ref=$THEIRS" --jq .content | base64 -d > "m3t/$n.theirs"
  gh api "repos/$REPO/contents/$f?ref=$BASE"   --jq .content | base64 -d > "m3b/$n.base"   || : > "m3b/$n.base"
  git merge-file -L ours -L base -L theirs "tree2/vllm/$f" "m3b/$n.base" "m3t/$n.theirs" \
    && echo "CLEAN $f" || echo "CONFLICT $f"
done
for f in "${ADD[@]}"; do
  mkdir -p "tree2/vllm/$(dirname "$f")"
  gh api "repos/$REPO/contents/$f?ref=$THEIRS" --jq .content | base64 -d > "tree2/vllm/$f"
done

# GDN conflict resolution (2026-09-26 ruling): nightly's fused runner already exports
# checkpoints (checkpoint_export=True); the PR's manual triton store block double-stores.
# Take OURS: delete the conflict region.
F=tree2/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py
sed -i '/<<<<<<< ours/,/>>>>>>> theirs/d' "$F"
# Stale API: nightly removed the autotune kwarg from PLE layer calls.
sed -i '/^[[:space:]]*autotune=False,$/d' tree2/vllm/models/qwen4_exp/nvidia/ple_layer.py
# GDN coalesce spec gate (heals mamba_hybrid uniform-params validator when COALESCE=1):
python3 - <<'PY'
p="tree2/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py"
s=open(p).read()
old="        if not self.prefill_checkpoint_blocks:\n            return spec"
new="        if not self.prefill_checkpoint_blocks:\n            if envs.VLLM_QWEN3_8_PREFILL_COALESCE:\n                return replace(spec, num_prefill_checkpoint_blocks=1)\n            return spec"
assert old in s and s.count(old)==1
open(p,"w").write(s.replace(old,new))
PY
grep -rl '<<<<<<<' tree2/vllm --include='*.py' && { echo "UNRESOLVED-CONFLICTS"; exit 1; }
find tree2/vllm -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null
# Verify every mutated file parses; then build: docker build -f recipes/Dockerfile.r6 (ple5 + tree2 in build ctx)
python3 - <<'PY'
import ast,sys
for f in ["envs.py","model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py","models/qwen4_exp/nvidia/ple_layer.py","models/qwen4_exp/nvidia/ple_attn.py","v1/core/sched/scheduler.py","models/qwen4_exp/common/hc_prefill.py","models/qwen3_8_flash_next/hc_prefill.py","model_executor/models/qwen3_next.py","models/qwen4_exp/amd/model.py"]:
    ast.parse(open("tree2/vllm/"+f).read())
print("TREE2-BUILD-READY")
PY
