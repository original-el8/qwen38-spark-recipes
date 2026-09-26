#!/usr/bin/env bash
# Build an image variant on ALL FOUR ranks and prove the code is identical everywhere.
#
# WHY THIS EXISTS: three times this session an image or deploy dir was built/created on
# one host only, then a 4-rank test ran against 3 missing workers (Docker tried to *pull*
# the local tag, failed, and the cycle burned ~10 minutes at ready=000). Never build ad hoc.
#
# Usage:
#   tools/build-fleet.sh <dockerfile-name> <image-tag> [src-dir]
#     dockerfile-name  e.g. Dockerfile.r6e          (lives in src-dir)
#     image-tag        e.g. spark-vllm:qwen38-r6e-hcprobe-r1
#     src-dir          build context on the ranks (default /home/jasonc/spark_vllm/r4ctx)
#
# Assumes the build context (tree2/, ple5/, verify*.py, Dockerfile.*) lives in src-dir on
# RANK 0 (maxwell) — that host is the canonical source, matching ldb-fast-matrix.sh's ship rule.
set -euo pipefail

DF=${1:?usage: build-fleet.sh <dockerfile-name> <image-tag> [src-dir]}
TAG=${2:?usage: build-fleet.sh <dockerfile-name> <image-tag> [src-dir]}
SRC=${3:-/home/jasonc/spark_vllm/r4ctx}
R0=${RANK0:-maxwell}
RANKS=(maxwell ampere faraday hertz)

# Files that make up the build context. Keep this in sync with the Dockerfile's COPY lines.
CTX=(tree2 ple5)
for f in "$DF" verify6.py verify7.py hc_probe_patch.py; do
  ssh "$R0" "test -f $SRC/$f" 2>/dev/null && CTX+=("$f")
done

echo "== building $TAG on all ranks from $SRC (ctx: ${CTX[*]}) =="

build_remote() {
  cat > /tmp/_build_one.sh <<EOF
set -e
mkdir -p $SRC
tar -C $SRC -xf -
cd $SRC
docker build -q -f $DF -t $TAG . > /dev/null
echo "\$(hostname)-BUILT"
EOF
}

build_remote
for h in ampere faraday hertz; do scp -q /tmp/_build_one.sh "$h:/tmp/_build_one.sh"; done

# rank 0 builds locally from its own context
ssh "$R0" "cd $SRC && docker build -q -f $DF -t $TAG . >/dev/null && echo $R0-BUILT"

# workers receive the context over the RANK0->worker path (same as ldb-fast-matrix.sh's ship)
for h in ampere faraday hertz; do
  ( ssh "$R0" "tar -C $SRC -cf - ${CTX[*]}" | ssh "$h" 'bash /tmp/_build_one.sh' ) &
done
wait

# Content identity: image IDs differ per host (build timestamps live in the config hash), so
# ID equality proves nothing. Hash the shipped python trees instead and require equality.
echo "== content identity =="
mapfile -t HASHES < <(for h in "${RANKS[@]}"; do
  ssh "$h" "docker run --rm --entrypoint bash $TAG -c 'find /usr/local/lib/python3.12/dist-packages/vllm/models /usr/local/lib/python3.12/dist-packages/vllm/model_executor /usr/local/lib/python3.12/dist-packages/vllm/v1 /usr/local/lib/python3.12/dist-packages/vllm/envs.py /usr/local/lib/python3.12/dist-packages/b12x/sequence/ple -name \"*.py\" -exec sha256sum {} + | sort | sha256sum'" 2>/dev/null | cut -c1-16
done | sort -u)
printf '   %s\n' "${HASHES[@]}"
if [ "${#HASHES[@]}" != 1 ]; then
  echo "FAIL-CONTENT-DIVERGENCE: ranks carry different code; do not serve this tag"; exit 1
fi
echo "OK $TAG identical on ${#RANKS[@]} ranks (tree ${HASHES[0]})"
