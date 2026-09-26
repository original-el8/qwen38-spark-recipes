#!/usr/bin/env bash
# One lever cycle on the TP4 staging deploy: gated restart with optional extra compose
# overlays, then the fast llm_decode_bench readout (C1 decode ctx0/64k + prefill scale).
# usage: ldb-fast-matrix.sh <arm-name> [staging-dir] [extra-overlay-file...]
# Readout lands in $LDB_DIR/benchmark_results.json + LLM's own results/ file; the table
# on stdout carries tok/s, TTFT, and MTP-normalized steps/s (accept len) rows.
#
# Hard-won guarantees (each clause exists because its absence produced a measured-on-the-
# WRONG-container result, 2026-09-26): we remove EVERY compose container on every rank —
# not just $PROJ — because a stale stack holding :8000 silently answers the bench while
# the arm's containers crash-loop on bind; the gate checks the TCP API is actually closed
# (`ss -xln` unix-socket grepping never blocks); and overlay files are shipped to every
# rank because compose resolves -f locally per host.
set -u
NAME=$1; shift
STG=${1:-/home/jasonc/spark_vllm/deployments/tp4-levers-20260926}; shift || true
PROJ=$(basename "$STG")
LDB_DIR=${LDB_DIR:-/home/jasonc/spark_vllm/llm-inference-bench}
API=http://maxwell:8000/v1
FL=""
for f in "$@"; do FL="$FL -f $(basename "$f")"; done

echo "== [$NAME] removing ALL compose containers on every rank"
for h in maxwell ampere faraday hertz; do
  ssh "$h" 'docker ps -aq --filter label=com.docker.compose.project | xargs -r docker rm -f' >/dev/null 2>&1
done

echo "== [$NAME] port gate: :8000 must actually close on every rank"
for h in maxwell ampere faraday hertz; do
  for _ in $(seq 1 40); do
    curl -sf -m 2 "http://$h:8000/v1/models" >/dev/null 2>&1 || break
    sleep 3
  done
  curl -sf -m 2 "http://$h:8000/v1/models" >/dev/null 2>&1 && { echo "FAIL-PORT-HELD:$h"; exit 1; }
done

echo "== [$NAME] ship overlays to every rank (compose reads -f locally)"
for f in "$@"; do
  scp -3 -q "maxwell:$STG/$(basename "$f")" "ampere:$STG/" "faraday:$STG/" "hertz:$STG/" 2>/dev/null || true
done

echo "== [$NAME] up $FL"
for h in ampere faraday hertz; do
  ssh "$h" "cd $STG && docker compose --env-file $h.env -p $PROJ -f compose.yml $FL up -d" >/dev/null
  sleep 15
done
ssh maxwell "cd $STG && docker compose --env-file maxwell.env -p $PROJ -f compose.yml $FL up -d" >/dev/null
READY=0
for _ in $(seq 1 60); do
  code=$(curl -s -o /dev/null -w '%{http_code}' -m 5 "$API/models" || echo 000)
  [ "$code" = 200 ] && { READY=1; break; }
  sleep 20
done
[ "$READY" = 1 ] || { echo "FAIL-READY-$NAME"; exit 1; }
# Serve-identity gate: the answering server must be OUR arm, not a ghost.
echo "== [$NAME] serving container on :8000:"
ssh maxwell 'docker ps --filter publish=8000 --format "{{.Names}} {{.Image}}"' | tee /dev/stderr | grep -q "$PROJ" || { echo "FAIL-IDENTITY:$NAME"; exit 1; }

echo "== [$NAME] host state (thermal/clock attribution protocol)"
{ for h in maxwell ampere faraday hertz; do
    printf "%s: " "$h"
    ssh "$h" "nvidia-smi --query-gpu=temperature.gpu,power.draw,clocks.sm --format=csv,noheader | tr '\n' ' '; uptime -p" 2>/dev/null
  done; } | tee -a /tmp/ldb-hoststate.log

echo "== [$NAME] LDB fast: C1 decode ctx0/64K + full prefill scale"
cd "$LDB_DIR" && python3 llm_decode_bench.py --host maxwell --port 8000 \
  --model Qwen3.8-Flash-Next --standalone-prefill \
  --prefill-contexts 8k,16k,32k,64k,128k --concurrency 1 --contexts 0,65536
echo "== [$NAME] done, arm left serving"
