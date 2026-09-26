#!/usr/bin/env bash
# Validate Path B (campaign overlay) at TP2 topology, then restore TP4 overlay serving.
# Rollback target on ANY failure = TP4 overlay (the serving config), never leave half-state.
set -uo pipefail
TP2DIR=/home/jasonc/spark_vllm/deployments/qwen38-qsa-selection-tp2-20260917
TP2PROJ=qwen38-qsa-selection-tp2-20260917
TP4DIR=/home/jasonc/spark_vllm/deployments/qwen38-qsa-selection-20260917
TP4PROJ=qwen38-qsa-selection-20260917
RECIPE=/tmp/spark-vllm-recipes

restore_tp4() {
  echo "== RESTORE TP4 overlay serving"
  for h in maxwell ampere; do   # tp2 profile only has state on the pair
    ssh $h "cd $TP2DIR && docker compose --env-file $h.env -p $TP2PROJ down --timeout 30" >/dev/null 2>&1
  done
  for h in maxwell ampere faraday hertz; do
    for i in $(seq 1 20); do ssh $h "ss -xln 2>/dev/null | grep -q ':8000 '" || break; sleep 3; done
  done
  for h in ampere faraday hertz; do
    ssh $h "cd $TP4DIR && docker compose --env-file $h.env -p $TP4PROJ up -d" >/dev/null; sleep 15
  done
  ssh maxwell "cd $TP4DIR && docker compose --env-file maxwell.env -p $TP4PROJ up -d" >/dev/null
  for i in $(seq 1 60); do
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://maxwell:8000/v1/models || echo 000)
    [ "$code" = 200 ] && { echo TP4-RESTORED-READY; return 0; }
    sleep 30
  done
  echo "CRITICAL: TP4 restore did not become ready"; return 1
}

echo "== stop TP4 overlay (name-safe down, all hosts) + wait port release"
for h in maxwell ampere faraday hertz; do
  ssh $h "cd $TP4DIR && docker compose -p $TP4PROJ down --timeout 30" >/dev/null || echo "warn: tp4 down $h"
done
for h in maxwell ampere faraday hertz; do
  for i in $(seq 1 40); do ssh $h "ss -xln 2>/dev/null | grep -q ':8000 '" || { echo "$h port free"; break; }; sleep 3; done
done

echo "== start TP2 overlay: ampere (rank1) then maxwell (rank0)"
ssh ampere "cd $TP2DIR && docker compose --env-file ampere.env -p $TP2PROJ up -d" >/dev/null || { echo "FAIL: ampere"; restore_tp4; exit 1; }
sleep 20
ssh maxwell "cd $TP2DIR && docker compose --env-file maxwell.env -p $TP2PROJ up -d" >/dev/null || { echo "FAIL: maxwell"; restore_tp4; exit 1; }

echo "== wait live API readiness"
READY=0
for i in $(seq 1 60); do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://maxwell:8000/v1/models || echo 000)
  [ "$code" = 200 ] && { READY=1; break; }
  st=$(ssh maxwell "docker inspect -f '{{.State.Status}}' $TP2PROJ-0 2>/dev/null || echo missing")
  case "$st" in exited|dead|missing) echo "FAIL: tp2 rank0 $st"; ssh maxwell "docker logs --tail 40 $TP2PROJ-0"; restore_tp4; exit 1;; esac
  sleep 30
done
[ "$READY" = 1 ] || { echo "FAIL: readiness cap"; ssh maxwell "docker logs --tail 40 $TP2PROJ-0"; restore_tp4; exit 1; }

echo "== prove we're on the overlay, not a ghost: manifest + RoCEnante markers on :8000 owner"
C0=$(ssh maxwell "cd $TP2DIR && docker compose -p $TP2PROJ ps --format '{{.Name}}' | grep -E -- '-0$' | head -1")
ssh maxwell "docker exec $C0 cat /opt/spark-vllm/image-manifest.json 2>/dev/null | grep -o 'qwen38-qsa-selection-76061de4-b12xd2d5368d-sm121-image-r1' | head -1" | grep -q . && echo "manifest: overlay OK" || { echo "FAIL: not the overlay image"; restore_tp4; exit 1; }
ssh maxwell "docker logs $C0 2>&1 | grep -acE \"Using \['B12X_ROCENANTE', 'PYNCCL'\] all-reduce backends\"" | grep -qv '^0$' && echo "RoCEnante: live" || echo "WARN: RoCEnante marker not seen (pair path may differ)"

echo "== warmup + smoke + bench"
python3 - <<'WARM'
import json, urllib.request
req = urllib.request.Request("http://maxwell:8000/v1/completions",
  data=json.dumps({"model":"Qwen3.8-Flash-Next","prompt":[i*2654435761 % 150000 + 1000 for i in range(2048)],
                   "max_tokens":32,"temperature":0.0}).encode(), headers={"content-type":"application/json"})
json.load(urllib.request.urlopen(req, timeout=900)); print("warmup ok")
WARM
ENDPOINT=http://maxwell:8000/v1 bash $RECIPE/tools/smoke.sh || { echo "FAIL: smoke"; restore_tp4; exit 1; }
python3 $RECIPE/tools/bench-quick.py http://maxwell:8000/v1 Qwen3.8-Flash-Next 3 | tee /tmp/tp2-overlay-confirmed.json

echo "== memory floors"
for h in maxwell ampere; do ssh $h "free -g" | awk -v h=$h 'NR==2{print h" MemAvailable:",$7,"GiB"}'; done

echo "== restore TP4 serving"
restore_tp4
