#!/usr/bin/env bash
# A/B arm: start the campaign overlay profile (its own recipe dir, image, cache namespaces),
# probe it with tools/bench-quick.py, then restore the stock karmic TP4 as fleet state.
set -uo pipefail
QDIR=/home/jasonc/spark_vllm/deployments/qwen38-karmic-tp4-20260925
QP=qwen38-karmic-tp4-20260925
CDIR=/home/jasonc/spark_vllm/deployments/qwen38-qsa-selection-20260917
CP=qwen38-qsa-selection-20260917
host_of(){ case $1 in 0) echo maxwell;; 1) echo ampere;; 2) echo faraday;; 3) echo hertz;; esac; }

restore_stock(){
  echo "== restore stock karmic TP4 (workers then coordinator)"
  for r in 1 2 3 0; do h=$(host_of $r)
    ssh $h "cd $QDIR && docker compose --env-file $h.env -p $QP up -d" || echo "CRITICAL: stock up failed on $h"
  done
  for i in $(seq 1 60); do
    ssh maxwell "docker logs $QP-0 2>&1 | grep -q 'Application startup complete'" && { echo STOCK-RESTORED-READY; return 0; }
    sleep 30
  done
  echo "CRITICAL: stock restore cap hit"; return 1
}

echo "== stop stock TP4 (0 first)"
for r in 0 1 2 3; do h=$(host_of $r)
  ssh $h "cd $QDIR && docker compose --env-file $h.env -p $QP down --timeout 30" || echo "warn: stock down $h"
done

echo "== start campaign profile (campaign serve.sh/env own the flags)"
for r in 1 2 3 0; do h=$(host_of $r)
  ssh $h "cd $CDIR && docker compose --env-file $h.env -p $CP up -d" || { echo "FAIL campaign up $h"; restore_stock; exit 1; }
done

echo "== wait readiness (campaign coordinator container)"
C0=$(ssh maxwell "cd $CDIR && docker compose -p $CP ps --format '{{.Name}}' | grep -E -- '-0\$' | head -1")
[ -n "$C0" ] || { echo 'FAIL: coordinator container not resolvable'; restore_stock; exit 1; }
echo "coordinator container: $C0"
READY=0
START_TS=$(date -u +%s)
for i in $(seq 1 110); do
  # readiness = live API, not log text: reused containers carry stale startup lines
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://maxwell:8000/v1/models || echo 000)
  [ "$code" = 200 ] && { READY=1; break; }
  ssh maxwell "docker inspect -f '{{.State.Running}}' $C0" 2>/dev/null | grep -q true || { echo "FAIL campaign rank0 exited"; ssh maxwell "docker logs --tail 40 $C0"; restore_stock; exit 1; }
  sleep 30
done
[ "$READY" = 1 ] || { echo "FAIL campaign readiness cap"; ssh maxwell "docker logs --tail 40 $C0"; restore_stock; exit 1; }
echo CAMPAIGN-READY

bash /tmp/spark-vllm-recipes/tools/smoke.sh
python3 /tmp/spark-vllm-recipes/tools/bench-quick.py http://maxwell:8000/v1 Qwen3.8-Flash-Next 3 | tee /tmp/tp4-campaign-probe.json

echo "== stop campaign, restore stock"
for r in 0 1 2 3; do h=$(host_of $r)
  ssh $h "cd $CDIR && docker compose --env-file $h.env -p $CP down --timeout 30" || echo "warn: campaign down $h"
done
restore_stock
