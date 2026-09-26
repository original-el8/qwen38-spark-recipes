#!/usr/bin/env bash
# TP2 stock-base validation: stop TP4, bring up TP2 (maxwell+ampere), gate, smoke+probe,
# then RESTORE TP4 as the final fleet state. Auto-reverts to TP2-up (no TP4) only if the
# TP2 gate itself fails, so the failure state is inspectable; restoring TP4 after a TP2
# failure is a manual `docker compose -p $QP up` per README order.
set -uo pipefail
Q4=/home/jasonc/spark_vllm/deployments/qwen38-karmic-tp4-20260925
QP=qwen38-karmic-tp4-20260925
Q2=/home/jasonc/spark_vllm/deployments/qwen38-karmic-tp2-20260925
T2=qwen38-karmic-tp2-20260925
host_of(){ case $1 in 0) echo maxwell;; 1) echo ampere;; 2) echo faraday;; 3) echo hertz;; esac; }

stop_tp2(){
  for r in 0 1; do h=$(host_of $r)
    ssh $h "cd $Q2 && docker compose --env-file $h.env -p $T2 down --timeout 30" || echo "warn: tp2 down $h"
  done
}
restore_tp4(){
  stop_tp2
  echo "== RESTORE TP4 (workers 1,2,3 then 0)"
  for h in ampere faraday hertz; do ssh $h "cd $Q4 && docker compose --env-file $h.env -p $QP up -d" || echo "warn: tp4 up on $h failed"; done
  sleep 5
  ssh maxwell "cd $Q4 && docker compose --env-file maxwell.env -p $QP up -d" || { echo "CRITICAL: tp4 coordinator up failed"; return 1; }
  for i in $(seq 1 110); do
    ssh maxwell "docker logs $QP-0 2>&1 | grep -q 'Application startup complete'" && { echo TP4-RESTORED-READY; return 0; }
    sleep 30
  done
  echo "CRITICAL: TP4 restore readiness cap hit"; return 1
}

echo "== stop TP4 (0 first, then workers)"
for r in 0 1 2 3; do h=$(host_of $r)
  ssh $h "cd $Q4 && docker compose --env-file $h.env -p $QP down --timeout 30" || echo "warn: tp4 down on $h"
done

echo "== start TP2: ampere (rank1) then maxwell (rank0)"
ssh ampere  "cd $Q2 && docker compose --env-file ampere.env  -p $T2 up -d" || { echo "FAIL tp2 worker"; restore_tp4; exit 1; }
sleep 3
ssh maxwell "cd $Q2 && docker compose --env-file maxwell.env -p $T2 up -d" || { echo "FAIL tp2 coordinator"; restore_tp4; exit 1; }

echo "== wait TP2 readiness"
READY=0
for i in $(seq 1 110); do
  ssh maxwell "docker logs $T2-0 2>&1 | grep -q 'Application startup complete'" && { READY=1; break; }
  ssh maxwell "docker inspect -f '{{.State.Running}}' $T2-0" 2>/dev/null | grep -q true || { echo "FAIL: tp2 rank0 exited"; ssh maxwell "docker logs --tail 40 $T2-0"; restore_tp4; exit 1; }
  sleep 30
done
[ "$READY" = 1 ] || { echo "FAIL: tp2 readiness cap"; ssh maxwell "docker logs --tail 40 $T2-0"; restore_tp4; exit 1; }
echo TP2-READY

echo "== smoke + quick probe (cells prefixed tp2-stock-karmic-nightly)"
bash /tmp/spark-vllm-recipes/tools/smoke.sh
python3 /tmp/spark-vllm-recipes/tools/bench-quick.py http://maxwell:8000/v1 Qwen3.8-Flash-Next 3 \
  | tee /tmp/tp2-stock-probe.json

echo "== restore TP4 fleet state"
restore_tp4
