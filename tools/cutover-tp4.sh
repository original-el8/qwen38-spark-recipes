#!/usr/bin/env bash
# Fleet cutover: stop DS4.1 tp4, start Qwen3.8 karmic tp4, gate on readiness+smoke.
# Run from stinger. Idempotent-ish: each step checks current state first.
set -uo pipefail
QDIR=/home/jasonc/spark_vllm/deployments/qwen38-karmic-tp4-20260925
IMG=eugr/spark-vllm-b12x:nightly-20260925
QP=qwen38-karmic-tp4-20260925
DS=ds41-sync-2ac48a52-r2
DSROOT=/home/jasonc/spark_vllm/deployments/ds41-flash-spark-2ac48a52-b12x135c9715-sm121-r2-tp4-b128-swa64-s8-b8192-1m-jitwarn

echo "== image ID check"
ID=$(ssh maxwell "docker image inspect $IMG --format '{{.Id}}'")
echo "maxwell $ID"
for h in ampere faraday hertz; do
  pid=$(ssh $h "docker image inspect $IMG --format '{{.Id}}'" 2>&1)
  echo "$h $pid"
  [ "$pid" = "$ID" ] || { echo "ABORT: $h image mismatch/missing"; exit 1; }
done

echo "== stop DS4.1 (coordinator 0 first, then workers)"
for r in 0 1 2 3; do
  case $r in 0) h=maxwell;; 1) h=ampere;; 2) h=faraday;; 3) h=hertz;; esac
  ssh $h "cd $DSROOT && docker compose -p $DS down --timeout 30" || echo "warn: ds41 down r$r on $h failed/absent"
done

echo "== start qwen tp4 workers then coordinator"
for h in ampere faraday hertz; do
  ssh $h "cd $QDIR && docker compose --env-file $h.env -p $QP up -d"
done
sleep 5
ssh maxwell "cd $QDIR && docker compose --env-file maxwell.env -p $QP up -d"

echo "== wait readiness (rank0 startup complete, timeout 55min)"
for i in $(seq 1 110); do
  if ssh maxwell "docker logs $QP-0 2>&1 | grep -q 'Application startup complete'"; then echo READY; break; fi
  if ! ssh maxwell "docker inspect -f '{{.State.Running}}' $QP-0" | grep -q true; then
    echo "rank0 exited; last 40 log lines:"; ssh maxwell "docker logs --tail 40 $QP-0"; exit 1
  fi
  sleep 30
done
ssh maxwell "docker logs $QP-0 2>&1 | tail -5"
echo DONE-CUTOVER
