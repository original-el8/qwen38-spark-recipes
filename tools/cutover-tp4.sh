#!/usr/bin/env bash
# Fleet cutover: stop DS4.1 tp4, start Qwen3.8 karmic tp4, gate on readiness.
# Auto-rolls back to DS4.1 if qwen38 fails to become ready. Run from stinger.
set -uo pipefail
QDIR=/home/jasonc/spark_vllm/deployments/qwen38-karmic-tp4-20260925
IMG=eugr/spark-vllm-b12x:nightly-20260925
QP=qwen38-karmic-tp4-20260925
DS=ds41-sync-2ac48a52-r2
DSROOT=/home/jasonc/spark_vllm/deployments/ds41-flash-spark-2ac48a52-b12x135c9715-sm121-r2-tp4-b128-swa64-s8-b8192-1m-jitwarn
host_of(){ case $1 in 0) echo maxwell;; 1) echo ampere;; 2) echo faraday;; 3) echo hertz;; esac; }

rollback(){
  echo "== ROLLBACK: stopping qwen ranks 0..3"
  for r in 0 1 2 3; do h=$(host_of $r)
    ssh $h "cd $QDIR && docker compose --env-file $h.env -p $QP down --timeout 30" || true
  done
  echo "== ROLLBACK: starting DS4.1 workers 1,2,3 then coordinator 0"
  for r in 1 2 3 0; do h=$(host_of $r)
    ssh $h "cd $DSROOT && docker compose -f compose.json --env-file $h.env -p $DS up -d" || echo "warn: ds41 up failed on $h"
  done
  echo "ROLLBACK-DONE (verify: docker ps on each host; ds41-$DS-0 must reach startup complete)"
}

echo "== image ID check"
ID=$(ssh maxwell "docker image inspect $IMG --format '{{.Id}}'") || { echo "ABORT: image missing on maxwell"; exit 1; }
echo "maxwell $ID"
for h in ampere faraday hertz; do
  pid=$(ssh $h "docker image inspect $IMG --format '{{.Id}}'" 2>&1)
  echo "$h $pid"
  [ "$pid" = "$ID" ] || { echo "ABORT: $h image mismatch/missing"; exit 1; }
done

echo "== staging check: deployment files present on all ranks"
for r in 0 1 2 3; do h=$(host_of $r)
  ssh $h "test -x $QDIR/serve.sh && test -f $QDIR/compose.yml && test -f $QDIR/$h.env && test -f $QDIR/config.json" \
    || { echo "ABORT: $h missing deployment files in $QDIR"; exit 1; }
done

echo "== stop DS4.1 (coordinator 0 first, then workers)"
for r in 0 1 2 3; do h=$(host_of $r)
  ssh $h "cd $DSROOT && docker compose -f compose.json --env-file $h.env -p $DS down --timeout 30" || echo "warn: ds41 down r$r on $h failed/absent"
done

echo "== start qwen tp4 workers then coordinator"
for h in ampere faraday hertz; do
  ssh $h "cd $QDIR && docker compose --env-file $h.env -p $QP up -d" || { echo "FAIL: qwen up on $h"; rollback; exit 1; }
done
sleep 5
ssh maxwell "cd $QDIR && docker compose --env-file maxwell.env -p $QP up -d" || { echo "FAIL: qwen coordinator up"; rollback; exit 1; }

echo "== wait readiness (rank0 'Application startup complete'; 55 min cap)"
READY=0
for i in $(seq 1 110); do
  if ssh maxwell "docker logs $QP-0 2>&1 | grep -q 'Application startup complete'"; then READY=1; break; fi
  if ! ssh maxwell "docker inspect -f '{{.State.Running}}' $QP-0" 2>/dev/null | grep -q true; then
    echo "FAIL: rank0 exited"; ssh maxwell "docker logs --tail 40 $QP-0"; rollback; exit 1
  fi
  sleep 30
done
if [ "$READY" != 1 ]; then
  echo "FAIL: no readiness within cap (hang?) — last 40 lines:"
  ssh maxwell "docker logs --tail 40 $QP-0"
  rollback
  exit 1
fi
echo READY
ssh maxwell "docker logs $QP-0 2>&1 | tail -5"
echo DONE-CUTOVER
