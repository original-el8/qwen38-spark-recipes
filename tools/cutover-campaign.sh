#!/usr/bin/env bash
# Persistent cutover: stock karmic TP4 -> campaign overlay TP4 (Path B) as the serving config.
# Rollback to stock on any failure; to reverse permanently later: compose-up the karmic dir.
set -uo pipefail
CDIR=/home/jasonc/spark_vllm/deployments/qwen38-karmic-tp4-20260925
CP=qwen38-karmic-tp4-20260925
DDIR=/home/jasonc/spark_vllm/deployments/qwen38-qsa-selection-20260917
CPROJ=qwen38-qsa-selection-20260917
HOSTS="maxwell ampere faraday hertz"

restore_stock() {
  echo "== ROLLBACK: stop campaign, restore stock TP4"
  for h in $HOSTS; do ssh $h "cd $DDIR && docker compose --env-file $h.env -p $CPROJ down --timeout 30" >/dev/null 2>&1; done
  for r in 1 2 3 0; do
    h=$( [ $r = 0 ] && echo maxwell || { [ $r = 1 ] && echo ampere || { [ $r = 2 ] && echo faraday || echo hertz; }; } )
    ssh $h "cd $CDIR && docker compose -p $CP up -d" >/dev/null 2>&1
    [ $r = 0 ] || sleep 20
  done
  echo "STOCK-RESTORED (API check below)"
}

echo "== stop stock TP4 (rank0 first, then workers)"
ssh maxwell "cd $CDIR && docker compose -p $CP stop spark-0" >/dev/null
for r in 1 2 3; do
  h=$( [ $r = 1 ] && echo ampere || { [ $r = 2 ] && echo faraday || echo hertz; } )
  ssh $h "cd $CDIR && docker compose -p $CP stop spark-$r" >/dev/null
done

echo "== start campaign overlay: workers (ampere,faraday,hertz) then coordinator (maxwell)"
for h in ampere faraday hertz; do
  ssh $h "cd $DDIR && docker compose --env-file $h.env -p $CPROJ up -d" >/dev/null || { echo "FAIL: $h"; restore_stock; exit 1; }
  sleep 20
done
ssh maxwell "cd $DDIR && docker compose --env-file maxwell.env -p $CPROJ up -d" >/dev/null || { echo "FAIL: maxwell"; restore_stock; exit 1; }

echo "== wait live API readiness (maxwell:8000)"
READY=0
for i in $(seq 1 110); do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://maxwell:8000/v1/models || echo 000)
  [ "$code" = 200 ] && { READY=1; break; }
  # early exit if coordinator died
  st=$(ssh maxwell "docker inspect -f '{{.State.Status}}' $CPROJ-0 2>/dev/null || echo missing")
  case "$st" in exited|dead|missing) echo "FAIL: coordinator $st"; ssh maxwell "docker logs --tail 40 $CPROJ-0"; restore_stock; exit 1;; esac
  sleep 30
done
[ "$READY" = 1 ] || { echo "FAIL: readiness cap"; ssh maxwell "docker logs --tail 40 $CPROJ-0"; restore_stock; exit 1; }

echo "== wait cudagraph capture completion (post-200 capture runs minutes; probing pre-capture measures eager path)"
CAP=0
for i in $(seq 1 60); do
  ssh maxwell "docker logs $CPROJ-0 2>&1 | grep -qiE 'Graph capturing finished|graph capture.*done|Capturing CUDA graphs.*finished'" && { CAP=1; break; }
  sleep 15
done
[ "$CAP" = 1 ] || echo "WARN: capture-complete line not seen within 15 min; proceeding (warmup probe will absorb it)"

echo "== warmup (throwaway prefill+decode to fully JIT/MTP-warm the engine)"
python3 - <<WARM
import json, urllib.request, time
req = urllib.request.Request("http://maxwell:8000/v1/completions",
  data=json.dumps({"model":"Qwen3.8-Flash-Next","prompt":[i*2654435761 % 150000 + 1000 for i in range(4096)],
                   "max_tokens":64,"temperature":0.0}).encode(),
  headers={"content-type":"application/json"})
t0=time.monotonic(); json.load(urllib.request.urlopen(req, timeout=900))
print("warmup ok in %.1fs" % (time.monotonic()-t0))
WARM

echo "== smoke"
# smoke.sh takes the served name as $1 (not the endpoint): positional here = model name
if ! ENDPOINT=http://maxwell:8000/v1 bash /tmp/spark-vllm-recipes/tools/smoke.sh; then
  echo "FAIL: smoke"; restore_stock; exit 1
fi

echo "== prefill verification (expect > 4,400 tok/s)"
python3 /tmp/spark-vllm-recipes/tools/bench-quick.py http://maxwell:8000/v1 Qwen3.8-Flash-Next 3 \
  | tee /tmp/campaign-cutover-verify.json | python3 -c 'import json,sys; d=json.load(sys.stdin); p=d["prefill-8192"]["mean"]; sys.exit(0 if p>4400 else 2)' \
  || { echo "FAIL: prefill below 4,400"; restore_stock; exit 1; }

echo "CAMPAIGN-CUTOVER-READY (Path B serving; stock stopped-but-intact in $CDIR)"
