#!/usr/bin/env bash
# Deterministic serving smoke for the Qwen3.8-Flash-Next recipes.
# Usage: ENDPOINT=http://maxwell:8000/v1 ./smoke.sh [served_model_name]
set -euo pipefail
ENDPOINT="${ENDPOINT:-http://maxwell:8000/v1}"
MODEL="${1:-Qwen3.8-Flash-Next}"

req() { curl -sf --max-time "${REQ_TIMEOUT:-300}" "$ENDPOINT/chat/completions" -H 'content-type: application/json' -d "$1"; }

echo "[1/4] model listing"
curl -sf --max-time 30 "$ENDPOINT/models" | python3 -c "
import json,sys
names={m['id'] for m in json.load(sys.stdin)['data']}
assert '$MODEL' in names, names
print('  ok:', sorted(names))"

echo "[2/4] arithmetic (temperature 0, no thinking)"
req "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 17*19? Answer with only the number.\"}],\"temperature\":0,\"max_tokens\":64,\"chat_template_kwargs\":{\"enable_thinking\":false}}" | python3 -c "
import json,sys
r=json.load(sys.stdin)
c=r['choices'][0]['message']['content']
assert '323' in c, c
assert r['choices'][0]['finish_reason'] in ('stop','length'), r['choices'][0]
print('  ok:', c[:40])"

echo "[3/4] tool call via qwen3_xml parser"
req "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"What is the weather in Oslo? Use the tool.\"}],\"temperature\":0,\"max_tokens\":128,\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"description\":\"Get current weather for a city\",\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}},\"required\":[\"city\"]}}}]}" | python3 -c "
import json,sys
r=json.load(sys.stdin)
m=r['choices'][0]['message']
tc=m.get('tool_calls') or []
assert tc and tc[0]['function']['name']=='get_weather', m
args=json.loads(tc[0]['function']['arguments'])
assert 'oslo' in json.dumps(args).lower(), args
print('  ok:', tc[0]['function']['name'], args)"

echo "[4/4] tool argument types via native schema"
req "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Book 3 tickets priced 42 each; compute total.\"}],\"temperature\":0,\"max_tokens\":160,\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"book\",\"description\":\"Book tickets\",\"parameters\":{\"type\":\"object\",\"properties\":{\"count\":{\"type\":\"integer\"},\"unit_price\":{\"type\":\"number\"}},\"required\":[\"count\",\"unit_price\"]}}}]}" | python3 -c "
import json,sys
r=json.load(sys.stdin)
m=r['choices'][0]['message']
tc=(m.get('tool_calls') or [{}])[0]
args=json.loads(tc.get('function',{}).get('arguments') or '{}')
assert args.get('count')==3 and args.get('unit_price')==42, args
print('  ok:', args)"

# [5/5] image input (needs --limit-mm-per-prompt image>=1); loose functional gate
printf '{"model":"%s","messages":[{"role":"user","content":[{"type":"text","text":"What single color dominates this image? Answer with the color name."},{"type":"image_url","image_url":{"url":"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="}}]}],"max_tokens":200,"temperature":0}' \
  | curl -sf -m 90 "$ENDPOINT/chat/completions" -H 'content-type: application/json' -d @- \
  | python3 -c "import json,sys; m=json.load(sys.stdin)['choices'][0]['message']; c=((m.get('content') or '')+' '+(m.get('reasoning') or '')).strip(); assert c, 'empty vision reply (content+reasoning)'; print('  ok: vision reply:', c[:80])" \
  || { echo "FAIL: image input step"; exit 1; }
# video LIMIT gate is startup-level: boot fails if video modality unsupported; deep
# video functionality is checked manually (engine accepts --limit-mm-per-prompt video:2).

echo "SMOKE PASS"
