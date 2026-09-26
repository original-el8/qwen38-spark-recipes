# Image provenance and how to reproduce it

Two image paths serve this model. The published recipes pin **Path A** (stock hub base) as the
default because the goal is reproducibility from public inputs; **Path B** is the fleet's
campaign-built overlay lineage that produced the historical measurements. The performance
comparison between them is in the top-level README.

## Path A — stock `eugr/spark-vllm-b12x` nightly (default)

`eugr` (Eugene Rakhmatulin, NVIDIA) publishes nightly Spark images on Docker Hub, built by
[eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker):

- `eugr/spark-vllm-b12x:latest` / `eugr/spark-vllm-b12x:nightly-YYYYMMDD`, `linux/arm64` only.
- Nightly source pair: vLLM [`local-inference-lab/vllm@dev/karmic-kraken`](https://github.com/local-inference-lab/vllm/tree/dev/karmic-kraken)
  + [`local-inference-lab/b12x@master`](https://github.com/local-inference-lab/b12x) (PyTorch 2.13.0, CUTLASS DSL 4.7.0 pinned by the build repo).
- The b12x source commit of a given nightly is recorded **inside the image** at
  `/workspace/b12x-source-commit` (tags carry no source identity), so always record both the
  tag and the digest after `docker pull`.
- `dev/karmic-kraken` is the successor lineage of `dev/jovian-judgement` (absorbed by PR #784,
  2026-09-14) and carries the Qwen3.8 Flash Next model (renamed `vllm/models/qwen4_exp/`),
  RoCE (RoCEnante) cross-node collectives, HC TP-sharding (`VLLM_QWEN3_8_FLASH_NEXT_HC_TP`),
  MTP, PLE, QSA, and the b12x linear/MoE/GDN wiring. b12x `master` includes every merged
  performance PR referenced below (QSA stable selection #394, NVFP4 FC1 padding #389, E4M3
  subnormals #388, RoCEnante #295/#315/#383).

Pin used by these recipes:

```text
eugr/spark-vllm-b12x:nightly-20260925   hub manifest digest sha256:c9e22735a5bb701c64b04ff9e08779d75d90abc045523fe0adf87b6092457c98
```

After pulling on every host, require the same **image ID** everywhere
(`docker image inspect -f '{{.Id}}'`); a tag match is not proof. Record
`docker run --rm --entrypoint cat <image> /workspace/b12x-source-commit`.

## Path B — fleet overlay images (campaign-built, historical measurements)

The 2026-09-16/17 optimization campaign measured on images built as **CPU-only source
overlays** on a pinned parent chain (never from a hub pull at build time):

```text
docker.io vllm/vllm-openai:glm53-flash @sha256:2c6da6c6f16ed15c91e412d896dba13701f25fe1861eaec9ddaa4db34d1d21c4
  └─ spark-vllm:glm53-flash-sm121              sha256:9b31ea01ffdf038e856b32bf9260f2f917c8e106c2dfa8b485f92f14ac8ef59c   (patch scripts from eugr/spark-vllm-docker @e9cf3596)
      └─ … ds41 lineage …
          └─ spark-vllm:ds41-flash-spark-2ac48a52-b12x135c9715-sm121-r2   sha256:13e30857576a76027b0abe3254a83c482e6db690548b39c3278fa389bc8db154   (overlay parent)
              ├─ spark-vllm:qwen38-padding-76061de4-b12x72baebbd-sm121-image-r1   sha256:a2fbec2f2e348631fad52413c61a1f72ed4df62c6d31c13b5cd6aa90a3540f7c   (native-160 NVFP4 widths; BF16-KV and FP8-KV baselines)
              └─ spark-vllm:qwen38-qsa-selection-76061de4-b12xd2d5368d-sm121-image-r1   sha256:953b00ee516cac5f295bfb26863d3cfdcb5f950a6461bed76e11a84281468346   (QSA stable-selection fusion = b12x PR #394 head)
```

Overlay inputs (clean `git archive` tars, never a dirty worktree):

| image | vLLM source | b12x source |
|---|---|---|
| `qwen38-padding-*` | `local-inference-lab/vllm@76061de4bff2adc741cb25018ca79991263228be` (branch `codex/qwen-moe-padding-20260917`, tree `1f6d00d6db3f9a0df733a731c6197a3c3774f568`, tar sha256 `a37c84074dbf68b490f1b22b96ff181f5cbc7d90a8547340322233d8e7ebae34`) | `b12x@72baebbda2a200762f37223ef25aa64e8a5cb734` (branch `codex/qwen-moe-decode-20260916`, tree `1afd2fe0ed0f1405dd513d52350fe68b9c150ad3`, tar sha256 `7cc4fb6b507eb99068092e59f331f78998a66a03ffcf8806d0f8e4b083bb61c7`) |
| `qwen38-qsa-selection-*` | same `76061de4…` tar | `b12x@d2d5368d6c8a5cc43d79bfc8a4a5c58db584fc47` (branch `codex/qwen-qsa-selection-20260917`, tree `f1db9cabc8f362134bc90fdf63b9401e545c47ff`, tar sha256 `99ae430493b984e61310ac291d7bb733aa6562b178e06d02a48122bd0ef4b891`) |

The Dockerfile and builder used are in this directory verbatim:
[`Dockerfile`](Dockerfile), [`build-image.py`](build-image.py),
[`install-candidate.py`](install-candidate.py). Mechanics:

1. Both worktrees must be clean; sources ship as `git archive` tars whose sha256 is recorded in
   `image-manifest.json`. Native-compatibility gate: `git diff <parent rev> HEAD -- csrc cmake
   requirements` must be **empty**, so the parent's compiled `.so` files stay valid.
2. `docker build --pull=false --network=none` on the builder (maxwell, native arm64/sm121):
   rename the parent's `vllm/`+`b12x/` trees, unpack the two source tars, copy every parent
   `.so`, `vllm/vllm_flash_attn`, `vllm/third_party/flashmla` into the new tree, synthesize
   `_version.py`, create a fresh `uv` venv (`--system-site-packages`), and run
   `install-candidate.py`, which fails the build unless `vllm`, `b12x`,
   `b12x.comm.roce._proxy`, `b12x.loader._native`, and
   `vllm.models.qwen3_8_flash_next` import. OCI labels record profile id, base id, commits and
   trees; `/opt/spark-vllm/image-manifest.json` mirrors it.
3. Immutable tags: a rebuild that produces different bytes must get a new tag.
4. Distribution to peers is `docker save | zstd -T4 -1` → rsync → `docker load`, with image-ID,
   label, and in-image-manifest verification on every host. (~24.4 GiB → ~10.6 GiB compressed,
   ≈200 s to three peers over 10 GbE.)

Why overlays exist: the campaign's vLLM branch (`76061de4`, HC token-ownership prefill, PLE
checkpoint export, MoE shard alignment — 10 commits on `dev/jovian-judgement@59fbf050`) was
measured before that work reached `dev/karmic-kraken`, and the fleet images must not recompile
kernels per host.

## Open questions tracked by the fleet (not blockers)

- The b12x beta line (`integration/karmic-kraken-beta`) leads `master` on PCIe/world-size-3
  work; nightlies track `master` only.
