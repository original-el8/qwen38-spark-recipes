"""Build paired vLLM and B12X source archives on Maxwell without GPU execution."""

import argparse
import hashlib
import json
import shlex
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent.parent
SOURCE = WORKSPACE / "worktrees/qwen38-moe-padding-20260917"
B12X_SOURCE = WORKSPACE / "worktrees/b12x-qwen-qsa-selection-20260917"
PARENT = (
    WORKSPACE
    / "images/ds41-flash-spark-2ac48a52-b12x135c9715-sm121-r2/image-manifest.json"
)
BASE = "sha256:13e30857576a76027b0abe3254a83c482e6db690548b39c3278fa389bc8db154"
BASE_TAG = "spark-vllm:ds41-flash-spark-2ac48a52-b12x135c9715-sm121-r2"


def run(args, **kwargs):
    return subprocess.run([str(a) for a in args], check=True, text=True, **kwargs)


def query(args):
    return run(args, capture_output=True).stdout.strip()


def remote(args, **kwargs):
    return run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            "maxwell",
            shlex.join([str(a) for a in args]),
        ],
        **kwargs,
    )


def main():
    global B12X_SOURCE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--b12x-source", type=Path, default=B12X_SOURCE)
    args = parser.parse_args()
    B12X_SOURCE = args.b12x_source.resolve()
    if query(["git", "-C", SOURCE, "status", "--porcelain"]):
        raise RuntimeError("Candidate source worktree must be committed and clean")
    revision = query(["git", "-C", SOURCE, "rev-parse", "HEAD"])
    tree = query(["git", "-C", SOURCE, "rev-parse", "HEAD^{tree}"])
    assert not query(["git", "-C", B12X_SOURCE, "status", "--porcelain"])
    b12x_revision = query(["git", "-C", B12X_SOURCE, "rev-parse", "HEAD"])
    b12x_tree = query(["git", "-C", B12X_SOURCE, "rev-parse", "HEAD^{tree}"])
    version_count = query(["git", "-C", SOURCE, "rev-list", "--count", "HEAD"])
    profile = (
        f"qwen38-qsa-selection-{revision[:8]}-b12x{b12x_revision[:8]}-sm121-{args.attempt}"
    )
    tag = f"spark-vllm:{profile}"
    output = WORKSPACE / "benchmark-results/qwen38-qsa-selection-20260917" / args.attempt
    context = WORKSPACE / "build-contexts" / profile
    output.mkdir(parents=True, exist_ok=False)
    context.mkdir(parents=True, exist_ok=False)
    archive = context / "vllm.tar"
    run(
        [
            "git",
            "-C",
            SOURCE,
            "archive",
            "--format=tar",
            f"--output={archive}",
            revision,
        ]
    )
    with archive.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    b12x_archive = context / "b12x.tar"
    run(
        [
            "git",
            "-C",
            B12X_SOURCE,
            "archive",
            "--format=tar",
            f"--output={b12x_archive}",
            b12x_revision,
        ]
    )
    with b12x_archive.open("rb") as stream:
        b12x_digest = hashlib.file_digest(stream, "sha256").hexdigest()
    parent = json.loads(PARENT.read_text())
    manifest = {
        "schema_version": 1,
        "kind": "spark-vllm-image-manifest",
        "profile_id": profile,
        "status": "candidate",
        "candidate_tag": tag,
        "target": parent["target"],
        "base": {
            "image_id": BASE,
            "torch": "2.13.0+cu130",
            "cuda": "13.0",
            "python": "3.12.3",
        },
        "sources": {
            "vllm": {
                "repository": parent["sources"]["vllm"]["repository"],
                "commit": revision,
                "tree": tree,
                "branch": "codex/qwen-moe-padding-20260917",
                "version": f"0.1.dev{version_count}+g{revision[:9]}",
                "archive_sha256": digest,
                "worktree": str(SOURCE),
            },
            "b12x": {
                **parent["sources"]["b12x"],
                "commit": b12x_revision,
                "tree": b12x_tree,
                "branch": "codex/qwen-qsa-selection-20260917",
                "upstream_commit": "5cc27234eaf4113dc41cb308753e152b1e71ce8e",
                "archive_sha256": b12x_digest,
                "worktree": str(B12X_SOURCE),
            },
        },
        "dependencies": parent["dependencies"],
        "model": {
            "name": "Qwen3.8-Flash-Next",
            "quantization": "NVFP4 QAD",
            "host_path": "/home/jasonc/models/Qwen3.8-Flash-Next-NVFP4",
            "weights_in_image": False,
            "export_manifest_sha256": "8a0b93599e3edb4ab25357e8af16cf1ac2c6a61354fcec9d6aa50ee9fbf94397",
        },
        "cache_namespace": f"/home/jasonc/.cache/{profile}",
        "qualification": {"status": "research-only", "builder": "maxwell"},
        "native_compatibility": {
            "source_base": "2ac48a52c16b362ac3418810e8d7d317003ae6d1",
            "statement": "csrc, cmake, requirements and external kernel sources match the base. CMake changes add an enabled-by-default pre-SM90 build option and preserve patched-header timestamps; compiled kernel semantics are unchanged.",
        },
    }
    assert not query(["git", "-C", SOURCE, "diff", "2ac48a52", revision, "--", "csrc", "cmake", "requirements"])
    (output / "native-build-diff.patch").write_text(query(["git", "-C", SOURCE, "diff", "2ac48a52", revision, "--", "CMakeLists.txt"]))
    for directory in (context, output):
        (directory / "image-manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
    for name in ("Dockerfile", "install-candidate.py"):
        shutil.copy2(ROOT / name, context / name)
    manifest["dependencies"]["test_support"] = json.loads(
        (ROOT / "test-dependencies.json").read_text()
    )
    manifest["dependencies"]["reuse_note"] = (
        "Runtime packages and native extensions are inherited unchanged from the immutable base. Pinned pure-Python test support enables repository fixtures and async tests."
    )
    for directory in (context, output):
        (directory / "image-manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
    shutil.copytree(ROOT / "test-wheels", context / "test-wheels")
    identity = remote(["hostname"], capture_output=True).stdout.strip().split(".")[0]
    architecture = remote(["uname", "-m"], capture_output=True).stdout.strip()
    assert identity == "maxwell" and architecture == "aarch64"
    assert (
        remote(
            ["docker", "image", "inspect", "--format", "{{.Id}}", BASE_TAG],
            capture_output=True,
        ).stdout.strip()
        == BASE
    )
    existing = subprocess.run(
        ["ssh", "maxwell", shlex.join(["docker", "image", "inspect", tag])],
        capture_output=True,
        check=False,
    )
    if existing.returncode == 0:
        raise RuntimeError("Immutable candidate tag already exists")
    available = remote(
        ["awk", "/MemAvailable/ {print $2}", "/proc/meminfo"], capture_output=True
    ).stdout.strip()
    if int(available) < 10 * 1024**2:
        raise RuntimeError(
            "Maxwell has less than 10 GiB available for the CPU-only build"
        )
    remote(["mkdir", "-p", context])
    run(
        [
            "rsync",
            "-rt",
            "-e",
            "ssh -o BatchMode=yes -o ConnectTimeout=10",
            str(context) + "/",
            "maxwell:" + str(context) + "/",
        ]
    )
    assert (
        remote(["sha256sum", archive], capture_output=True).stdout.split()[0] == digest
    )
    assert (
        remote(["sha256sum", b12x_archive], capture_output=True).stdout.split()[0]
        == b12x_digest
    )
    for package in manifest["dependencies"]["test_support"]:
        assert (
            remote(
                ["sha256sum", context / "test-wheels" / package["filename"]],
                capture_output=True,
            ).stdout.split()[0]
            == package["sha256"]
        )
    command = [
        "docker",
        "build",
        "--pull=false",
        "--network=none",
        "--progress=plain",
        "-t",
        tag,
    ]
    for name, value in {
        "BASE_IMAGE": BASE_TAG,
        "BASE_IMAGE_ID": BASE,
        "PROFILE_ID": profile,
        "VLLM_COMMIT": revision,
        "VLLM_TREE": tree,
        "VLLM_VERSION": manifest["sources"]["vllm"]["version"],
        "B12X_COMMIT": manifest["sources"]["b12x"]["commit"],
        "B12X_TREE": manifest["sources"]["b12x"]["tree"],
    }.items():
        command.extend(["--build-arg", f"{name}={value}"])
    command.append(str(context))
    started = time.time()
    with (output / "docker-build.log").open("w") as log:
        remote(command, stdout=log, stderr=subprocess.STDOUT)
    image_id = remote(
        ["docker", "image", "inspect", "--format", "{{.Id}}", tag], capture_output=True
    ).stdout.strip()
    actual = remote(
        [
            "docker",
            "run",
            "--rm",
            "--network=none",
            "--entrypoint",
            "cat",
            image_id,
            "/opt/spark-vllm/image-manifest.json",
        ],
        capture_output=True,
    ).stdout
    assert json.loads(actual) == manifest
    receipt = {
        "profile_id": profile,
        "image_tag": tag,
        "image_id": image_id,
        "architecture": "arm64",
        "base_image_id": BASE,
        "manifest_path": "/opt/spark-vllm/image-manifest.json",
        "sources": manifest["sources"],
        "builder": "maxwell",
        "build_complete": True,
        "started_at": started,
        "completed_at": time.time(),
        "gpu_execution_performed": False,
    }
    (output / "candidate-build.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
