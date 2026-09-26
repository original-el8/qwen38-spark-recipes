"""Install paired source trees while retaining the parent Torch and CUDA ABI."""

import hashlib
import json
import shutil
from importlib import metadata
from pathlib import Path

root = Path("/opt/spark-vllm")
manifest = json.loads((root / "image-manifest.json").read_text())
parent = json.loads((root / "parent-image-manifest.json").read_text())
assert metadata.version("torch") == "2.13.0+cu130"
assert metadata.version("nvidia-cutlass-dsl") == "4.6.2"
old = root / "base-vllm" / "vllm"
source = root / "vllm" / "vllm"
preserved = {}
for binary in old.glob("*.so"):
    target = source / binary.name
    shutil.copy2(binary, target)
    with binary.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    with target.open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == digest
    preserved[binary.name] = digest
assert {
    "_C_stable_libtorch.abi3.so",
    "_moe_C_stable_libtorch.abi3.so",
    "fs_io_C.abi3.so",
    "cumem_allocator.abi3.so",
} <= preserved.keys()
for name in ("vllm_flash_attn", "third_party/flashmla"):
    target = source / name
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(old / name, target)
revision = manifest["sources"]["vllm"]["commit"]
version = manifest["sources"]["vllm"]["version"]
development_version = version.split(".", 2)[2].split("+", 1)[0]
(source / "_version.py").write_text(
    f"__version__ = version = {version!r}\n"
    f"__version_tuple__ = version_tuple = (0, 1, {development_version!r}, 'g{revision[:9]}')\n"
    f"__commit_id__ = {revision!r}\n"
)
import vllm  # noqa: E402 - validate the installed source after writing its version
import b12x  # noqa: E402
from b12x.comm.roce import _proxy  # noqa: E402
from b12x.loader import _native  # noqa: E402

assert vllm.__version__ == version
assert metadata.version("pytest") == "9.1.1"
for package in manifest["dependencies"]["test_support"]:
    assert metadata.version(package["package"]) == package["version"]
assert Path(b12x.__file__).resolve().is_relative_to(root / "b12x")
from b12x.preparation import PreparationRequest
from vllm.models.qwen3_8_flash_next import Qwen3_8FlashNextForConditionalGeneration
assert PreparationRequest is not None
assert Qwen3_8FlashNextForConditionalGeneration is not None
assert _proxy.load().roce_abi_version() > 0
_native.load()
(root / "cache-build-checks.json").write_text(
    json.dumps(
        {
            "parent_image_id": manifest["base"]["image_id"],
            "preserved_extensions": preserved,
            "source_version": vllm.__version__,
            "torch": metadata.version("torch"),
            "cutlass_dsl": metadata.version("nvidia-cutlass-dsl"),
            "pytest": metadata.version("pytest"),
            "b12x_source": str(Path(b12x.__file__).resolve()),
            "b12x_commit": manifest["sources"]["b12x"]["commit"],
            "roce_abi": _proxy.load().roce_abi_version(),
        },
        indent=2,
    )
    + "\n"
)
shutil.rmtree(root / "base-vllm")
shutil.rmtree(root / "base-b12x")
