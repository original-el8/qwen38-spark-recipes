import inspect
import vllm.models.qwen4_exp.nvidia.model as M
from vllm.models.qwen4_exp.common import hc_prefill
assert hasattr(hc_prefill, "RowOwnership") and hasattr(hc_prefill, "configure")
src = inspect.getsource(M)
# The serving path is ForConditionalGeneration -> language_model.model, so the HC gate
# must live in Qwen4ExpModel.forward, not only in ForCausalLM.forward.
assert "hc_prefill.eligible(self, positions.shape[-1])" in src, "model-level HC gate missing"
assert "hc_prefill_eager=True" in src, "eager HC dispatch missing"
for needle in ("hc_prefill.configure(self", "hc_owner", "hc_owner.reduce", "defer_hc_reductions", "hc_block_output"):
    assert needle in src, needle
from vllm.model_executor.layers.linear import _register_b12x_row_parallel_collective  # noqa: F401
# the gate must be in Qwen4ExpModel.forward specifically
assert "hc_prefill.eligible" in inspect.getsource(M.Qwen4ExpModel.forward), "gate not in Qwen4ExpModel.forward"
print("R6H-HC-GAUNTLET-OK")
