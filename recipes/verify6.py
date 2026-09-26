from b12x.sequence.ple import api, _preparation
assert hasattr(api, 'export_checkpoint')
assert 'export_checkpoint' in dir(_preparation._PleState)
import vllm, vllm.envs as E
assert [k for k in dir(E) if 'HC_PREFILL' in k]
import vllm.models.qwen4_exp.common.hc_prefill
import vllm.models.qwen4_exp.nvidia.ple_layer, vllm.models.qwen4_exp.nvidia.ple_attn
import vllm.v1.core.sched.scheduler
import inspect
assert 'autotune=' not in inspect.getsource(vllm.models.qwen4_exp.nvidia.ple_layer)
print('R6-GAUNTLET-OK')
