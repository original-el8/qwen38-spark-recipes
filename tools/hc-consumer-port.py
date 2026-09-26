import ast, sys
src = open('/tmp/model_ours.py').read()
def rep(old, new, n=1, tag=""):
    global src
    c = src.count(old)
    assert c == n, f"anchor count {c}!={n} for {tag}: {old[:60]!r}"
    src = src.replace(old, new)
    return True

# E1 imports
rep("from vllm.distributed import get_pp_group\n",
    "from vllm.distributed import get_pp_group, tensor_model_parallel_all_reduce\n", 1, "E1a")
rep("from .ple_layer import Qwen4ExpPLELayer\n",
    "from .ple_layer import Qwen4ExpPLELayer\nfrom ..common import hc_prefill\n", 1, "E1b")

# E2 decoder init defer flag
rep("""        self.layer_idx = extract_layer_index(prefix)
        if vllm_config.parallel_config.use_sequence_parallel_moe:""",
    """        self.layer_idx = extract_layer_index(prefix)
        self.defer_hc_reductions = (
            envs.VLLM_QWEN3_8_HC_PREFILL_MODE != "off"
            and self.layer_idx < config.num_hidden_layers
        )
        if vllm_config.parallel_config.use_sequence_parallel_moe:""", 1, "E2")

# E3 linear_attn kwargs
rep("""                overlap_input_projections=uses_b12x(vllm_config)
                and envs.VLLM_QWEN3_8_FLASH_NEXT_OVERLAP,
            )""",
    """                overlap_input_projections=uses_b12x(vllm_config)
                and envs.VLLM_QWEN3_8_FLASH_NEXT_OVERLAP,
                reduce_results=not self.defer_hc_reductions,
                prefill_checkpoint_blocks=int(envs.VLLM_QWEN3_8_PREFILL_COALESCE),
            )""", 1, "E3")

# E4 self_attn reduce_results (both attention branches)
rep("""                    quant_config=quant_config,
                    prefix=f"{prefix}.self_attn",
                )""",
    """                    quant_config=quant_config,
                    reduce_results=not self.defer_hc_reductions,
                    prefix=f"{prefix}.self_attn",
                )""", 2, "E4")

# E5 MoE + MLP reduce_results
rep("""            self.mlp = Qwen4ExpSparseMoeBlock(
                vllm_config=vllm_config, prefix=f"{prefix}.mlp"
            )""",
    """            self.mlp = Qwen4ExpSparseMoeBlock(
                vllm_config=vllm_config,
                prefix=f"{prefix}.mlp",
                reduce_results=not self.defer_hc_reductions,
            )""", 1, "E5a")
rep("""                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )""",
    """                quant_config=quant_config,
                reduce_results=not self.defer_hc_reductions,
                prefix=f"{prefix}.mlp",
            )""", 1, "E5b")

# E6 MoE block signature
rep("""    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        parallel_config = vllm_config.parallel_config
        if parallel_config.use_sequence_parallel_moe:
            raise NotImplementedError(
                "Qwen4Exp HC does not support sequence-parallel MoE"
            )
        super().__init__(vllm_config=vllm_config, prefix=prefix)""",
    """    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        *,
        reduce_results: bool = True,
    ) -> None:
        parallel_config = vllm_config.parallel_config
        if parallel_config.use_sequence_parallel_moe:
            raise NotImplementedError(
                "Qwen4Exp HC does not support sequence-parallel MoE"
            )
        super().__init__(
            vllm_config=vllm_config, prefix=prefix, reduce_results=reduce_results
        )""", 1, "E6")

# E7 decoder forward: signature + ownership plumbing
rep("""        output_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if prev_block_output is None:
            assert prev_injection is None
        attn_hc = self.attn_hyper_connection
        if self.ple is not None:""",
    """        output_indices: torch.Tensor | None = None,
        hc_owner: hc_prefill.RowOwnership | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if prev_block_output is None:
            assert prev_injection is None
        attn_hc = self.attn_hyper_connection
        if hc_owner is not None and output_indices is not None:
            raise ValueError("Qwen HC ownership cannot compact decode outputs")
        if self.ple is not None:""", 1, "E7a")

# ple path: gather before, and the port's ple returns a delta (ours adds internally)
rep("""            if input_ids is None or query_start_loc is None or ngram_context is None:
                raise RuntimeError("PLE inputs were not prepared")
            hidden_states = self.ple(""",
    """            if hc_owner is not None:
                hidden_states = hc_owner.gather(hidden_states)
            if input_ids is None or query_start_loc is None or ngram_context is None:
                raise RuntimeError("PLE inputs were not prepared")
            hidden_states = self.ple(""", 1, "E7b")

rep("""            hidden_states = self.ple(
                hidden_states,
                input_ids,
                query_start_loc,
                ngram_context,
            )
""",
    """            hidden_states = self.ple(
                hidden_states,
                input_ids,
                query_start_loc,
                ngram_context,
            )
            if hc_owner is not None:
                hidden_states = hc_owner.local(hidden_states)
""", 1, "E7c")

# attn block_input gather + deferred reduces
rep("""        # Fuse a pending combine with this HC module's mix when possible.
        if prev_block_output is not None:
            hidden_states, block_input, injection = attn_hc.combine_and_mix(
                hidden_states, prev_block_output, prev_injection
            )
        else:
            hidden_states, block_input, injection = attn_hc.mix(hidden_states)
""",
    """        # Fuse a pending combine with this HC module's mix when possible.
        if prev_block_output is not None:
            hidden_states, block_input, injection = attn_hc.combine_and_mix(
                hidden_states, prev_block_output, prev_injection
            )
        else:
            hidden_states, block_input, injection = attn_hc.mix(hidden_states)

        if hc_owner is not None:
            block_input = hc_owner.gather(block_input)
""", 1, "E7d")

rep("""            raise ValueError("Invalid layer_type")

        if output_indices is not None:""",
    """            raise ValueError("Invalid layer_type")

        if self.defer_hc_reductions:
            attn_out = (
                hc_owner.reduce(attn_out)
                if hc_owner is not None
                else tensor_model_parallel_all_reduce(attn_out)
            )
        if output_indices is not None:""", 1, "E7e")

rep("""        mlp_hc = self.mlp_hyper_connection
        hidden_states, block_input, injection = mlp_hc.combine_and_mix(
            hidden_states, attn_out, injection
        )
        mlp_out = self.mlp(block_input)
        return hidden_states, mlp_out, injection""",
    """        mlp_hc = self.mlp_hyper_connection
        hidden_states, block_input, injection = mlp_hc.combine_and_mix(
            hidden_states, attn_out, injection
        )
        if hc_owner is not None:
            block_input = hc_owner.gather(block_input)
        mlp_out = self.mlp(block_input)
        if self.defer_hc_reductions:
            mlp_out = (
                hc_owner.reduce(mlp_out)
                if hc_owner is not None
                else tensor_model_parallel_all_reduce(mlp_out)
            )
        return hidden_states, mlp_out, injection""", 1, "E7f")

# E8 model init: configure + b12x row-parallel collective registration
rep("""        else:
            self.register_buffer("_mtp_hidden_buffer", None, persistent=False)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)""",
    """        else:
            self.register_buffer("_mtp_hidden_buffer", None, persistent=False)
        hc_prefill.configure(self, vllm_config, envs.VLLM_QWEN3_8_HC_PREFILL_MODE)
        if self.hc_prefill_mode != "off":
            from vllm.model_executor.layers.linear import (
                _register_b12x_row_parallel_collective,
            )

            _register_b12x_row_parallel_collective(
                self, f"{prefix}.hc_block_output", config.hidden_size, True
            )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)""", 1, "E8")

# E9 model forward: ownership + row-sharded prefill
rep("""        deepstack_input_embeds: IntermediateTensors | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:""",
    """        deepstack_input_embeds: IntermediateTensors | None = None,
        hc_prefill_eager: bool = False,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:""", 1, "E9a")

rep("""            hidden_states = intermediate_tensors["hidden_states"]

        block_output = None
        injection = None
        last_layer = None""",
    """            hidden_states = intermediate_tensors["hidden_states"]

        hc_owner = (
            hc_prefill.create(self, hidden_states.shape[0])
            if hc_prefill_eager
            else None
        )
        full_rows = hidden_states.shape[0]
        if hc_owner is not None:
            hidden_states = hc_owner.local(hidden_states)
        block_output = None
        injection = None
        last_layer = None""", 1, "E9b")

rep("""                input_ids=input_ids,
                query_start_loc=query_start_loc,
                ngram_context=ngram_context,
            )
            if deepstack_input_embeds is not None and layer_idx < len(""",
    """                input_ids=input_ids,
                query_start_loc=query_start_loc,
                ngram_context=ngram_context,
                hc_owner=hc_owner,
            )
            if deepstack_input_embeds is not None and layer_idx < len(""", 1, "E9c")

rep("""                deepstack_embed = deepstack_input_embeds[
                    f"deepstack_input_embeds_{layer_idx}"
                ]
                deepstack_embed = (""",
    """                deepstack_embed = deepstack_input_embeds[
                    f"deepstack_input_embeds_{layer_idx}"
                ]
                if hc_owner is not None:
                    deepstack_embed = hc_owner.local(deepstack_embed)
                deepstack_embed = (""", 1, "E9d")

rep("""        if self._mtp_hidden_buffer is not None:
            # Capture the pre-final-mixer multi-stream hidden state
            # [T, hc_count*H] for the MTP drafter (zero extra compute:
            # this tensor is needed by the final mixer regardless).
            num_tokens = multi_hidden.shape[0]
            self._mtp_hidden_buffer[:num_tokens].copy_(multi_hidden)
        return sample_hidden_states""",
    """        if hc_owner is not None:
            sample_hidden_states = hc_owner.gather(sample_hidden_states)
            if self._mtp_hidden_buffer is not None:
                multi_hidden = hc_owner.gather(multi_hidden)
        if self._mtp_hidden_buffer is not None:
            # Capture the pre-final-mixer multi-stream hidden state
            # [T, hc_count*H] for the MTP drafter (zero extra compute:
            # this tensor is needed by the final mixer regardless).
            num_tokens = multi_hidden.shape[0]
            self._mtp_hidden_buffer[:num_tokens].copy_(multi_hidden)
        if hc_prefill_eager:
            hc_prefill.report(self, hc_owner, full_rows)
        return sample_hidden_states""", 1, "E9e")

open('/tmp/model_hc.py','w').write(src)
ast.parse(src)
print("HC-PORT-PATCH-OK lines:", len(src.split("\n")))

# ---------------------------------------------------------------------------
# NOTE (2026-09-26): the gate relocation applied after this script runs.
# This script moves the consumer wiring (above), but the HC entry gate must end
# up in Qwen4ExpModel.forward, NOT only Qwen4ExpForCausalLM.forward: the serving
# path is Qwen4ExpForConditionalGeneration.forward -> self.language_model.model(...),
# which never calls the ForCausalLM wrapper. Gating only in ForCausalLM leaves HC
# silently inert (observed: zero eligible() calls during 16k prefill). The r6h
# variant relocates the gate into Qwen4ExpModel.forward, guarded by
# `not hc_prefill_eager and hc_prefill_mode != "off"`, and re-enters itself with
# hc_prefill_eager=True. See RECIPE.md "HC token-row-sharded prefill".
# ---------------------------------------------------------------------------
