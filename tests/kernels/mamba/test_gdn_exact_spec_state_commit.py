# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)
from vllm.model_executor.layers.mamba.ops.gdn_exact_spec_state import (
    commit_gdn_exact_spec_state,
    gdn_exact_spec_verify,
)
from vllm.platforms import current_platform
from vllm.third_party.flash_linear_attention.ops.fused_sigmoid_gating import (
    fused_sigmoid_gating_delta_rule_update,
)
from vllm.utils.torch_utils import set_random_seed

pytestmark = pytest.mark.skip_global_cleanup


def _make_validation_inputs(num_speculative_tokens: int):
    layer = cast(Any, object.__new__(QwenGatedDeltaNetAttention))
    torch.nn.Module.__init__(layer)
    layer.speculative_config = SimpleNamespace(
        method="mtp",
        draft_model_config=SimpleNamespace(
            hf_config=SimpleNamespace(model_type="qwen3_5_mtp")
        ),
        uses_dynamic_speculative_decoding=lambda: False,
    )
    layer.num_spec = num_speculative_tokens
    layer.cache_config = SimpleNamespace(
        mamba_cache_mode="align",
        enable_prefix_caching=True,
    )
    layer.model_config = SimpleNamespace(dtype=torch.bfloat16)
    layer.get_state_dtype = lambda: (torch.bfloat16, torch.float16)
    vllm_config = SimpleNamespace(
        mamba_config=SimpleNamespace(enable_stochastic_rounding=False),
        scheduler_config=SimpleNamespace(async_scheduling=False),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
        ),
        kv_transfer_config=None,
    )
    config = SimpleNamespace(model_type="qwen3_5_text")
    return layer, vllm_config, config


@pytest.mark.parametrize("num_speculative_tokens", [1, 3])
def test_exact_state_contract_accepts_supported_matrix(
    num_speculative_tokens: int,
) -> None:
    layer, vllm_config, config = _make_validation_inputs(num_speculative_tokens)

    layer._validate_exact_spec_state_commit(vllm_config, config)


def test_exact_state_contract_rejects_k0() -> None:
    layer, vllm_config, config = _make_validation_inputs(0)

    with pytest.raises(ValueError, match="K1-K3"):
        layer._validate_exact_spec_state_commit(vllm_config, config)


def test_exact_state_contract_rejects_async_scheduling() -> None:
    layer, vllm_config, config = _make_validation_inputs(3)
    vllm_config.scheduler_config.async_scheduling = True

    with pytest.raises(ValueError, match="async scheduling"):
        layer._validate_exact_spec_state_commit(vllm_config, config)


@pytest.mark.parametrize("num_reqs", [1, 8])
@pytest.mark.parametrize("num_speculative_tokens", [1, 3])
def test_gdn_exact_spec_state_commit_is_bit_exact(
    num_reqs: int, num_speculative_tokens: int
) -> None:
    if not current_platform.is_cuda():
        pytest.skip("CUDA-only Triton kernel")

    device = torch.device("cuda")
    set_random_seed(7)
    H, HV, K, V = 16, 32, 128, 128
    spec_len = num_speculative_tokens + 1
    num_tokens = num_reqs * spec_len
    num_state_slots = num_tokens + 1

    q = torch.randn(1, num_tokens, H, K, dtype=torch.bfloat16, device=device)
    k = torch.randn_like(q)
    v = torch.randn(1, num_tokens, HV, V, dtype=torch.bfloat16, device=device)
    a = torch.randn(num_tokens, HV, dtype=torch.bfloat16, device=device)
    b = torch.randn_like(a)
    A_log = torch.randn(HV, dtype=torch.float32, device=device)
    dt_bias = torch.randn(HV, dtype=torch.float32, device=device)
    initial_state = torch.randn(
        num_state_slots,
        HV,
        V,
        K,
        dtype=torch.float16,
        device=device,
    )
    state_indices = torch.arange(
        1, num_tokens + 1, dtype=torch.int32, device=device
    ).view(num_reqs, spec_len)
    cu_seqlens = torch.arange(
        0, num_tokens + 1, spec_len, dtype=torch.int32, device=device
    )
    previous_accepted = (
        torch.arange(num_reqs, dtype=torch.int32, device=device) % spec_len + 1
    )
    current_accepted = (
        torch.arange(num_reqs, dtype=torch.int32, device=device).flip(0) % spec_len + 1
    )

    baseline_state = initial_state.clone()
    baseline_output, _ = fused_sigmoid_gating_delta_rule_update(
        A_log=A_log,
        a=a,
        b=b,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        initial_state=baseline_state,
        inplace_final_state=True,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=state_indices,
        num_accepted_tokens=previous_accepted,
        use_qk_l2norm_in_kernel=True,
    )

    candidate_state = initial_state.clone()
    scratch_a = torch.empty(num_reqs, spec_len, HV, dtype=a.dtype, device=device)
    scratch_b = torch.empty_like(scratch_a)
    scratch_k = torch.empty(num_reqs, spec_len, H, K, dtype=k.dtype, device=device)
    scratch_v = torch.empty(num_reqs, spec_len, HV, V, dtype=v.dtype, device=device)
    source_state_indices = torch.empty(num_reqs, dtype=torch.int32, device=device)
    candidate_output = gdn_exact_spec_verify(
        A_log=A_log,
        a=a,
        b=b,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        state=candidate_state,
        cu_seqlens=cu_seqlens,
        state_indices=state_indices,
        previous_num_accepted=previous_accepted,
        scratch_a=scratch_a,
        scratch_b=scratch_b,
        scratch_k=scratch_k,
        scratch_v=scratch_v,
        source_state_indices=source_state_indices,
    )

    # Model mixed-batch compaction: packed spec rows map back to sparse request rows.
    spec_request_indices = torch.arange(
        1, 2 * num_reqs, 2, dtype=torch.int32, device=device
    )
    global_accepted = torch.ones(2 * num_reqs, dtype=torch.int32, device=device)
    global_accepted[spec_request_indices.long()] = current_accepted
    commit_gdn_exact_spec_state(
        A_log=A_log,
        dt_bias=dt_bias,
        scratch_a=scratch_a,
        scratch_b=scratch_b,
        scratch_k=scratch_k,
        scratch_v=scratch_v,
        state=candidate_state,
        source_state_indices=source_state_indices,
        destination_state_indices=state_indices,
        spec_request_indices=spec_request_indices,
        current_num_accepted=global_accepted,
        num_rows=num_reqs,
    )
    torch.accelerator.synchronize()

    assert torch.equal(candidate_output, baseline_output)
    projection = torch.randn(
        HV * V,
        64,
        dtype=torch.bfloat16,
        device=device,
    )
    baseline_logits = baseline_output.reshape(num_tokens, HV * V) @ projection
    candidate_logits = candidate_output.reshape(num_tokens, HV * V) @ projection
    baseline_tokens = baseline_logits.argmax(dim=-1)
    candidate_tokens = candidate_logits.argmax(dim=-1)
    baseline_logprobs = (
        baseline_logits.float().log_softmax(dim=-1).gather(1, baseline_tokens[:, None])
    )
    candidate_logprobs = (
        candidate_logits.float()
        .log_softmax(dim=-1)
        .gather(1, candidate_tokens[:, None])
    )
    assert torch.equal(candidate_logits, baseline_logits)
    assert torch.equal(candidate_tokens, baseline_tokens)
    assert torch.equal(candidate_logprobs, baseline_logprobs)
    row_indices = torch.arange(num_reqs, device=device)
    accepted_slots = current_accepted.long() - 1
    selected_state_indices = state_indices[row_indices, accepted_slots].long()
    assert torch.equal(
        candidate_state[selected_state_indices],
        baseline_state[selected_state_indices],
    )

    if num_reqs == 8 and num_speculative_tokens == 3:
        graph = torch.cuda.CUDAGraph()
        candidate_state.copy_(initial_state)
        with torch.cuda.graph(graph):
            graph_output = gdn_exact_spec_verify(
                A_log=A_log,
                a=a,
                b=b,
                dt_bias=dt_bias,
                q=q,
                k=k,
                v=v,
                state=candidate_state,
                cu_seqlens=cu_seqlens,
                state_indices=state_indices,
                previous_num_accepted=previous_accepted,
                scratch_a=scratch_a,
                scratch_b=scratch_b,
                scratch_k=scratch_k,
                scratch_v=scratch_v,
                source_state_indices=source_state_indices,
            )
            commit_gdn_exact_spec_state(
                A_log=A_log,
                dt_bias=dt_bias,
                scratch_a=scratch_a,
                scratch_b=scratch_b,
                scratch_k=scratch_k,
                scratch_v=scratch_v,
                state=candidate_state,
                source_state_indices=source_state_indices,
                destination_state_indices=state_indices,
                spec_request_indices=spec_request_indices,
                current_num_accepted=global_accepted,
                num_rows=num_reqs,
            )
        for _ in range(2):
            candidate_state.copy_(initial_state)
            graph.replay()
            torch.accelerator.synchronize()
            assert torch.equal(graph_output, baseline_output)
            assert torch.equal(
                candidate_state[selected_state_indices],
                baseline_state[selected_state_indices],
            )


def test_gdn_exact_spec_state_skips_invalid_allocation() -> None:
    if not current_platform.is_cuda():
        pytest.skip("CUDA-only Triton kernel")

    device = torch.device("cuda")
    H, HV, K, V = 1, 1, 8, 8
    state = torch.randn(2, HV, V, K, dtype=torch.float16, device=device)
    original = state.clone()
    commit_gdn_exact_spec_state(
        A_log=torch.zeros(HV, dtype=torch.float32, device=device),
        dt_bias=torch.zeros(HV, dtype=torch.float32, device=device),
        scratch_a=torch.zeros(1, 2, HV, dtype=torch.bfloat16, device=device),
        scratch_b=torch.zeros(1, 2, HV, dtype=torch.bfloat16, device=device),
        scratch_k=torch.zeros(1, 2, H, K, dtype=torch.bfloat16, device=device),
        scratch_v=torch.zeros(1, 2, HV, V, dtype=torch.bfloat16, device=device),
        state=state,
        source_state_indices=torch.tensor([1], dtype=torch.int32, device=device),
        destination_state_indices=torch.zeros(1, 2, dtype=torch.int32, device=device),
        spec_request_indices=torch.tensor([0], dtype=torch.int32, device=device),
        current_num_accepted=torch.tensor([1], dtype=torch.int32, device=device),
        num_rows=1,
    )
    torch.accelerator.synchronize()

    assert torch.equal(state, original)


def test_gdn_exact_spec_state_rejects_non_fp16_state() -> None:
    if not current_platform.is_cuda():
        pytest.skip("CUDA-only Triton kernel")

    device = torch.device("cuda")
    H, HV, K, V = 1, 1, 8, 8
    with pytest.raises(ValueError, match="requires FP16 state"):
        gdn_exact_spec_verify(
            A_log=torch.zeros(HV, dtype=torch.float32, device=device),
            a=torch.zeros(1, HV, dtype=torch.bfloat16, device=device),
            b=torch.zeros(1, HV, dtype=torch.bfloat16, device=device),
            dt_bias=torch.zeros(HV, dtype=torch.float32, device=device),
            q=torch.zeros(1, 1, H, K, dtype=torch.bfloat16, device=device),
            k=torch.zeros(1, 1, H, K, dtype=torch.bfloat16, device=device),
            v=torch.zeros(1, 1, HV, V, dtype=torch.bfloat16, device=device),
            state=torch.zeros(2, HV, V, K, dtype=torch.float32, device=device),
            cu_seqlens=torch.tensor([0, 1], dtype=torch.int32, device=device),
            state_indices=torch.tensor([[1]], dtype=torch.int32, device=device),
            previous_num_accepted=torch.ones(1, dtype=torch.int32, device=device),
            scratch_a=torch.empty(1, 1, HV, dtype=torch.bfloat16, device=device),
            scratch_b=torch.empty(1, 1, HV, dtype=torch.bfloat16, device=device),
            scratch_k=torch.empty(1, 1, H, K, dtype=torch.bfloat16, device=device),
            scratch_v=torch.empty(1, 1, HV, V, dtype=torch.bfloat16, device=device),
            source_state_indices=torch.empty(1, dtype=torch.int32, device=device),
        )


def test_gdn_exact_spec_state_rejects_scratch_overflow() -> None:
    if not current_platform.is_cuda():
        pytest.skip("CUDA-only Triton kernel")

    device = torch.device("cuda")
    H, HV, K, V = 1, 1, 8, 8
    with pytest.raises(ValueError, match="exceeds scratch capacity"):
        gdn_exact_spec_verify(
            A_log=torch.zeros(HV, dtype=torch.float32, device=device),
            a=torch.zeros(2, HV, dtype=torch.bfloat16, device=device),
            b=torch.zeros(2, HV, dtype=torch.bfloat16, device=device),
            dt_bias=torch.zeros(HV, dtype=torch.float32, device=device),
            q=torch.zeros(1, 2, H, K, dtype=torch.bfloat16, device=device),
            k=torch.zeros(1, 2, H, K, dtype=torch.bfloat16, device=device),
            v=torch.zeros(1, 2, HV, V, dtype=torch.bfloat16, device=device),
            state=torch.zeros(3, HV, V, K, dtype=torch.float16, device=device),
            cu_seqlens=torch.tensor([0, 1, 2], dtype=torch.int32, device=device),
            state_indices=torch.tensor([[1], [2]], dtype=torch.int32, device=device),
            previous_num_accepted=torch.ones(2, dtype=torch.int32, device=device),
            scratch_a=torch.empty(1, 1, HV, dtype=torch.bfloat16, device=device),
            scratch_b=torch.empty(1, 1, HV, dtype=torch.bfloat16, device=device),
            scratch_k=torch.empty(1, 1, H, K, dtype=torch.bfloat16, device=device),
            scratch_v=torch.empty(1, 1, HV, V, dtype=torch.bfloat16, device=device),
            source_state_indices=torch.empty(1, dtype=torch.int32, device=device),
        )
