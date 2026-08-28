# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Exact-order GDN speculative verify and accepted-state commit kernels."""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize=["N", "T"])
def _gdn_exact_spec_verify_kernel(
    A_log,
    a,
    b,
    dt_bias,
    q,
    k,
    v,
    o,
    state,
    cu_seqlens,
    state_indices,
    previous_num_accepted,
    scratch_a,
    scratch_b,
    scratch_k,
    scratch_v,
    source_state_indices,
    scale,
    N: tl.int64,
    T: tl.int64,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    MAX_SPEC_LEN: tl.constexpr,
    stride_state_token: tl.constexpr,
    stride_indices_seq: tl.constexpr,
    stride_indices_tok: tl.constexpr,
):
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    is_leader = (i_k == 0) & (i_v == 0) & (i_hv == 0)
    bos = tl.load(cu_seqlens + i_n).to(tl.int64)
    eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
    seq_len = eos - bos
    if (seq_len <= 0) | (seq_len > MAX_SPEC_LEN):
        if is_leader:
            tl.store(source_state_indices + i_n, -1)
        return

    previous_accepted = tl.load(previous_num_accepted + i_n).to(tl.int64)
    if (previous_accepted < 1) | (previous_accepted > MAX_SPEC_LEN):
        if is_leader:
            tl.store(source_state_indices + i_n, -1)
        return
    state_idx = tl.load(
        state_indices
        + i_n * stride_indices_seq
        + (previous_accepted - 1) * stride_indices_tok
    ).to(tl.int64)
    if is_leader:
        tl.store(source_state_indices + i_n, state_idx)
    if state_idx <= 0:
        return

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]

    p_q = q + (bos * H + i_h) * K + o_k
    p_k = k + (bos * H + i_h) * K + o_k
    p_v = v + (bos * HV + i_hv) * V + o_v
    p_a = a + bos * HV + i_hv
    p_b = b + bos * HV + i_hv
    p_o = o + ((i_k * T + bos) * HV + i_hv) * V + o_v

    p_h = (
        state
        + state_idx * stride_state_token
        + i_hv * V * K
        + o_v[:, None] * K
        + o_k[None, :]
    )
    b_h = tl.load(p_h, mask=mask_h, other=0).to(tl.float32)
    A_log_h = tl.load(A_log + i_hv).to(tl.float32)
    dt_bias_h = tl.load(dt_bias + i_hv).to(tl.float32)

    for i_t in range(0, seq_len):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k_raw = tl.load(p_k, mask=mask_k, other=0)
        b_v_raw = tl.load(p_v, mask=mask_v, other=0)
        b_a_raw = tl.load(p_a)
        b_b_raw = tl.load(p_b)

        scratch_token = i_n * MAX_SPEC_LEN + i_t
        if (i_v == 0) & (i_hv == i_h * (HV // H)):
            p_scratch_k = scratch_k + (scratch_token * H + i_h) * K + o_k
            tl.store(p_scratch_k, b_k_raw, mask=mask_k)
        if i_k == 0:
            p_scratch_v = scratch_v + (scratch_token * HV + i_hv) * V + o_v
            tl.store(p_scratch_v, b_v_raw, mask=mask_v)
            if i_v == 0:
                tl.store(scratch_a + scratch_token * HV + i_hv, b_a_raw)
                tl.store(scratch_b + scratch_token * HV + i_hv, b_b_raw)

        b_k = b_k_raw.to(tl.float32)
        b_v = b_v_raw.to(tl.float32)
        x = b_a_raw.to(tl.float32) + dt_bias_h
        softplus_x = tl.where(x <= 20.0, tl.log(1 + tl.exp(x)), x)
        b_g = -tl.exp(A_log_h) * softplus_x
        b_beta = tl.sigmoid(b_b_raw.to(tl.float32))

        b_q = b_q * tl.rsqrt(tl.sum(b_q * b_q) + 1e-6)
        b_k = b_k * tl.rsqrt(tl.sum(b_k * b_k) + 1e-6)
        b_q = b_q * scale
        b_h *= tl.exp(b_g)
        b_v -= tl.sum(b_h * b_k[None, :], 1)
        b_v *= b_beta
        b_h += b_v[:, None] * b_k[None, :]
        b_o = tl.sum(b_h * b_q[None, :], 1)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        p_q += H * K
        p_k += H * K
        p_v += HV * V
        p_a += HV
        p_b += HV
        p_o += HV * V


@triton.jit
def _gdn_exact_spec_commit_kernel(
    A_log,
    dt_bias,
    scratch_a,
    scratch_b,
    scratch_k,
    scratch_v,
    state,
    source_state_indices,
    destination_state_indices,
    spec_request_indices,
    current_num_accepted,
    N: tl.int64,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    MAX_SPEC_LEN: tl.constexpr,
    stride_state_token: tl.constexpr,
    stride_indices_seq: tl.constexpr,
    stride_indices_tok: tl.constexpr,
):
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)

    request_idx = tl.load(spec_request_indices + i_n).to(tl.int64)
    if (request_idx < 0) | (request_idx >= N):
        return
    num_accepted = tl.load(current_num_accepted + request_idx).to(tl.int64)
    if (num_accepted < 1) | (num_accepted > MAX_SPEC_LEN):
        return
    source_idx = tl.load(source_state_indices + i_n).to(tl.int64)
    destination_idx = tl.load(
        destination_state_indices
        + i_n * stride_indices_seq
        + (num_accepted - 1) * stride_indices_tok
    ).to(tl.int64)
    if (source_idx <= 0) | (destination_idx <= 0):
        return

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]
    p_source = (
        state
        + source_idx * stride_state_token
        + i_hv * V * K
        + o_v[:, None] * K
        + o_k[None, :]
    )
    b_h = tl.load(p_source, mask=mask_h, other=0).to(tl.float32)
    A_log_h = tl.load(A_log + i_hv).to(tl.float32)
    dt_bias_h = tl.load(dt_bias + i_hv).to(tl.float32)

    for i_t in range(0, MAX_SPEC_LEN):
        scratch_token = i_n * MAX_SPEC_LEN + i_t
        b_k = tl.load(
            scratch_k + (scratch_token * H + i_h) * K + o_k,
            mask=mask_k,
            other=0,
        ).to(tl.float32)
        b_v = tl.load(
            scratch_v + (scratch_token * HV + i_hv) * V + o_v,
            mask=mask_v,
            other=0,
        ).to(tl.float32)
        b_a = tl.load(scratch_a + scratch_token * HV + i_hv).to(tl.float32)
        b_b = tl.load(scratch_b + scratch_token * HV + i_hv).to(tl.float32)

        x = b_a + dt_bias_h
        softplus_x = tl.where(x <= 20.0, tl.log(1 + tl.exp(x)), x)
        b_g = -tl.exp(A_log_h) * softplus_x
        b_beta = tl.sigmoid(b_b)
        b_k = b_k * tl.rsqrt(tl.sum(b_k * b_k) + 1e-6)
        b_h *= tl.exp(b_g)
        b_v -= tl.sum(b_h * b_k[None, :], 1)
        b_v *= b_beta
        b_h += b_v[:, None] * b_k[None, :]

        p_destination = (
            state
            + destination_idx * stride_state_token
            + i_hv * V * K
            + o_v[:, None] * K
            + o_k[None, :]
        )
        tl.store(
            p_destination,
            b_h.to(p_destination.dtype.element_ty),
            mask=mask_h & (num_accepted == i_t + 1),
        )


def gdn_exact_spec_verify(
    *,
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    state_indices: torch.Tensor,
    previous_num_accepted: torch.Tensor,
    scratch_a: torch.Tensor,
    scratch_b: torch.Tensor,
    scratch_k: torch.Tensor,
    scratch_v: torch.Tensor,
    source_state_indices: torch.Tensor,
) -> torch.Tensor:
    """Verify sequentially while storing compact inputs instead of states."""
    _, T, H, K = k.shape
    HV, V = v.shape[2:]
    N = len(cu_seqlens) - 1
    max_spec_len = scratch_a.shape[1]
    if scratch_a.shape[0] < N or source_state_indices.shape[0] < N:
        raise ValueError(f"spec request count {N} exceeds scratch capacity")
    if N * max_spec_len < T:
        raise ValueError(f"spec token count {T} exceeds {N} * {max_spec_len}")
    if state.dtype != torch.float16:
        raise ValueError(
            f"exact GDN state commit requires FP16 state, got {state.dtype}"
        )
    activation_tensors = (a, b, q, k, v, scratch_a, scratch_b, scratch_k, scratch_v)
    if any(tensor.dtype != a.dtype for tensor in activation_tensors):
        raise ValueError("exact GDN scratch must preserve the activation dtype")
    if state_indices.ndim != 2 or state_indices.shape[1] != max_spec_len:
        raise ValueError("state indices must match the configured speculative window")

    BK = triton.next_power_of_2(K)
    BV = min(triton.next_power_of_2(V), 32)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1, "NK > 1 is not supported"
    out = q.new_empty(NK, *v.shape)
    _gdn_exact_spec_verify_kernel[(NK, NV, N * HV)](
        A_log,
        a.contiguous(),
        b.contiguous(),
        dt_bias,
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        out,
        state,
        cu_seqlens,
        state_indices,
        previous_num_accepted,
        scratch_a,
        scratch_b,
        scratch_k,
        scratch_v,
        source_state_indices,
        K**-0.5,
        N=N,
        T=T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        MAX_SPEC_LEN=max_spec_len,
        stride_state_token=state.stride(0),
        stride_indices_seq=state_indices.stride(0),
        stride_indices_tok=state_indices.stride(1),
        num_warps=4,
        num_stages=3,
    )
    return out.squeeze(0)


def commit_gdn_exact_spec_state(
    *,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scratch_a: torch.Tensor,
    scratch_b: torch.Tensor,
    scratch_k: torch.Tensor,
    scratch_v: torch.Tensor,
    state: torch.Tensor,
    source_state_indices: torch.Tensor,
    destination_state_indices: torch.Tensor,
    spec_request_indices: torch.Tensor,
    current_num_accepted: torch.Tensor,
    num_rows: int,
) -> None:
    """Commit accepted states in the baseline sequential FP16 order."""
    if num_rows == 0:
        return
    if num_rows > scratch_k.shape[0]:
        raise ValueError(f"commit row count {num_rows} exceeds scratch capacity")
    _, max_spec_len, H, K = scratch_k.shape
    HV, V = scratch_v.shape[2:]
    BK = triton.next_power_of_2(K)
    BV = min(triton.next_power_of_2(V), 32)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1, "NK > 1 is not supported"
    _gdn_exact_spec_commit_kernel[(NK, NV, num_rows * HV)](
        A_log,
        dt_bias,
        scratch_a,
        scratch_b,
        scratch_k,
        scratch_v,
        state,
        source_state_indices,
        destination_state_indices,
        spec_request_indices,
        current_num_accepted,
        N=current_num_accepted.shape[0],
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        MAX_SPEC_LEN=max_spec_len,
        stride_state_token=state.stride(0),
        stride_indices_seq=destination_state_indices.stride(0),
        stride_indices_tok=destination_state_indices.stride(1),
        num_warps=4,
        num_stages=3,
    )
