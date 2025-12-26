import os
from typing import List, Optional, Tuple, Union

import torch
import transformer_engine.pytorch.attention.dot_product_attention.context_parallel as te_cp
import transformer_engine.pytorch.attention.dot_product_attention.utils as dpa_utils
import transformer_engine_torch as tex
from megatron.core.extensions.transformer_engine import TEDotProductAttention
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import (
    get_te_version,
    is_te_min_version,
)
from torch import Tensor
from transformer_engine.pytorch.attention.dot_product_attention.context_parallel import (
    _get_cu_seqlens_info_with_cp,
)
from transformer_engine.pytorch.attention.dot_product_attention.context_parallel import (
    flash_attn_a2a_communicate as _flash_attn_a2a_communicate,
)
from transformer_engine.pytorch.attention.dot_product_attention.context_parallel import (
    flash_attn_fwd_out_correction,
    flash_attn_fwd_out_correction_init,
    flash_attn_fwd_second_half_out_correction,
    flash_attn_fwd_second_half_softmax_lse_correction,
    flash_attn_fwd_softmax_lse_correction,
)
from transformer_engine.pytorch.attention.dot_product_attention.context_parallel import (
    flash_attn_p2p_communicate as _flash_attn_p2p_communicate,
)
from transformer_engine.pytorch.attention.dot_product_attention.context_parallel import (
    get_cu_seqlens_on_cp_rank,
    get_fa_args,
    get_seq_chunk_ids_for_reordering_after_attn,
    get_seq_chunk_ids_for_reordering_before_attn,
)
from transformer_engine.pytorch.attention.dot_product_attention.utils import (
    FlashAttentionUtils as fa_utils,
)
from transformer_engine.pytorch.constants import (
    TE_DType,
    dist_group_type,
)
from transformer_engine.pytorch.cpp_extensions.fused_attn import (
    FusedAttnBackend,
    fused_attn_bwd,
)
from transformer_engine.pytorch.cpp_extensions.fused_attn import (
    fused_attn_fwd as _fused_attn_fwd,
)
from transformer_engine.pytorch.distributed import (
    get_distributed_rank,
    get_distributed_world_size,
)
from transformer_engine.pytorch.float8_tensor import Float8Tensor
from transformer_engine.pytorch.tensor.quantized_tensor import (
    Quantizer,
    prepare_for_saving,
    restore_from_saved,
)
from transformer_engine.pytorch.utils import (
    combine_tensors,
    get_cudnn_version,
    get_device_compute_capability,
)

from AutoTuner.utils.nvtx import nvtx_decorator, nvtx_range_pop, nvtx_range_push

from .common import CommonOpsForTest


def fused_attn_fwd(
    is_training: bool,
    max_seqlen_q: int,
    max_seqlen_kv: int,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    fake_dtype: torch.dtype,
    fused_attention_backend: tex.NVTE_Fused_Attn_Backend,
    attn_bias: torch.Tensor = None,
    cu_seqlens_q_padded: torch.Tensor = None,
    cu_seqlens_kv_padded: torch.Tensor = None,
    page_table_k: torch.Tensor = None,
    page_table_v: torch.Tensor = None,
    s_quantizer: Quantizer = None,
    o_quantizer: Quantizer = None,
    attn_scale: float = None,
    dropout: float = 0.0,
    fast_zero_fill: bool = True,
    qkv_layout: str = "sbh3d",
    attn_bias_type: str = "no_bias",
    attn_mask_type: str = "padding",
    window_size: Tuple[int, int] = (-1, -1),
    rng_gen: torch.Generator = None,
) -> Tuple[Union[torch.Tensor, None], ...]:
    key = f"fused_attn_fwd.{attn_mask_type}"
    nvtx_range_push(suffix=key)

    out = _fused_attn_fwd(
        is_training=is_training,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_kv=max_seqlen_kv,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_kv,
        q=q,
        k=k,
        v=v,
        fake_dtype=fake_dtype,
        fused_attention_backend=fused_attention_backend,
        attn_bias=attn_bias,
        cu_seqlens_q_padded=cu_seqlens_q_padded,
        cu_seqlens_kv_padded=cu_seqlens_kv_padded,
        page_table_k=page_table_k,
        page_table_v=page_table_v,
        s_quantizer=s_quantizer,
        o_quantizer=o_quantizer,
        attn_scale=attn_scale,
        dropout=dropout,
        fast_zero_fill=fast_zero_fill,
        qkv_layout=qkv_layout,
        attn_bias_type=attn_bias_type,
        attn_mask_type=attn_mask_type,
        window_size=window_size,
        rng_gen=rng_gen,
    )
    nvtx_range_pop(suffix=key)
    return out


def flash_attn_a2a_communicate(
    a2a_inputs: Union[torch.Tensor, List[torch.Tensor]],
    chunk_ids_for_a2a: torch.Tensor,
    seq_dim: int,
    cp_size: int,
    cp_group: dist_group_type,
    cp_stream: torch.cuda.Stream,
    before_attn: bool,
) -> Union[torch.Tensor, List[torch.Tensor]]:
    attn_pos = "before" if before_attn else "after"
    key = f"flash_attn_ulysses_a2a_communicate.{attn_pos}_attn"
    nvtx_range_push(suffix=key)
    res = _flash_attn_a2a_communicate(
        a2a_inputs=a2a_inputs,
        chunk_ids_for_a2a=chunk_ids_for_a2a,
        seq_dim=seq_dim,
        cp_size=cp_size,
        cp_group=cp_group,
        cp_stream=cp_stream,
        before_attn=before_attn,
    )
    nvtx_range_pop(suffix=key)
    return res


def flash_attn_p2p_communicate(
    rank, send_tensor, send_dst, recv_tensor, recv_src, cp_group, batch_p2p_comm, iter
):
    key = f"flash_attn_ulysses_p2p_communicate.step{iter}"
    nvtx_range_push(suffix=key)
    res = _flash_attn_p2p_communicate(
        rank, send_tensor, send_dst, recv_tensor, recv_src, cp_group, batch_p2p_comm
    )
    nvtx_range_pop(suffix=key)
    return res


class ProfiledAttnFuncWithCPAndKVP2P(torch.autograd.Function):
    """
    almost the same as AttnFuncWithCPAndKVP2P

    Based on this implmentation, we further support nvtx push and pop diving into it
    """

    @staticmethod
    def forward(
        ctx,
        is_training,
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_kv,
        max_seqlen_q,
        max_seqlen_kv,
        cu_seqlens_q_padded,
        cu_seqlens_kv_padded,
        dropout_p,
        softmax_scale,
        qkv_format,
        attn_mask_type,
        attn_bias_type,
        attn_bias,
        deterministic,
        use_fused_attention,
        fp8,
        fp8_meta,
        cp_group,
        cp_global_ranks,
        cp_stream,
        quantizers,
        pad_between_seqs,
        use_flash_attn_3,
    ):
        # pylint: disable=missing-function-docstring
        nvtx_range_push("transformer_engine.AttnFuncWithCPAndKVP2P.forward")
        enable_mla = k.shape[-1] != v.shape[-1]
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        if isinstance(cp_group, list):
            assert (
                qkv_format != "thd"
            ), f"{qkv_format} format is not supported with hierarchical CP implementation yet!"
            assert attn_bias_type == "no_bias", (
                f"{attn_bias_type} bias type is not supported with hierarchical CP implementation"
                " yet!"
            )
            cp_group_a2a = cp_group[0]
            cp_size_a2a = get_distributed_world_size(cp_group_a2a)
            rank_a2a = get_distributed_rank(cp_group_a2a)
            cp_group = cp_group[1]
        else:
            cp_group_a2a = None
            cp_size_a2a = 1
            rank_a2a = 0

        cp_size = get_distributed_world_size(cp_group)
        rank = get_distributed_rank(cp_group)
        send_dst = cp_global_ranks[(rank + 1) % cp_size * cp_size_a2a + rank_a2a]
        recv_src = cp_global_ranks[(rank - 1) % cp_size * cp_size_a2a + rank_a2a]
        device_compute_capability = get_device_compute_capability()
        batch_p2p_comm = int(os.getenv("NVTE_BATCH_MHA_P2P_COMM", "0")) or (
            device_compute_capability < (10, 0) and cp_size == 2
        )

        causal = "causal" in attn_mask_type
        padding = "padding" in attn_mask_type

        batch_dim = None
        seq_dim = None
        cu_seqlens_q_half, cu_seqlens_kv_half = None, None
        if qkv_format in ["bshd", "sbhd"]:
            seq_dim = qkv_format.index("s")
            if enable_mla:
                qkv_layout = qkv_format + "_" + qkv_format + "_" + qkv_format
            else:
                qkv_layout = qkv_format + "_" + qkv_format[:-2] + "2" + qkv_format[-2:]
            cu_seqlens_q_padded, cu_seqlens_kv_padded = None, None
            if use_fused_attention:
                batch_dim = qkv_format.index("b")
                cu_seqlens_q, cu_seqlens_q_half = _get_cu_seqlens_info_with_cp(
                    q.shape[batch_dim], max_seqlen_q, cp_size, cu_seqlens_q
                )
                cu_seqlens_kv, cu_seqlens_kv_half = _get_cu_seqlens_info_with_cp(
                    q.shape[batch_dim], max_seqlen_kv, cp_size, cu_seqlens_kv
                )
        else:
            qkv_layout = qkv_format + "_" + qkv_format + "_" + qkv_format
            cu_seqlens_q_padded = cu_seqlens_q_padded // cp_size
            cu_seqlens_kv_padded = cu_seqlens_kv_padded // cp_size

        max_seqlen_q = max_seqlen_q // cp_size
        max_seqlen_kv = max_seqlen_kv // cp_size
        cu_seqlens_q_per_step = [None for _ in range(cp_size)]
        cu_seqlens_kv_per_step = [None for _ in range(cp_size)]

        fused_attn_backend = None
        qkv_dtype = q.dtype
        amax_per_step = None
        S_quantizer_per_step = [None for _ in range(cp_size)]
        O_CP_quantizer_per_step = [None for _ in range(cp_size)]
        # "fp8_mha" decides outputs in fp8, while inputs are inferred from the real dtype
        is_input_fp8 = False
        is_output_fp8 = False

        (
            QKV_quantizer,
            O_quantizer,
            O_CP_quantizer,
            S_quantizer,
            dQKV_quantizer,
            dQKV_CP_quantizer,
            dO_quantizer,
            dP_quantizer,
        ) = dpa_utils.get_attention_quantizers(
            fp8, quantizers, cp_specific_quantizers=True
        )

        if fp8:
            if use_fused_attention:
                fused_attn_backend = FusedAttnBackend["FP8"]

                assert isinstance(k, q.__class__) and isinstance(
                    v, q.__class__
                ), "q, k, and v must have the same type."
                is_input_fp8 = isinstance(q, Float8Tensor)
                is_output_fp8 = fp8_meta is not None and fp8_meta["recipe"].fp8_mha
                if is_input_fp8:
                    QKV_quantizer = q._quantizer
                    q, k, v = q._data, k._data, v._data
                else:
                    q_f16, k_f16, v_f16 = q, k, v
                    if cp_size_a2a == 1 or int(os.getenv("NVTE_FP8_DPA_BWD", "1")):
                        q = QKV_quantizer(q_f16)._data
                    if int(os.getenv("NVTE_FP8_DPA_BWD", "1")):
                        k, v = [QKV_quantizer(x)._data for x in [k_f16, v_f16]]
                amax_per_step = torch.zeros(
                    (2, cp_size), dtype=torch.float32, device=q.device
                )
                # partial result quantizer
                for i in range(cp_size):
                    S_quantizer_per_step[i] = S_quantizer.copy()
                    S_quantizer_per_step[i].amax = amax_per_step[0][i].reshape((1,))
                    O_CP_quantizer_per_step[i] = O_CP_quantizer.copy()
                    O_CP_quantizer_per_step[i].amax = amax_per_step[1][i].reshape((1,))
            else:
                assert False, "FP8 is only supported with Fused Attention!"
        else:
            q_f16 = q
            if use_fused_attention:
                fused_attn_backend = FusedAttnBackend["F16_arbitrary_seqlen"]

        if cp_size_a2a > 1:
            chunk_ids_for_a2a = get_seq_chunk_ids_for_reordering_before_attn(
                cp_size_a2a, q.device
            )

            q, k, v = flash_attn_a2a_communicate(
                [q, k, v],
                chunk_ids_for_a2a,
                seq_dim,
                cp_size_a2a,
                cp_group_a2a,
                cp_stream,
                True,
            )
            if not fp8:
                q_f16 = q
            elif not is_input_fp8 and not int(os.getenv("NVTE_FP8_DPA_BWD", "1")):
                q_f16 = q
                q = QKV_quantizer(q_f16)._data

        assert qkv_format == "thd" or (
            q.shape[seq_dim] % 2 == 0 and k.shape[seq_dim] % 2 == 0
        ), "Sequence length per GPU needs to be divisible by 2!"
        if causal:
            if qkv_format == "bshd":
                # [b, s, np, hn] -> [b, 2, s//2, np, hn]
                q, k, v = [
                    x.view(x.shape[0], 2, x.shape[1] // 2, *x.shape[2:])
                    for x in [q, k, v]
                ]
            elif qkv_format == "sbhd":
                # [s, b, np, hn] -> [2, s//2, b, np, hn]
                q, k, v = [x.view(2, x.shape[0] // 2, *x.shape[1:]) for x in [q, k, v]]
        if attn_bias is not None:
            assert len(attn_bias.shape) == 4, (
                "Only support bias shape of [b, h, sq, sk] for forward, "
                "and [1, h, sq, sk] for backward!"
            )
            assert (
                attn_bias.shape[-2] % 2 == 0
                and attn_bias.shape[-1] % (2 * cp_size) == 0
            ), "Sequence length does not meet divisible requirements!"
            # [b, np, sq, sk] -> [b, np, 2, sq//2, 2*cp, sk//(2*cp)]
            attn_bias_ = attn_bias.view(
                *attn_bias.shape[:-2],
                2,
                attn_bias.shape[-2] // 2,
                2 * cp_size,
                attn_bias.shape[-1] // (2 * cp_size),
            )
            # [b, np, sq, sk] -> [b, np, sq, 2*cp, sk//(2*cp)]
            attn_bias = attn_bias.view(
                *attn_bias.shape[:-1], 2 * cp_size, attn_bias.shape[-1] // (2 * cp_size)
            )
        assert (
            q.shape[-1] % 8 == 0
        ), "hidden size per attention head should be multiple of 8"

        softmax_lse_in_packed_format = False
        if qkv_format == "thd":
            if use_fused_attention:
                softmax_lse_in_packed_format = get_cudnn_version() >= (9, 6, 0)
            else:
                softmax_lse_in_packed_format = fa_utils.v2_6_0_plus or use_flash_attn_3

        flash_attn_fwd = None
        if not use_fused_attention:
            fa_forward_kwargs = {"softmax_scale": softmax_scale}
            if use_flash_attn_3:
                from transformer_engine.pytorch.attention.dot_product_attention.backends import (
                    _flash_attn_fwd_v3,
                )

                flash_attn_fwd = _flash_attn_fwd_v3  # pylint: disable=possibly-used-before-assignment
                fa_forward_kwargs["window_size"] = (-1, 0) if causal else (-1, -1)
            else:
                if qkv_format == "thd":
                    from transformer_engine.pytorch.attention.dot_product_attention.backends import (
                        _flash_attn_varlen_fwd,
                    )

                    flash_attn_fwd = _flash_attn_varlen_fwd
                else:
                    from transformer_engine.pytorch.attention.dot_product_attention.backends import (
                        _flash_attn_fwd,
                    )

                    flash_attn_fwd = _flash_attn_fwd
                fa_forward_kwargs["dropout_p"] = dropout_p
                fa_forward_kwargs["return_softmax"] = False
                if fa_utils.v2_3_plus and not fa_utils.v2_7_0_plus:
                    fa_forward_kwargs["window_size"] = (-1, 0) if causal else (-1, -1)
                elif fa_utils.v2_7_0_plus:
                    fa_forward_kwargs["window_size_left"] = -1
                    fa_forward_kwargs["window_size_right"] = 0 if causal else -1
                if fa_utils.v2_4_plus:
                    fa_forward_kwargs["alibi_slopes"] = None
                if fa_utils.v2_5_7_plus and qkv_format == "thd":
                    fa_forward_kwargs["block_table"] = None
                if fa_utils.v2_6_0_plus:
                    fa_forward_kwargs["softcap"] = 0.0

        # Flash Attn inputs
        q_inputs = [None, None]
        kv_inputs = [None, None]
        attn_bias_inputs = [None, None]
        # Flash Attn outputs
        out_per_step = [None for _ in range(cp_size)]
        softmax_lse_per_step = [None for _ in range(cp_size)]
        rng_states = [None for _ in range(cp_size)]
        attn_biases = [None for _ in range(cp_size)]

        # create two streams to resolve wave quantization issue of Flash Attn in each step
        flash_attn_streams = [torch.cuda.current_stream(), cp_stream]
        # synchronize fwd results correction across steps
        fwd_results_correction_done = torch.cuda.Event()

        p2p_comm_buffers = [None for _ in range(cp_size)]
        if enable_mla:
            # If MLA, the shape of k and v does not match, so we flatten them
            # and split them after receiving them.
            k_shape = k.shape
            k_numel = k.numel()
            v_shape = v.shape
            p2p_comm_buffers[0] = torch.cat((k.view(-1), v.view(-1)), dim=-1)
        elif qkv_format in ["bshd", "sbhd"]:
            p2p_comm_buffers[0] = torch.cat((k.unsqueeze(-3), v.unsqueeze(-3)), dim=-3)
        else:  # qkv_format == "thd"
            p2p_comm_buffers[0] = torch.cat((k.unsqueeze(0), v.unsqueeze(0)), dim=0)
        send_recv_reqs = [[], []]

        out = None
        for i in range(cp_size + 1):
            if i < cp_size:
                with torch.cuda.stream(flash_attn_streams[i % 2]):
                    # wait until KV is received
                    nvtx_range_push(suffix=f"cp_attn.stream_step{i}.compute")
                    for req in send_recv_reqs[(i + 1) % 2]:
                        req.wait()

                    if i < (cp_size - 1):
                        p2p_comm_buffers[i + 1] = torch.empty_like(p2p_comm_buffers[i])
                        send_recv_reqs[i % 2] = flash_attn_p2p_communicate(
                            rank,
                            p2p_comm_buffers[i],
                            send_dst,
                            p2p_comm_buffers[i + 1],
                            recv_src,
                            cp_group,
                            batch_p2p_comm,
                            i,
                        )

                    if (
                        not fp8
                        or is_input_fp8
                        or int(os.getenv("NVTE_FP8_DPA_BWD", "1"))
                    ):
                        kv_inputs[i % 2] = p2p_comm_buffers[i]
                    else:
                        # KV exchange is in BF16/FP16, cast received KV in each step
                        kv_inputs[i % 2] = QKV_quantizer(p2p_comm_buffers[i])._data
                    if enable_mla:
                        # If MLA, k and v are flattened, so split them after receiving.
                        k_part = kv_inputs[i % 2][:k_numel].view(*k_shape)
                        v_part = kv_inputs[i % 2][k_numel:].view(*v_shape)
                    if causal:
                        if i == 0:
                            if pad_between_seqs:
                                cu_seqlens_q_per_step[i] = get_cu_seqlens_on_cp_rank(
                                    cu_seqlens_q,
                                    cu_seqlens_q_padded,
                                    cp_size,
                                    rank,
                                    True,
                                    True,
                                )
                                cu_seqlens_kv_per_step[i] = get_cu_seqlens_on_cp_rank(
                                    cu_seqlens_kv,
                                    cu_seqlens_kv_padded,
                                    cp_size,
                                    rank,
                                    True,
                                    True,
                                )
                            elif qkv_format == "thd":
                                cu_seqlens_q_per_step[i] = cu_seqlens_q // cp_size
                                cu_seqlens_kv_per_step[i] = cu_seqlens_kv // cp_size
                            else:
                                cu_seqlens_q_per_step[i] = cu_seqlens_q
                                cu_seqlens_kv_per_step[i] = cu_seqlens_kv
                            if qkv_format == "bshd":
                                # [b, 2, sq//2, np, hn] -> [b, sq, np, hn]
                                q_inputs[i % 2] = q.view(q.shape[0], -1, *q.shape[-2:])
                                if enable_mla:
                                    # [b, 2, sk//2, np, hn] -> [b, sk, np, hn]
                                    k_part = k_part.view(
                                        k_part.shape[0], -1, *k_part.shape[-2:]
                                    )
                                    v_part = v_part.view(
                                        v_part.shape[0], -1, *v_part.shape[-2:]
                                    )
                                else:
                                    # [b, 2, sk//2, 2, np, hn] -> [b, sk, 2, np, hn]
                                    kv_inputs[i % 2] = kv_inputs[i % 2].view(
                                        k.shape[0], -1, 2, *k.shape[-2:]
                                    )
                            elif qkv_format == "sbhd":
                                # [2, sq//2, b, np, hn] -> [sq, b, np, hn]
                                q_inputs[i % 2] = q.view(-1, *q.shape[-3:])
                                if enable_mla:
                                    # [2, sk//2, b, np, hn] -> [sk, b, np, hn]
                                    k_part = k_part.view(-1, *k_part.shape[2:])
                                    v_part = v_part.view(-1, *v_part.shape[2:])
                                else:
                                    # [2, sk//2, b, 2, np, hn] -> [sk, b, 2, np, hn]
                                    kv_inputs[i % 2] = kv_inputs[i % 2].view(
                                        -1, k.shape[2], 2, *k.shape[-2:]
                                    )
                            elif qkv_format == "thd":
                                q_inputs[i % 2] = q
                            if use_fused_attention:
                                if attn_bias is not None:
                                    idx = (rank - i) % cp_size
                                    attn_bias_inputs[i % 2] = torch.cat(
                                        (
                                            attn_bias[..., idx, :],
                                            attn_bias[..., (2 * cp_size - idx - 1), :],
                                        ),
                                        dim=-1,
                                    ).contiguous()

                                q_part = q_inputs[i % 2]
                                if not enable_mla:
                                    # If MHA, then split the KV into k_part and v_part.
                                    # Otherwise (MHA), k_part and v_part have already been split.
                                    k_part = (
                                        kv_inputs[i % 2][..., 0, :, :]
                                        if qkv_format in ["bshd", "sbhd"]
                                        else kv_inputs[i % 2][0]
                                    )
                                    v_part = (
                                        kv_inputs[i % 2][..., 1, :, :]
                                        if qkv_format in ["bshd", "sbhd"]
                                        else kv_inputs[i % 2][1]
                                    )
                                fp8_meta_kwargs = {}
                                if fp8:
                                    q_part = QKV_quantizer.create_tensor_from_data(
                                        q_part, fake_dtype=qkv_dtype, internal=True
                                    )
                                    k_part = QKV_quantizer.create_tensor_from_data(
                                        k_part, fake_dtype=qkv_dtype, internal=True
                                    )
                                    v_part = QKV_quantizer.create_tensor_from_data(
                                        v_part, fake_dtype=qkv_dtype, internal=True
                                    )
                                    fp8_meta_kwargs["s_quantizer"] = (
                                        S_quantizer_per_step[i]
                                    )
                                    fp8_meta_kwargs["o_quantizer"] = (
                                        O_CP_quantizer_per_step[i]
                                    )
                                out_per_step[i], aux_ctx_tensors = fused_attn_fwd(
                                    is_training,
                                    max_seqlen_q,
                                    max_seqlen_kv,
                                    cu_seqlens_q_per_step[i],
                                    cu_seqlens_kv_per_step[i],
                                    q_part,
                                    k_part,
                                    v_part,
                                    fake_dtype=qkv_dtype,
                                    fused_attention_backend=fused_attn_backend,
                                    attn_scale=softmax_scale,
                                    dropout=dropout_p,
                                    qkv_layout=qkv_layout,
                                    attn_mask_type=attn_mask_type,
                                    attn_bias_type=attn_bias_type,
                                    attn_bias=attn_bias_inputs[i % 2],
                                    cu_seqlens_q_padded=cu_seqlens_q_padded,
                                    cu_seqlens_kv_padded=cu_seqlens_kv_padded,
                                    **fp8_meta_kwargs,
                                )
                                if fp8:
                                    softmax_lse_per_step[i], _, rng_states[i] = (
                                        aux_ctx_tensors
                                    )
                                else:
                                    softmax_lse_per_step[i], rng_states[i], *rest = (
                                        aux_ctx_tensors
                                    )
                                    attn_biases[i] = rest[0] if len(rest) > 0 else None
                            else:
                                fa_forward_args_thd = get_fa_args(
                                    True,
                                    use_flash_attn_3,
                                    qkv_format,
                                    cu_seqlens_q=cu_seqlens_q_per_step[i],
                                    cu_seqlens_kv=cu_seqlens_kv_per_step[i],
                                    max_seqlen_q=max_seqlen_q,
                                    max_seqlen_kv=max_seqlen_kv,
                                )
                                # Need to add MLA support once Flash Attention supports MLA
                                nvtx_range_push(suffix=f"flash_attn_fwd.causal.step{i}")
                                fa_outputs = flash_attn_fwd(
                                    q_inputs[i % 2],
                                    (
                                        kv_inputs[i % 2][..., 0, :, :]
                                        if qkv_format in ["bshd", "sbhd"]
                                        else kv_inputs[i % 2][0]
                                    ),
                                    (
                                        kv_inputs[i % 2][..., 1, :, :]
                                        if qkv_format in ["bshd", "sbhd"]
                                        else kv_inputs[i % 2][1]
                                    ),
                                    *fa_forward_args_thd,
                                    causal=True,
                                    **fa_forward_kwargs,
                                )
                                nvtx_range_pop(suffix=f"flash_attn_fwd.causal.step{i}")
                                if not fa_utils.v2_7_0_plus:
                                    out_per_step[i] = fa_outputs[4]
                                    softmax_lse_per_step[i] = fa_outputs[5]
                                    if not use_flash_attn_3:
                                        rng_states[i] = fa_outputs[7]
                                else:
                                    out_per_step[i] = fa_outputs[0]
                                    softmax_lse_per_step[i] = fa_outputs[1]
                                    if not use_flash_attn_3:
                                        rng_states[i] = fa_outputs[3]
                        elif i <= rank:
                            if pad_between_seqs:
                                cu_seqlens_q_per_step[i] = get_cu_seqlens_on_cp_rank(
                                    cu_seqlens_q,
                                    cu_seqlens_q_padded,
                                    cp_size,
                                    rank,
                                    True,
                                    True,
                                )
                                cu_seqlens_kv_per_step[i] = get_cu_seqlens_on_cp_rank(
                                    cu_seqlens_kv,
                                    cu_seqlens_kv_padded,
                                    cp_size,
                                    (rank - i) % cp_size,
                                    True,
                                    False,
                                )
                            elif qkv_format == "thd":
                                cu_seqlens_q_per_step[i] = cu_seqlens_q // cp_size
                                cu_seqlens_kv_per_step[i] = cu_seqlens_kv // (
                                    cp_size * 2
                                )
                            else:
                                cu_seqlens_q_per_step[i] = cu_seqlens_q
                                cu_seqlens_kv_per_step[i] = cu_seqlens_kv_half
                            if qkv_format == "bshd":
                                # [b, 2, sq//2, np, hn] -> [b, sq, np, hn]
                                q_inputs[i % 2] = q.view(q.shape[0], -1, *q.shape[-2:])
                                if enable_mla:
                                    # [b, 2, sk//2, np, hn] -> [b, sk//2, np, hn]
                                    k_part = k_part[:, 0, ...]
                                    v_part = v_part[:, 0, ...]
                                else:
                                    # [b, 2, sk//2, 2, np, hn] -> [b, sk//2, 2, np, hn]
                                    kv_inputs[i % 2] = kv_inputs[i % 2][:, 0, ...]
                            elif qkv_format == "sbhd":
                                # [2, sq//2, b, np, hn] -> [sq, b, np, hn]
                                q_inputs[i % 2] = q.view(-1, *q.shape[-3:])
                                if enable_mla:
                                    # [2, sk//2, b, np, hn] -> [sk//2, b, np, hn]
                                    k_part = k_part[0]
                                    v_part = v_part[0]
                                else:
                                    # [2, sk//2, b, 2, np, hn] -> [sk//2, b, 2, np, hn]
                                    kv_inputs[i % 2] = kv_inputs[i % 2][0]
                            elif qkv_format == "thd":
                                q_inputs[i % 2] = q
                                if enable_mla:
                                    # [t, np, hn] -> [t/2, np, hn]
                                    k_part = tex.thd_read_half_tensor(
                                        k_part, cu_seqlens_kv_padded, 0
                                    )
                                    v_part = tex.thd_read_half_tensor(
                                        v_part, cu_seqlens_kv_padded, 0
                                    )
                                else:
                                    # [2, t, np, hn] -> [2, t/2, np, hn]
                                    kv_inputs[i % 2] = tex.thd_read_half_tensor(
                                        kv_inputs[i % 2], cu_seqlens_kv_padded, 0
                                    )
                            if use_fused_attention:
                                if enable_mla:
                                    k_part = k_part.contiguous()
                                    v_part = v_part.contiguous()
                                else:
                                    kv_inputs[i % 2] = kv_inputs[i % 2].contiguous()
                                if attn_bias is not None:
                                    idx = (rank - i) % cp_size
                                    attn_bias_inputs[i % 2] = attn_bias[
                                        ..., idx, :
                                    ].contiguous()

                                q_part = q_inputs[i % 2]
                                if not enable_mla:
                                    k_part = (
                                        kv_inputs[i % 2][..., 0, :, :]
                                        if qkv_format in ["bshd", "sbhd"]
                                        else kv_inputs[i % 2][0]
                                    )
                                    v_part = (
                                        kv_inputs[i % 2][..., 1, :, :]
                                        if qkv_format in ["bshd", "sbhd"]
                                        else kv_inputs[i % 2][1]
                                    )
                                fp8_meta_kwargs = {}
                                if fp8:
                                    q_part = QKV_quantizer.create_tensor_from_data(
                                        q_part, fake_dtype=qkv_dtype, internal=True
                                    )
                                    k_part = QKV_quantizer.create_tensor_from_data(
                                        k_part, fake_dtype=qkv_dtype, internal=True
                                    )
                                    v_part = QKV_quantizer.create_tensor_from_data(
                                        v_part, fake_dtype=qkv_dtype, internal=True
                                    )
                                    fp8_meta_kwargs["s_quantizer"] = (
                                        S_quantizer_per_step[i]
                                    )
                                    fp8_meta_kwargs["o_quantizer"] = (
                                        O_CP_quantizer_per_step[i]
                                    )
                                new_attn_mask_type = "padding" if padding else "no_mask"
                                out_per_step[i], aux_ctx_tensors = fused_attn_fwd(
                                    is_training,
                                    max_seqlen_q,
                                    max_seqlen_kv // 2,
                                    cu_seqlens_q_per_step[i],
                                    cu_seqlens_kv_per_step[i],
                                    q_part,
                                    k_part,
                                    v_part,
                                    qkv_dtype,
                                    fused_attn_backend,
                                    attn_scale=softmax_scale,
                                    dropout=dropout_p,
                                    qkv_layout=qkv_layout,
                                    attn_mask_type=new_attn_mask_type,
                                    attn_bias_type=attn_bias_type,
                                    attn_bias=attn_bias_inputs[i % 2],
                                    cu_seqlens_q_padded=cu_seqlens_q_padded,
                                    cu_seqlens_kv_padded=(
                                        None
                                        if cu_seqlens_kv_padded is None
                                        else cu_seqlens_kv_padded // 2
                                    ),
                                    **fp8_meta_kwargs,
                                )
                                if fp8:
                                    softmax_lse_per_step[i], _, rng_states[i] = (
                                        aux_ctx_tensors
                                    )
                                else:
                                    softmax_lse_per_step[i], rng_states[i], *rest = (
                                        aux_ctx_tensors
                                    )
                                    attn_biases[i] = rest[0] if len(rest) > 0 else None
                            else:
                                fa_forward_args_thd = get_fa_args(
                                    True,
                                    use_flash_attn_3,
                                    qkv_format,
                                    cu_seqlens_q=cu_seqlens_q_per_step[i],
                                    cu_seqlens_kv=cu_seqlens_kv_per_step[i],
                                    max_seqlen_q=max_seqlen_q,
                                    max_seqlen_kv=max_seqlen_kv // 2,
                                )
                                if use_flash_attn_3 or (
                                    fa_utils.v2_3_plus and not fa_utils.v2_7_0_plus
                                ):
                                    fa_forward_kwargs["window_size"] = (-1, -1)
                                elif fa_utils.v2_7_0_plus:
                                    fa_forward_kwargs["window_size_left"] = -1
                                    fa_forward_kwargs["window_size_right"] = -1
                                # Need to add MLA support once Flash Attention supports MLA
                                nvtx_range_push(suffix=f"flash_attn_fwd.causal.step{i}")
                                fa_outputs = flash_attn_fwd(
                                    q_inputs[i % 2],
                                    (
                                        kv_inputs[i % 2][..., 0, :, :]
                                        if qkv_format in ["bshd", "sbhd"]
                                        else kv_inputs[i % 2][0]
                                    ),
                                    (
                                        kv_inputs[i % 2][..., 1, :, :]
                                        if qkv_format in ["bshd", "sbhd"]
                                        else kv_inputs[i % 2][1]
                                    ),
                                    *fa_forward_args_thd,
                                    causal=False,
                                    **fa_forward_kwargs,
                                )
                                nvtx_range_pop(suffix=f"flash_attn_fwd.causal.step{i}")
                                if not fa_utils.v2_7_0_plus:
                                    out_per_step[i] = fa_outputs[4]
                                    softmax_lse_per_step[i] = fa_outputs[5]
                                    if not use_flash_attn_3:
                                        rng_states[i] = fa_outputs[7]
                                else:
                                    out_per_step[i] = fa_outputs[0]
                                    softmax_lse_per_step[i] = fa_outputs[1]
                                    if not use_flash_attn_3:
                                        rng_states[i] = fa_outputs[3]
                        else:
                            if pad_between_seqs:
                                cu_seqlens_q_per_step[i] = get_cu_seqlens_on_cp_rank(
                                    cu_seqlens_q,
                                    cu_seqlens_q_padded,
                                    cp_size,
                                    rank,
                                    False,
                                    True,
                                )
                                cu_seqlens_kv_per_step[i] = get_cu_seqlens_on_cp_rank(
                                    cu_seqlens_kv,
                                    cu_seqlens_kv_padded,
                                    cp_size,
                                    (rank - i) % cp_size,
                                    True,
                                    True,
                                )
                            elif qkv_format == "thd":
                                cu_seqlens_q_per_step[i] = cu_seqlens_q // (cp_size * 2)
                                cu_seqlens_kv_per_step[i] = cu_seqlens_kv // cp_size
                            else:
                                cu_seqlens_q_per_step[i] = cu_seqlens_q_half
                                cu_seqlens_kv_per_step[i] = cu_seqlens_kv
                            if qkv_format == "bshd":
                                # [b, 2, sq//2, np, hn] -> [b, sq//2, np, hn]
                                q_inputs[i % 2] = q[:, 1, ...]
                                if enable_mla:
                                    # [b, 2, sk//2, np, hn] -> [b, sk, np, hn]
                                    k_part = k_part.view(
                                        k_part.shape[0], -1, *k_part.shape[-2:]
                                    )
                                    v_part = v_part.view(
                                        v_part.shape[0], -1, *v_part.shape[-2:]
                                    )
                                else:
                                    # [b, 2, sk//2, 2, np, hn] -> [b, sk, 2, np, hn]
                                    kv_inputs[i % 2] = kv_inputs[i % 2].view(
                                        k.shape[0], -1, 2, *k.shape[-2:]
                                    )
                            elif qkv_format == "sbhd":
                                # [2, sq//2, b, np, hn] -> [sq//2, b, np, hn]
                                q_inputs[i % 2] = q[1]
                                if enable_mla:
                                    # [2, sk//2, b, np, hn] -> [sk, b, np, hn]
                                    k_part = k_part.view(-1, *k_part.shape[2:])
                                    v_part = v_part.view(-1, *v_part.shape[2:])
                                else:
                                    # [2, sk//2, b, 2, np, hn] -> [sk, b, 2, np, hn]
                                    kv_inputs[i % 2] = kv_inputs[i % 2].view(
                                        -1, k.shape[2], 2, *k.shape[-2:]
                                    )
                            elif qkv_format == "thd":
                                # [t, np, hn] -> [t/2, np, hn]
                                q_inputs[i % 2] = tex.thd_read_half_tensor(
                                    q, cu_seqlens_q_padded, 1
                                )
                            if use_fused_attention:
                                q_inputs[i % 2] = q_inputs[i % 2].contiguous()
                                if attn_bias is not None:
                                    idx = (rank - i) % cp_size
                                    attn_bias_inputs[i % 2] = torch.cat(
                                        (
                                            attn_bias_[..., 1, :, idx, :],
                                            attn_bias_[
                                                ..., 1, :, (2 * cp_size - idx - 1), :
                                            ],
                                        ),
                                        dim=-1,
                                    ).contiguous()

                                q_part = q_inputs[i % 2]
                                if not enable_mla:
                                    k_part = (
                                        kv_inputs[i % 2][..., 0, :, :]
                                        if qkv_format in ["bshd", "sbhd"]
                                        else kv_inputs[i % 2][0]
                                    )
                                    v_part = (
                                        kv_inputs[i % 2][..., 1, :, :]
                                        if qkv_format in ["bshd", "sbhd"]
                                        else kv_inputs[i % 2][1]
                                    )
                                fp8_meta_kwargs = {}
                                if fp8:
                                    q_part = QKV_quantizer.create_tensor_from_data(
                                        q_part, fake_dtype=qkv_dtype, internal=True
                                    )
                                    k_part = QKV_quantizer.create_tensor_from_data(
                                        k_part, fake_dtype=qkv_dtype, internal=True
                                    )
                                    v_part = QKV_quantizer.create_tensor_from_data(
                                        v_part, fake_dtype=qkv_dtype, internal=True
                                    )
                                    fp8_meta_kwargs["s_quantizer"] = (
                                        S_quantizer_per_step[i]
                                    )
                                    fp8_meta_kwargs["o_quantizer"] = (
                                        O_CP_quantizer_per_step[i]
                                    )
                                out_per_step[i], aux_ctx_tensors = fused_attn_fwd(
                                    is_training,
                                    max_seqlen_q // 2,
                                    max_seqlen_kv,
                                    cu_seqlens_q_per_step[i],
                                    cu_seqlens_kv_per_step[i],
                                    q_part,
                                    k_part,
                                    v_part,
                                    qkv_dtype,
                                    fused_attn_backend,
                                    attn_scale=softmax_scale,
                                    dropout=dropout_p,
                                    qkv_layout=qkv_layout,
                                    attn_mask_type="padding" if padding else "no_mask",
                                    attn_bias_type=attn_bias_type,
                                    attn_bias=attn_bias_inputs[i % 2],
                                    cu_seqlens_q_padded=(
                                        None
                                        if cu_seqlens_q_padded is None
                                        else cu_seqlens_q_padded // 2
                                    ),
                                    cu_seqlens_kv_padded=cu_seqlens_kv_padded,
                                    **fp8_meta_kwargs,
                                )
                                if fp8:
                                    softmax_lse_per_step[i], _, rng_states[i] = (
                                        aux_ctx_tensors
                                    )
                                else:
                                    softmax_lse_per_step[i], rng_states[i], *rest = (
                                        aux_ctx_tensors
                                    )
                                    attn_biases[i] = rest[0] if len(rest) > 0 else None
                            else:
                                fa_forward_args_thd = get_fa_args(
                                    True,
                                    use_flash_attn_3,
                                    qkv_format,
                                    cu_seqlens_q=cu_seqlens_q_per_step[i],
                                    cu_seqlens_kv=cu_seqlens_kv_per_step[i],
                                    max_seqlen_q=max_seqlen_q // 2,
                                    max_seqlen_kv=max_seqlen_kv,
                                )
                                if use_flash_attn_3 or (
                                    fa_utils.v2_3_plus and not fa_utils.v2_7_0_plus
                                ):
                                    fa_forward_kwargs["window_size"] = (-1, -1)
                                elif fa_utils.v2_7_0_plus:
                                    fa_forward_kwargs["window_size_left"] = -1
                                    fa_forward_kwargs["window_size_right"] = -1
                                # Need to add MLA support once Flash Attention supports MLA
                                nvtx_range_push(suffix=f"flash_attn_fwd.causal.step{i}")
                                fa_outputs = flash_attn_fwd(
                                    q_inputs[i % 2],
                                    (
                                        kv_inputs[i % 2][..., 0, :, :]
                                        if qkv_format in ["bshd", "sbhd"]
                                        else kv_inputs[i % 2][0]
                                    ),
                                    (
                                        kv_inputs[i % 2][..., 1, :, :]
                                        if qkv_format in ["bshd", "sbhd"]
                                        else kv_inputs[i % 2][1]
                                    ),
                                    *fa_forward_args_thd,
                                    causal=False,
                                    **fa_forward_kwargs,
                                )
                                nvtx_range_pop(suffix=f"flash_attn_fwd.causal.step{i}")
                                if not fa_utils.v2_7_0_plus:
                                    out_per_step[i] = fa_outputs[4]
                                    softmax_lse_per_step[i] = fa_outputs[5]
                                    if not use_flash_attn_3:
                                        rng_states[i] = fa_outputs[7]
                                else:
                                    out_per_step[i] = fa_outputs[0]
                                    softmax_lse_per_step[i] = fa_outputs[1]
                                    if not use_flash_attn_3:
                                        rng_states[i] = fa_outputs[3]
                    else:
                        if pad_between_seqs:
                            cu_seqlens_q_per_step[i] = get_cu_seqlens_on_cp_rank(
                                cu_seqlens_q,
                                cu_seqlens_q_padded,
                                cp_size,
                                rank,
                                True,
                                True,
                            )
                            cu_seqlens_kv_per_step[i] = get_cu_seqlens_on_cp_rank(
                                cu_seqlens_kv,
                                cu_seqlens_kv_padded,
                                cp_size,
                                (rank - i) % cp_size,
                                True,
                                True,
                            )
                        elif qkv_format == "thd":
                            cu_seqlens_q_per_step[i] = cu_seqlens_q // cp_size
                            cu_seqlens_kv_per_step[i] = cu_seqlens_kv // cp_size
                        else:
                            cu_seqlens_q_per_step[i] = cu_seqlens_q
                            cu_seqlens_kv_per_step[i] = cu_seqlens_kv
                        if use_fused_attention:
                            if attn_bias is not None:
                                idx = (rank - i) % cp_size
                                attn_bias_inputs[i % 2] = torch.cat(
                                    (
                                        attn_bias[..., idx, :],
                                        attn_bias[..., (2 * cp_size - idx - 1), :],
                                    ),
                                    dim=-1,
                                ).contiguous()

                            q_part = q
                            if not enable_mla:
                                k_part = (
                                    kv_inputs[i % 2][..., 0, :, :]
                                    if qkv_format in ["bshd", "sbhd"]
                                    else kv_inputs[i % 2][0]
                                )
                                v_part = (
                                    kv_inputs[i % 2][..., 1, :, :]
                                    if qkv_format in ["bshd", "sbhd"]
                                    else kv_inputs[i % 2][1]
                                )
                            fp8_meta_kwargs = {}
                            if fp8:
                                q_part = QKV_quantizer.create_tensor_from_data(
                                    q_part, fake_dtype=qkv_dtype, internal=True
                                )
                                k_part = QKV_quantizer.create_tensor_from_data(
                                    k_part, fake_dtype=qkv_dtype, internal=True
                                )
                                v_part = QKV_quantizer.create_tensor_from_data(
                                    v_part, fake_dtype=qkv_dtype, internal=True
                                )
                                fp8_meta_kwargs["s_quantizer"] = S_quantizer_per_step[i]
                                fp8_meta_kwargs["o_quantizer"] = (
                                    O_CP_quantizer_per_step[i]
                                )
                            out_per_step[i], aux_ctx_tensors = fused_attn_fwd(
                                is_training,
                                max_seqlen_q,
                                max_seqlen_kv,
                                cu_seqlens_q_per_step[i],
                                cu_seqlens_kv_per_step[i],
                                q_part,
                                k_part,
                                v_part,
                                qkv_dtype,
                                fused_attn_backend,
                                attn_scale=softmax_scale,
                                dropout=dropout_p,
                                qkv_layout=qkv_layout,
                                attn_mask_type=attn_mask_type,
                                attn_bias_type=attn_bias_type,
                                attn_bias=attn_bias_inputs[i % 2],
                                cu_seqlens_q_padded=cu_seqlens_q_padded,
                                cu_seqlens_kv_padded=cu_seqlens_kv_padded,
                                **fp8_meta_kwargs,
                            )
                            if fp8:
                                softmax_lse_per_step[i], _, rng_states[i] = (
                                    aux_ctx_tensors
                                )
                            else:
                                softmax_lse_per_step[i], rng_states[i], *rest = (
                                    aux_ctx_tensors
                                )
                                attn_biases[i] = rest[0] if len(rest) > 0 else None
                        else:
                            fa_forward_args_thd = get_fa_args(
                                True,
                                use_flash_attn_3,
                                qkv_format,
                                cu_seqlens_q=cu_seqlens_q_per_step[i],
                                cu_seqlens_kv=cu_seqlens_kv_per_step[i],
                                max_seqlen_q=max_seqlen_q,
                                max_seqlen_kv=max_seqlen_kv,
                            )
                            # Need to add MLA support once Flash Attention supports MLA
                            nvtx_range_push(suffix=f"flash_attn_fwd.non-causal.step{i}")
                            fa_outputs = flash_attn_fwd(
                                q,
                                (
                                    kv_inputs[i % 2][..., 0, :, :]
                                    if qkv_format in ["bshd", "sbhd"]
                                    else kv_inputs[i % 2][0]
                                ),
                                (
                                    kv_inputs[i % 2][..., 1, :, :]
                                    if qkv_format in ["bshd", "sbhd"]
                                    else kv_inputs[i % 2][1]
                                ),
                                *fa_forward_args_thd,
                                causal=False,
                                **fa_forward_kwargs,
                            )
                            nvtx_range_pop(suffix=f"flash_attn_fwd.non-causal.step{i}")
                            if not fa_utils.v2_7_0_plus:
                                out_per_step[i] = fa_outputs[4]
                                softmax_lse_per_step[i] = fa_outputs[5]
                                if not use_flash_attn_3:
                                    rng_states[i] = fa_outputs[7]
                            else:
                                out_per_step[i] = fa_outputs[0]
                                softmax_lse_per_step[i] = fa_outputs[1]
                                if not use_flash_attn_3:
                                    rng_states[i] = fa_outputs[3]
                    nvtx_range_pop(suffix=f"cp_attn.stream_step{i}.compute")

            if i > 0:
                # wait until fwd restuls correction of last step is done
                if i > 1:
                    nvtx_range_push(suffix=f"cp_attn.wait_correction_event.idx{i}")
                    flash_attn_streams[(i - 1) % 2].wait_event(
                        fwd_results_correction_done
                    )
                    nvtx_range_pop(suffix=f"cp_attn.wait_correction_event.idx{i}")

                with torch.cuda.stream(flash_attn_streams[(i - 1) % 2]):
                    nvtx_range_push(suffix=f"cp_attn.stream_idx{i-1}.lse_correction")
                    if use_fused_attention:
                        # [b, np, sq, 1] -> [b, np, sq] or
                        # [t, np, 1] -> [t, np]
                        softmax_lse_per_step[i - 1].squeeze_(-1)
                        if softmax_lse_in_packed_format:
                            softmax_lse_per_step[i - 1] = (
                                softmax_lse_per_step[i - 1].transpose(0, 1).contiguous()
                            )
                    if fp8:
                        out_per_step[i - 1] = out_per_step[i - 1].dequantize(
                            dtype=torch.float32
                        )
                    if i == 1:
                        softmax_lse = torch.clone(softmax_lse_per_step[0])
                        if qkv_format == "thd":
                            if enable_mla:
                                out = torch.zeros_like(
                                    v if not fp8 else out_per_step[0]
                                ).view(v_shape)
                            else:
                                # MHA or GQA
                                out = torch.zeros_like(
                                    q if not fp8 else out_per_step[0]
                                ).view(q.shape)
                    elif (i - 1) <= rank or not causal:
                        flash_attn_fwd_softmax_lse_correction(
                            softmax_lse, softmax_lse_per_step[i - 1]
                        )
                    else:
                        if qkv_format == "thd":
                            tex.thd_second_half_lse_correction(
                                softmax_lse,
                                softmax_lse_per_step[i - 1],
                                cu_seqlens_q_padded,
                                softmax_lse_in_packed_format,
                            )
                        else:
                            flash_attn_fwd_second_half_softmax_lse_correction(
                                softmax_lse.view(*softmax_lse.shape[:-1], 2, -1),
                                softmax_lse_per_step[i - 1],
                            )
                    nvtx_range_pop(suffix=f"cp_attn.stream_idx{i-1}.lse_correction")

                if i < cp_size:
                    flash_attn_streams[(i - 1) % 2].record_event(
                        fwd_results_correction_done
                    )

        torch.cuda.current_stream().wait_stream(flash_attn_streams[1])

        second_half_lse_seqlen = None
        if causal and rank < (cp_size - 1):
            second_half_lse_seqlen = softmax_lse_per_step[-1].shape[-1]

        for i in range(cp_size):
            if i <= rank or not causal:
                if qkv_format in ["bshd", "sbhd"]:
                    if i == 0:
                        out = flash_attn_fwd_out_correction_init(
                            out_per_step[0],
                            softmax_lse,
                            softmax_lse_per_step[0],
                            seq_dim,
                        )
                        if enable_mla:
                            out = out.view(v_shape)
                        else:
                            out = out.view(q.shape)
                    else:
                        flash_attn_fwd_out_correction(
                            out.view(*out_per_step[i].shape),
                            out_per_step[i],
                            softmax_lse,
                            softmax_lse_per_step[i],
                            seq_dim,
                        )
                elif qkv_format == "thd":
                    tex.thd_out_correction(
                        out,
                        out_per_step[i],
                        softmax_lse,
                        softmax_lse_per_step[i],
                        cu_seqlens_q_padded,
                        False,
                        softmax_lse_in_packed_format,
                    )
            else:
                if qkv_format in ["bshd", "sbhd"]:
                    flash_attn_fwd_second_half_out_correction(
                        out,
                        out_per_step[i],
                        softmax_lse,
                        softmax_lse_per_step[i],
                        seq_dim,
                    )
                elif qkv_format == "thd":
                    tex.thd_out_correction(
                        out,
                        out_per_step[i],
                        softmax_lse,
                        softmax_lse_per_step[i],
                        cu_seqlens_q_padded,
                        True,
                        softmax_lse_in_packed_format,
                    )

        kv = p2p_comm_buffers[-1]
        if qkv_format == "bshd":
            out = out.view(out.shape[0], -1, *out.shape[-2:])
            ctx.batch_size = out.shape[0]
        elif qkv_format == "sbhd":
            out = out.view(-1, *out.shape[-3:])
            ctx.batch_size = out.shape[1]

        if cp_size_a2a > 1:
            chunk_ids_for_a2a = get_seq_chunk_ids_for_reordering_after_attn(
                cp_size_a2a, out.device
            )
            out = flash_attn_a2a_communicate(
                out,
                chunk_ids_for_a2a,
                seq_dim,
                cp_size_a2a,
                cp_group_a2a,
                cp_stream,
                False,
            )
            if use_fused_attention:
                if qkv_format == "bshd":
                    # [b*s, np, hn] -> [b, s, np, hn]
                    out = out.view(ctx.batch_size, -1, *out.shape[-2:])
                elif qkv_format == "sbhd":
                    # [s*b, np, hn] -> [s, b, np, hn]
                    out = out.view(-1, ctx.batch_size, *out.shape[-2:])
        elif not use_fused_attention:
            out = out.view(-1, *out.shape[-2:])

        if fp8 and use_fused_attention:
            amax_cp_fwd = amax_per_step.amax(dim=1)
            S_quantizer.amax.copy_(amax_cp_fwd[0])
            O_CP_quantizer.amax.copy_(amax_cp_fwd[1])

        out_fp8 = None
        out_f16 = out.to(qkv_dtype)

        if fp8 and (is_output_fp8 or int(os.getenv("NVTE_FP8_DPA_BWD", "1"))):
            out_fp8 = O_quantizer(out_f16)  # final result

        out_ret = out_fp8 if (fp8 and is_output_fp8) else out_f16

        if fp8 and int(os.getenv("NVTE_FP8_DPA_BWD", "1")):
            q_save, kv_save, out_save = q, kv, out_fp8._data
        elif fp8 and is_input_fp8:
            q_save, kv_save, out_save = q, kv, out_f16
        else:
            q_f16 = q_f16.view(q.shape)
            q_save, kv_save, out_save = q_f16, kv, out_f16

        tensors_to_save, tensor_objects = prepare_for_saving(
            q_save,
            kv_save,
            out_save,
            softmax_lse,
            cu_seqlens_q_padded,
            cu_seqlens_kv_padded,
            *cu_seqlens_q_per_step,
            *cu_seqlens_kv_per_step,
            *rng_states,
            *attn_biases,
        )
        ctx.save_for_backward(*tensors_to_save)
        ctx.tensor_objects = tensor_objects

        ctx.cp_group_a2a = cp_group_a2a
        ctx.cp_size_a2a = cp_size_a2a
        ctx.rank_a2a = rank_a2a
        ctx.cp_group = cp_group
        ctx.cp_global_ranks = cp_global_ranks
        ctx.cp_stream = cp_stream
        ctx.dropout_p = dropout_p
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_kv = max_seqlen_kv
        ctx.softmax_scale = softmax_scale
        ctx.qkv_format = qkv_format
        ctx.attn_mask_type = attn_mask_type
        ctx.attn_bias_type = attn_bias_type
        ctx.attn_bias_shape = None if attn_bias is None else attn_bias.shape
        ctx.deterministic = deterministic
        ctx.use_fused_attention = use_fused_attention
        ctx.softmax_lse_in_packed_format = softmax_lse_in_packed_format
        ctx.second_half_lse_seqlen = second_half_lse_seqlen
        ctx.fp8 = fp8 and int(os.getenv("NVTE_FP8_DPA_BWD", "1"))
        ctx.fp8_meta = fp8_meta
        ctx.is_input_fp8 = is_input_fp8
        ctx.is_output_fp8 = is_output_fp8
        ctx.use_flash_attn_3 = use_flash_attn_3

        ctx.enable_mla = enable_mla
        if enable_mla:
            ctx.k_numel = k_numel
            ctx.k_shape = k_shape
            ctx.v_shape = v_shape

        ctx.qkv_dtype = qkv_dtype
        ctx.dQKV_quantizer = dQKV_quantizer
        ctx.dQKV_CP_quantizer = dQKV_CP_quantizer
        ctx.dO_quantizer = dO_quantizer
        ctx.dP_quantizer = dP_quantizer
        ctx.QKV_quantizer = QKV_quantizer
        ctx.O_quantizer = O_quantizer
        ctx.S_quantizer = S_quantizer
        if ctx.fp8:
            ctx.QKV_quantizer = QKV_quantizer.copy()
            ctx.QKV_quantizer.scale = QKV_quantizer.scale.clone()
            ctx.O_quantizer = O_quantizer.copy()
            ctx.O_quantizer.scale = O_quantizer.scale.clone()
            ctx.S_quantizer = S_quantizer.copy()
            ctx.S_quantizer.scale = S_quantizer.scale.clone()
        nvtx_range_pop("transformer_engine.AttnFuncWithCPAndKVP2P.forward")

        return out_ret

    @staticmethod
    def backward(ctx, dout):
        # pylint: disable=missing-function-docstring
        nvtx_range_push("transformer_engine.AttnFuncWithCPAndKVP2P.backward")
        cp_size_a2a = ctx.cp_size_a2a
        rank_a2a = ctx.rank_a2a

        cp_size = get_distributed_world_size(ctx.cp_group)
        rank = get_distributed_rank(ctx.cp_group)
        send_dst = ctx.cp_global_ranks[(rank - 1) % cp_size * cp_size_a2a + rank_a2a]
        recv_src = ctx.cp_global_ranks[(rank + 1) % cp_size * cp_size_a2a + rank_a2a]
        device_compute_capability = get_device_compute_capability()
        batch_p2p_comm = int(os.getenv("NVTE_BATCH_MHA_P2P_COMM", "0")) or (
            device_compute_capability < (10, 0) and cp_size == 2
        )

        (
            q,
            kv,
            out,
            softmax_lse,
            cu_seqlens_q_padded,
            cu_seqlens_kv_padded,
            *other_tensors,
        ) = restore_from_saved(ctx.tensor_objects, ctx.saved_tensors)
        cu_seqlens_q_per_step = other_tensors[:cp_size]
        cu_seqlens_kv_per_step = other_tensors[cp_size : cp_size * 2]
        rng_states = other_tensors[cp_size * 2 : cp_size * 3]
        attn_biases = other_tensors[cp_size * 3 : cp_size * 4]

        causal = "causal" in ctx.attn_mask_type
        padding = "padding" in ctx.attn_mask_type

        seq_dim = None
        if ctx.qkv_format in ["bshd", "sbhd"]:
            seq_dim = ctx.qkv_format.index("s")
            if ctx.enable_mla:
                qkv_layout = (
                    ctx.qkv_format + "_" + ctx.qkv_format + "_" + ctx.qkv_format
                )
            else:
                qkv_layout = (
                    ctx.qkv_format
                    + "_"
                    + ctx.qkv_format[:-2]
                    + "2"
                    + ctx.qkv_format[-2:]
                )
        else:
            qkv_layout = ctx.qkv_format + "_" + ctx.qkv_format + "_" + ctx.qkv_format

        if attn_biases[0] is not None:
            # [b, np, sq, 2*cp, sk//(2*cp)]
            attn_dbias = torch.zeros(
                *ctx.attn_bias_shape,
                dtype=attn_biases[0].dtype,
                device=attn_biases[0].device,
            )
            # [b, np, sq, 2*cp, sk//(2*cp)] -> [b, np, 2, sq//2, 2*cp, sk//(2*cp)]
            attn_dbias_ = attn_dbias.view(
                *attn_dbias.shape[:-3],
                2,
                attn_dbias.shape[-3] // 2,
                *attn_dbias.shape[-2:],
            )
        else:
            attn_dbias = None
            attn_dbias_ = None

        softmax_lse_ = None
        if causal and ctx.second_half_lse_seqlen is not None:
            if ctx.qkv_format == "thd":
                softmax_lse_ = tex.thd_read_second_half_lse(
                    softmax_lse,
                    cu_seqlens_q_padded,
                    ctx.softmax_lse_in_packed_format,
                    ctx.second_half_lse_seqlen,
                )
            else:
                # [b, np, sq] -> [b, np, 2, sq//2]
                softmax_lse_ = softmax_lse.view(*softmax_lse.shape[:-1], 2, -1)
                softmax_lse_ = softmax_lse_[..., 1, :].contiguous()
            if ctx.use_fused_attention:
                if ctx.softmax_lse_in_packed_format:
                    softmax_lse_ = softmax_lse_.transpose(0, 1).contiguous()
                # [b, np, sq//2] -> [b, np, sq//2, 1] or
                # [t//2, np] -> [t//2, np, 1]
                softmax_lse_.unsqueeze_(-1)
        if ctx.use_fused_attention:
            if ctx.softmax_lse_in_packed_format:
                softmax_lse = softmax_lse.transpose(0, 1).contiguous()
            # [b, np, sq] -> [b, np, sq, 1] or
            # [t, np] -> [t, np, 1]
            softmax_lse.unsqueeze_(-1)
            dout = dout.contiguous()

        dq = None
        dout_dtype = dout.dtype
        fused_attn_backend = None
        fused_attn_dqkv_dtype = None
        amax_per_step = None
        dP_quantizer_per_step = [None for _ in range(cp_size)]
        dQKV_CP_quantizer_per_step = [None for _ in range(cp_size)]
        if ctx.fp8:
            if ctx.use_fused_attention:
                fused_attn_backend = FusedAttnBackend["FP8"]

                if ctx.is_output_fp8:
                    assert isinstance(
                        dout, Float8Tensor
                    ), "dout must be Float8Tensors for FP8 MHA!"
                    ctx.dO_quantizer = dout._quantizer
                else:
                    dout = ctx.dO_quantizer(dout)
                fused_attn_dqkv_dtype = TE_DType[dout._data.dtype]
                dq_fp8 = torch.empty(
                    (cp_size, *q.shape), dtype=dout._data.dtype, device=q.device
                )
                dkv_fp8 = torch.empty(
                    (cp_size, *kv.shape), dtype=dout._data.dtype, device=kv.device
                )
                dkv_fp8_ = torch.empty_like(dkv_fp8)
                p2p_comm_buffers = [[kv, dkv_fp8], [torch.empty_like(kv), dkv_fp8_]]
                dout = dout._data
                fp8_meta_kwargs = {}
                fp8_meta_kwargs["s_quantizer"] = ctx.S_quantizer
                amax_per_step = torch.zeros(
                    (2, cp_size), dtype=torch.float32, device=q.device
                )
                for i in range(cp_size):
                    dP_quantizer_per_step[i] = ctx.dP_quantizer.copy()
                    dP_quantizer_per_step[i].amax = amax_per_step[0][i].reshape((1,))
                    dQKV_CP_quantizer_per_step[i] = ctx.dQKV_CP_quantizer.copy()
                    dQKV_CP_quantizer_per_step[i].amax = amax_per_step[1][i].reshape(
                        (1,)
                    )
            else:
                assert False, "FP8 is only supported with Fused Attention!"
        else:
            if ctx.fp8_meta is not None:
                if ctx.is_input_fp8:
                    q = ctx.QKV_quantizer.create_tensor_from_data(
                        q, fake_dtype=ctx.qkv_dtype, internal=True
                    )
                    kv = ctx.QKV_quantizer.create_tensor_from_data(
                        kv, fake_dtype=ctx.qkv_dtype, internal=True
                    )
                    q = q.dequantize(dtype=ctx.qkv_dtype)
                    kv = kv.dequantize(dtype=ctx.qkv_dtype)
                if ctx.is_output_fp8:
                    assert isinstance(
                        dout, Float8Tensor
                    ), "dout must be Float8Tensors for FP8 MHA!"
                    if cp_size_a2a == 1:
                        dout = dout.dequantize(dtype=dout_dtype)
                    else:
                        ctx.dO_quantizer = dout._quantizer
                        dout = dout._data
            dq = torch.empty_like(q)
            p2p_comm_buffers = [
                torch.empty((2, *kv.shape), dtype=kv.dtype, device=kv.device),
                torch.empty((2, *kv.shape), dtype=kv.dtype, device=kv.device),
            ]
            p2p_comm_buffers[0][0].copy_(kv)
            if ctx.use_fused_attention:
                fp8_meta_kwargs = {}
                fused_attn_dqkv_dtype = TE_DType[dout_dtype]
                fused_attn_backend = FusedAttnBackend["F16_arbitrary_seqlen"]

        if cp_size_a2a > 1:
            if not ctx.use_fused_attention:
                out = out.view(ctx.batch_size, -1, *out.shape[-2:])
                dout = dout.view(*out.shape)
            chunk_ids_for_a2a = get_seq_chunk_ids_for_reordering_before_attn(
                cp_size_a2a, out.device
            )
            out, dout = flash_attn_a2a_communicate(
                [out, dout],
                chunk_ids_for_a2a,
                seq_dim,
                cp_size_a2a,
                ctx.cp_group_a2a,
                ctx.cp_stream,
                True,
            )
            if not ctx.fp8 and ctx.fp8_meta is not None and ctx.is_output_fp8:
                dout = ctx.dO_quantizer.create_tensor_from_data(
                    dout, fake_dtype=dout_dtype, internal=True
                )
                dout = dout.dequantize(dtype=dout_dtype)

        if ctx.enable_mla:
            out = out.view(*ctx.v_shape)
            dout = dout.view(*ctx.v_shape)
        else:
            # MHA or GQA
            out = out.view(*q.shape)
            dout = dout.view(*q.shape)
        send_recv_reqs = []

        flash_attn_bwd = None
        if not ctx.use_fused_attention:
            fa_backward_kwargs = {"softmax_scale": ctx.softmax_scale}
            if ctx.use_flash_attn_3:
                from transformer_engine.pytorch.attention.dot_product_attention.backends import (
                    _flash_attn_bwd_v3,
                )

                flash_attn_bwd = _flash_attn_bwd_v3  # pylint: disable=possibly-used-before-assignment
                fa_backward_kwargs["deterministic"] = ctx.deterministic
            else:
                if ctx.qkv_format == "thd":
                    from transformer_engine.pytorch.attention.dot_product_attention.backends import (
                        _flash_attn_varlen_bwd,
                    )

                    flash_attn_bwd = _flash_attn_varlen_bwd
                else:
                    from transformer_engine.pytorch.attention.dot_product_attention.backends import (
                        _flash_attn_bwd,
                    )

                    flash_attn_bwd = _flash_attn_bwd
                fa_backward_kwargs["dropout_p"] = ctx.dropout_p
                if fa_utils.v2_4_plus:
                    fa_backward_kwargs["alibi_slopes"] = None
                if fa_utils.v2_4_1_plus:
                    fa_backward_kwargs["deterministic"] = ctx.deterministic
                if fa_utils.v2_6_0_plus:
                    fa_backward_kwargs["softcap"] = 0.0

        for i in range(cp_size):
            # wait until KV is received
            for req in send_recv_reqs:
                req.wait()

            send_tensor = p2p_comm_buffers[i % 2]
            recv_tensor = p2p_comm_buffers[(i + 1) % 2]
            if ctx.fp8:
                if i < cp_size - 1:
                    send_recv_reqs = flash_attn_p2p_communicate(
                        rank,
                        send_tensor[0],
                        send_dst,
                        recv_tensor[0],
                        recv_src,
                        ctx.cp_group,
                        batch_p2p_comm,
                        i,
                    )
                else:
                    dkv_a2a_req = torch.distributed.all_to_all_single(
                        dkv_fp8,
                        dkv_fp8_,
                        group=ctx.cp_group,
                        async_op=True,
                    )
                    send_recv_reqs = [dkv_a2a_req]
            else:
                if i == 0:
                    send_tensor = send_tensor[0]
                    recv_tensor = recv_tensor[0]
                if i == (cp_size - 1):
                    send_tensor = send_tensor[1]
                    recv_tensor = recv_tensor[1]
                send_recv_reqs = flash_attn_p2p_communicate(
                    rank,
                    send_tensor,
                    send_dst,
                    recv_tensor,
                    recv_src,
                    ctx.cp_group,
                    batch_p2p_comm,
                    i,
                )

            kv = p2p_comm_buffers[i % 2][0]
            q_, kv_, out_, dout_ = None, None, None, None
            dq_, dk_, dv_ = None, None, None
            if ctx.enable_mla:
                k_part = kv[: ctx.k_numel].view(*ctx.k_shape)
                v_part = kv[ctx.k_numel :].view(*ctx.v_shape)
            # In reversed order of fwd
            if causal:
                if i == (cp_size - 1):
                    if ctx.qkv_format == "bshd":
                        # [b, 2, sq//2, np, hn] -> [b, sq, np, hn]
                        q_, out_, dout_ = [
                            x.view(x.shape[0], -1, *x.shape[-2:])
                            for x in [q, out, dout]
                        ]
                        if ctx.enable_mla:
                            # [b, 2, sk//2, np, hn] -> [b, sk, np, hn]
                            k_part = k_part.view(
                                k_part.shape[0], -1, *k_part.shape[-2:]
                            )
                            v_part = v_part.view(
                                v_part.shape[0], -1, *v_part.shape[-2:]
                            )
                        else:
                            # [b, 2, sk//2, 2, np, hn] -> [b, sk, 2, np, hn]
                            kv_ = kv.view(kv.shape[0], -1, *kv.shape[-3:])
                    elif ctx.qkv_format == "sbhd":
                        # [2, sq//2, b, np, hn] -> [sq, b, np, hn]
                        q_, out_, dout_ = [
                            x.view(-1, *x.shape[-3:]) for x in [q, out, dout]
                        ]
                        if ctx.enable_mla:
                            # [2, sk//2, b, np, hn] -> [sk, b, np, hn]
                            k_part = k_part.view(-1, *k_part.shape[-3:])
                            v_part = v_part.view(-1, *v_part.shape[-3:])
                        else:
                            # [2, sk//2, b, 2, np, hn] -> [sk, b, 2, np, hn]
                            kv_ = kv.view(-1, *kv.shape[-4:])
                    elif ctx.qkv_format == "thd":
                        q_, kv_, out_, dout_ = q, kv, out, dout
                    if ctx.use_fused_attention:
                        if ctx.fp8:
                            aux_ctx_tensors = [
                                softmax_lse,
                                softmax_lse,
                                rng_states[cp_size - i - 1],
                            ]
                        else:
                            aux_ctx_tensors = [softmax_lse, rng_states[cp_size - i - 1]]
                        if attn_dbias is not None:
                            aux_ctx_tensors += [attn_biases[cp_size - i - 1]]
                        q_part = q_
                        if not ctx.enable_mla:
                            k_part = (
                                kv_[..., 0, :, :]
                                if ctx.qkv_format in ["bshd", "sbhd"]
                                else kv_[0]
                            )
                            v_part = (
                                kv_[..., 1, :, :]
                                if ctx.qkv_format in ["bshd", "sbhd"]
                                else kv_[1]
                            )
                        out_part = out_
                        dout_part = dout_

                        if ctx.fp8:
                            q_part = ctx.QKV_quantizer.create_tensor_from_data(
                                q_part, fake_dtype=ctx.qkv_dtype, internal=True
                            )
                            k_part = ctx.QKV_quantizer.create_tensor_from_data(
                                k_part, fake_dtype=ctx.qkv_dtype, internal=True
                            )
                            v_part = ctx.QKV_quantizer.create_tensor_from_data(
                                v_part, fake_dtype=ctx.qkv_dtype, internal=True
                            )
                            out_part = ctx.O_quantizer.create_tensor_from_data(
                                out_part, fake_dtype=ctx.qkv_dtype, internal=True
                            )
                            dout_part = ctx.dO_quantizer.create_tensor_from_data(
                                dout_part, fake_dtype=dout_dtype, internal=True
                            )
                            fp8_meta_kwargs["dp_quantizer"] = dP_quantizer_per_step[i]
                            fp8_meta_kwargs["dqkv_quantizer"] = (
                                dQKV_CP_quantizer_per_step[i]
                            )
                        dq_, dk_, dv_, dbias_ = fused_attn_bwd(
                            ctx.max_seqlen_q,
                            ctx.max_seqlen_kv,
                            cu_seqlens_q_per_step[cp_size - i - 1],
                            cu_seqlens_kv_per_step[cp_size - i - 1],
                            q_part,
                            k_part,
                            v_part,
                            out_part,
                            dout_part,
                            dout_dtype,
                            fused_attn_dqkv_dtype,
                            aux_ctx_tensors,
                            fused_attn_backend,
                            cu_seqlens_q_padded=cu_seqlens_q_padded,
                            cu_seqlens_kv_padded=cu_seqlens_kv_padded,
                            attn_scale=ctx.softmax_scale,
                            dropout=ctx.dropout_p,
                            qkv_layout=qkv_layout,
                            attn_mask_type=ctx.attn_mask_type,
                            attn_bias_type=ctx.attn_bias_type,
                            deterministic=ctx.deterministic,
                            **fp8_meta_kwargs,
                        )
                        if ctx.fp8:
                            dq_ = dq_._data
                            dk_ = dk_._data
                            dv_ = dv_._data
                    else:
                        dq_ = torch.empty_like(q_)
                        dkv_ = torch.empty_like(kv_)
                        fa_backward_args_thd = get_fa_args(
                            False,
                            ctx.use_flash_attn_3,
                            ctx.qkv_format,
                            cu_seqlens_q=cu_seqlens_q_per_step[cp_size - i - 1],
                            cu_seqlens_kv=cu_seqlens_kv_per_step[cp_size - i - 1],
                            max_seqlen_q=ctx.max_seqlen_q,
                            max_seqlen_kv=ctx.max_seqlen_kv,
                            dq=dq_,
                            dk=(
                                dkv_[..., 0, :, :]
                                if ctx.qkv_format in ["bshd", "sbhd"]
                                else dkv_[0]
                            ),
                            dv=(
                                dkv_[..., 1, :, :]
                                if ctx.qkv_format in ["bshd", "sbhd"]
                                else dkv_[1]
                            ),
                        )
                        if ctx.use_flash_attn_3 or (
                            fa_utils.v2_3_plus and not fa_utils.v2_7_0_plus
                        ):
                            fa_backward_kwargs["window_size"] = (-1, 0)
                        elif fa_utils.v2_7_0_plus:
                            fa_backward_kwargs["window_size_left"] = -1
                            fa_backward_kwargs["window_size_right"] = 0
                        if not ctx.use_flash_attn_3:
                            fa_backward_kwargs["rng_state"] = rng_states[
                                cp_size - i - 1
                            ]
                        # Need to add MLA support once Flash Attention supports MLA
                        flash_attn_bwd(
                            dout_,
                            q_,
                            (
                                kv_[..., 0, :, :]
                                if ctx.qkv_format in ["bshd", "sbhd"]
                                else kv_[0]
                            ),
                            (
                                kv_[..., 1, :, :]
                                if ctx.qkv_format in ["bshd", "sbhd"]
                                else kv_[1]
                            ),
                            out_,
                            softmax_lse,
                            *fa_backward_args_thd,
                            causal=True,
                            **fa_backward_kwargs,
                        )
                elif i >= (cp_size - rank - 1):
                    if ctx.qkv_format == "bshd":
                        # [b, 2, sq//2, np, hn] -> [b, sq, np, hn]
                        q_, out_, dout_ = [
                            x.view(x.shape[0], -1, *x.shape[-2:])
                            for x in [q, out, dout]
                        ]
                        if ctx.enable_mla:
                            # [b, 2, sk//2, np, hn] -> [b, sk, np, hn]
                            k_part = k_part[:, 0]
                            v_part = v_part[:, 0]
                        else:
                            # [b, 2, sk//2, 2, np, hn] -> [b, sk//2, 2, np, hn]
                            kv_ = kv[:, 0]
                    elif ctx.qkv_format == "sbhd":
                        # [2, sq//2, b, np, hn] -> [sq, b, np, hn]
                        q_, out_, dout_ = [
                            x.view(-1, *x.shape[-3:]) for x in [q, out, dout]
                        ]
                        if ctx.enable_mla:
                            # [2, sk//2, b, np, hn] -> [sk, b, np, hn]
                            k_part = k_part[0]
                            v_part = v_part[0]
                        else:
                            # [2, sk//2, b, 2, np, hn] -> [sk//2, b, 2, np, hn]
                            kv_ = kv[0]
                    elif ctx.qkv_format == "thd":
                        q_, out_, dout_ = q, out, dout
                        if ctx.enable_mla:
                            # [t, np, hn] -> [t/2, np, hn]
                            k_part = tex.thd_read_half_tensor(
                                k_part, cu_seqlens_kv_padded, 0
                            )
                            v_part = tex.thd_read_half_tensor(
                                v_part, cu_seqlens_kv_padded, 0
                            )
                        else:
                            # [2, t, np, hn] -> [2, t/2, np, hn]
                            kv_ = tex.thd_read_half_tensor(kv, cu_seqlens_kv_padded, 0)
                    if ctx.use_fused_attention:
                        if ctx.enable_mla:
                            k_part = k_part.contiguous()
                            v_part = v_part.contiguous()
                        else:
                            kv_ = kv_.contiguous()
                        if ctx.fp8:
                            aux_ctx_tensors = [
                                softmax_lse,
                                softmax_lse,
                                rng_states[cp_size - i - 1],
                            ]
                        else:
                            aux_ctx_tensors = [softmax_lse, rng_states[cp_size - i - 1]]
                        if attn_dbias is not None:
                            aux_ctx_tensors += [attn_biases[cp_size - i - 1]]
                        q_part = q_
                        if not ctx.enable_mla:
                            k_part = (
                                kv_[..., 0, :, :]
                                if ctx.qkv_format in ["bshd", "sbhd"]
                                else kv_[0]
                            )
                            v_part = (
                                kv_[..., 1, :, :]
                                if ctx.qkv_format in ["bshd", "sbhd"]
                                else kv_[1]
                            )
                        out_part = out_
                        dout_part = dout_

                        if ctx.fp8:
                            q_part = ctx.QKV_quantizer.create_tensor_from_data(
                                q_part, fake_dtype=ctx.qkv_dtype, internal=True
                            )
                            k_part = ctx.QKV_quantizer.create_tensor_from_data(
                                k_part, fake_dtype=ctx.qkv_dtype, internal=True
                            )
                            v_part = ctx.QKV_quantizer.create_tensor_from_data(
                                v_part, fake_dtype=ctx.qkv_dtype, internal=True
                            )
                            out_part = ctx.O_quantizer.create_tensor_from_data(
                                out_part, fake_dtype=ctx.qkv_dtype, internal=True
                            )
                            dout_part = ctx.dO_quantizer.create_tensor_from_data(
                                dout_part, fake_dtype=dout_dtype, internal=True
                            )
                            fp8_meta_kwargs["dp_quantizer"] = dP_quantizer_per_step[i]
                            fp8_meta_kwargs["dqkv_quantizer"] = (
                                dQKV_CP_quantizer_per_step[i]
                            )
                        dq_, dk_, dv_, dbias_ = fused_attn_bwd(
                            ctx.max_seqlen_q,
                            ctx.max_seqlen_kv // 2,
                            cu_seqlens_q_per_step[cp_size - i - 1],
                            cu_seqlens_kv_per_step[cp_size - i - 1],
                            q_part,
                            k_part,
                            v_part,
                            out_part,
                            dout_part,
                            dout_dtype,
                            fused_attn_dqkv_dtype,
                            aux_ctx_tensors,
                            fused_attn_backend,
                            cu_seqlens_q_padded=cu_seqlens_q_padded,
                            cu_seqlens_kv_padded=(
                                None
                                if cu_seqlens_kv_padded is None
                                else cu_seqlens_kv_padded // 2
                            ),
                            attn_scale=ctx.softmax_scale,
                            dropout=ctx.dropout_p,
                            qkv_layout=qkv_layout,
                            attn_mask_type="padding" if padding else "no_mask",
                            attn_bias_type=ctx.attn_bias_type,
                            deterministic=ctx.deterministic,
                            **fp8_meta_kwargs,
                        )
                        if ctx.fp8:
                            dq_ = dq_._data
                            dk_ = dk_._data
                            dv_ = dv_._data
                    else:
                        dq_ = torch.empty_like(q_)
                        dkv_ = torch.empty_like(kv_)
                        fa_backward_args_thd = get_fa_args(
                            False,
                            ctx.use_flash_attn_3,
                            ctx.qkv_format,
                            cu_seqlens_q=cu_seqlens_q_per_step[cp_size - i - 1],
                            cu_seqlens_kv=cu_seqlens_kv_per_step[cp_size - i - 1],
                            max_seqlen_q=ctx.max_seqlen_q,
                            max_seqlen_kv=ctx.max_seqlen_kv // 2,
                            dq=dq_,
                            dk=(
                                dkv_[..., 0, :, :]
                                if ctx.qkv_format in ["bshd", "sbhd"]
                                else dkv_[0]
                            ),
                            dv=(
                                dkv_[..., 1, :, :]
                                if ctx.qkv_format in ["bshd", "sbhd"]
                                else dkv_[1]
                            ),
                        )
                        if ctx.use_flash_attn_3 or (
                            fa_utils.v2_3_plus and not fa_utils.v2_7_0_plus
                        ):
                            fa_backward_kwargs["window_size"] = (-1, -1)
                        elif fa_utils.v2_7_0_plus:
                            fa_backward_kwargs["window_size_left"] = -1
                            fa_backward_kwargs["window_size_right"] = -1
                        if not ctx.use_flash_attn_3:
                            fa_backward_kwargs["rng_state"] = rng_states[
                                cp_size - i - 1
                            ]
                        # Need to add MLA support once Flash Attention supports MLA
                        flash_attn_bwd(
                            dout_,
                            q_,
                            (
                                kv_[..., 0, :, :]
                                if ctx.qkv_format in ["bshd", "sbhd"]
                                else kv_[0]
                            ),
                            (
                                kv_[..., 1, :, :]
                                if ctx.qkv_format in ["bshd", "sbhd"]
                                else kv_[1]
                            ),
                            out_,
                            softmax_lse,
                            *fa_backward_args_thd,
                            causal=False,
                            **fa_backward_kwargs,
                        )
                else:
                    if ctx.qkv_format == "bshd":
                        # [b, 2, sq//2, np, hn] -> [b, sq//2, np, hn]
                        q_, out_, dout_ = q[:, 1], out[:, 1], dout[:, 1]
                        if ctx.enable_mla:
                            # [b, 2, sk//2, np, hn] -> [b, sk, np, hn]
                            k_part = k_part.view(
                                k_part.shape[0], -1, *k_part.shape[-2:]
                            )
                            v_part = v_part.view(
                                v_part.shape[0], -1, *v_part.shape[-2:]
                            )
                        else:
                            # [b, 2, sk//2, 2, np, hn] -> [b, sk, 2, np, hn]
                            kv_ = kv.view(kv.shape[0], -1, *kv.shape[-3:])
                    elif ctx.qkv_format == "sbhd":
                        # [2, sq//2, b, np, hn] -> [sq//2, b, np, hn]
                        q_, out_, dout_ = q[1], out[1], dout[1]
                        if ctx.enable_mla:
                            # [2, sk//2, b, np, hn] -> [sk, b, np, hn]
                            k_part = k_part.view(-1, *k_part.shape[-3:])
                            v_part = v_part.view(-1, *v_part.shape[-3:])
                        else:
                            # [2, sk//2, b, 2, np, hn] -> [sk, b, 2, np, hn]
                            kv_ = kv.view(-1, *kv.shape[-4:])
                    elif ctx.qkv_format == "thd":
                        # [t, np, hn] -> [t/2, np, hn]
                        q_, out_, dout_ = [
                            tex.thd_read_half_tensor(x, cu_seqlens_q_padded, 1)
                            for x in [q, out, dout]
                        ]
                        kv_ = kv
                    if ctx.use_fused_attention:
                        q_, out_, dout_ = [x.contiguous() for x in [q_, out_, dout_]]
                        if ctx.fp8:
                            aux_ctx_tensors = [
                                softmax_lse_,
                                softmax_lse_,
                                rng_states[cp_size - i - 1],
                            ]
                        else:
                            aux_ctx_tensors = [
                                softmax_lse_,
                                rng_states[cp_size - i - 1],
                            ]
                        if attn_dbias is not None:
                            aux_ctx_tensors += [attn_biases[cp_size - i - 1]]

                        q_part = q_
                        if not ctx.enable_mla:
                            k_part = (
                                kv_[..., 0, :, :]
                                if ctx.qkv_format in ["bshd", "sbhd"]
                                else kv_[0]
                            )
                            v_part = (
                                kv_[..., 1, :, :]
                                if ctx.qkv_format in ["bshd", "sbhd"]
                                else kv_[1]
                            )
                        out_part = out_
                        dout_part = dout_

                        if ctx.fp8:
                            q_part = ctx.QKV_quantizer.create_tensor_from_data(
                                q_part, fake_dtype=ctx.qkv_dtype, internal=True
                            )
                            k_part = ctx.QKV_quantizer.create_tensor_from_data(
                                k_part, fake_dtype=ctx.qkv_dtype, internal=True
                            )
                            v_part = ctx.QKV_quantizer.create_tensor_from_data(
                                v_part, fake_dtype=ctx.qkv_dtype, internal=True
                            )
                            out_part = ctx.O_quantizer.create_tensor_from_data(
                                out_part, fake_dtype=ctx.qkv_dtype, internal=True
                            )
                            dout_part = ctx.dO_quantizer.create_tensor_from_data(
                                dout_part, fake_dtype=dout_dtype, internal=True
                            )
                            fp8_meta_kwargs["dp_quantizer"] = dP_quantizer_per_step[i]
                            fp8_meta_kwargs["dqkv_quantizer"] = (
                                dQKV_CP_quantizer_per_step[i]
                            )
                        dq_, dk_, dv_, dbias_ = fused_attn_bwd(
                            ctx.max_seqlen_q // 2,
                            ctx.max_seqlen_kv,
                            cu_seqlens_q_per_step[cp_size - i - 1],
                            cu_seqlens_kv_per_step[cp_size - i - 1],
                            q_part,
                            k_part,
                            v_part,
                            out_part,
                            dout_part,
                            dout_dtype,
                            fused_attn_dqkv_dtype,
                            aux_ctx_tensors,
                            fused_attn_backend,
                            cu_seqlens_q_padded=(
                                None
                                if cu_seqlens_q_padded is None
                                else cu_seqlens_q_padded // 2
                            ),
                            cu_seqlens_kv_padded=cu_seqlens_kv_padded,
                            attn_scale=ctx.softmax_scale,
                            dropout=ctx.dropout_p,
                            qkv_layout=qkv_layout,
                            attn_mask_type="padding" if padding else "no_mask",
                            attn_bias_type=ctx.attn_bias_type,
                            deterministic=ctx.deterministic,
                            **fp8_meta_kwargs,
                        )
                        if ctx.fp8:
                            dq_ = dq_._data
                            dk_ = dk_._data
                            dv_ = dv_._data
                    else:
                        dq_ = torch.empty_like(q_)
                        dkv_ = torch.empty_like(kv_)
                        fa_backward_args_thd = get_fa_args(
                            False,
                            ctx.use_flash_attn_3,
                            ctx.qkv_format,
                            cu_seqlens_q=cu_seqlens_q_per_step[cp_size - i - 1],
                            cu_seqlens_kv=cu_seqlens_kv_per_step[cp_size - i - 1],
                            max_seqlen_q=ctx.max_seqlen_q // 2,
                            max_seqlen_kv=ctx.max_seqlen_kv,
                            dq=dq_,
                            dk=(
                                dkv_[..., 0, :, :]
                                if ctx.qkv_format in ["bshd", "sbhd"]
                                else dkv_[0]
                            ),
                            dv=(
                                dkv_[..., 1, :, :]
                                if ctx.qkv_format in ["bshd", "sbhd"]
                                else dkv_[1]
                            ),
                        )
                        if ctx.use_flash_attn_3 or (
                            fa_utils.v2_3_plus and not fa_utils.v2_7_0_plus
                        ):
                            fa_backward_kwargs["window_size"] = (-1, -1)
                        elif fa_utils.v2_7_0_plus:
                            fa_backward_kwargs["window_size_left"] = -1
                            fa_backward_kwargs["window_size_right"] = -1
                        if not ctx.use_flash_attn_3:
                            fa_backward_kwargs["rng_state"] = rng_states[
                                cp_size - i - 1
                            ]
                        # Need to add MLA support once Flash Attention supports MLA
                        flash_attn_bwd(
                            dout_,
                            q_,
                            (
                                kv_[..., 0, :, :]
                                if ctx.qkv_format in ["bshd", "sbhd"]
                                else kv_[0]
                            ),
                            (
                                kv_[..., 1, :, :]
                                if ctx.qkv_format in ["bshd", "sbhd"]
                                else kv_[1]
                            ),
                            out_,
                            softmax_lse_,
                            *fa_backward_args_thd,
                            causal=False,
                            **fa_backward_kwargs,
                        )
            else:
                if ctx.use_fused_attention:
                    if ctx.fp8:
                        aux_ctx_tensors = [
                            softmax_lse,
                            softmax_lse,
                            rng_states[cp_size - i - 1],
                        ]
                    else:
                        aux_ctx_tensors = [softmax_lse, rng_states[cp_size - i - 1]]
                    if attn_dbias is not None:
                        aux_ctx_tensors += [attn_biases[cp_size - i - 1]]
                    q_part = q
                    if not ctx.enable_mla:
                        k_part = (
                            kv[..., 0, :, :]
                            if ctx.qkv_format in ["bshd", "sbhd"]
                            else kv[0]
                        )
                        v_part = (
                            kv[..., 1, :, :]
                            if ctx.qkv_format in ["bshd", "sbhd"]
                            else kv[1]
                        )
                    out_part = out
                    dout_part = dout

                    if ctx.fp8:
                        q_part = ctx.QKV_quantizer.create_tensor_from_data(
                            q_part, fake_dtype=ctx.qkv_dtype, internal=True
                        )
                        k_part = ctx.QKV_quantizer.create_tensor_from_data(
                            k_part, fake_dtype=ctx.qkv_dtype, internal=True
                        )
                        v_part = ctx.QKV_quantizer.create_tensor_from_data(
                            v_part, fake_dtype=ctx.qkv_dtype, internal=True
                        )
                        out_part = ctx.O_quantizer.create_tensor_from_data(
                            out_part, fake_dtype=ctx.qkv_dtype, internal=True
                        )
                        dout_part = ctx.dO_quantizer.create_tensor_from_data(
                            dout_part, fake_dtype=dout_dtype, internal=True
                        )
                        fp8_meta_kwargs["dp_quantizer"] = dP_quantizer_per_step[i]
                        fp8_meta_kwargs["dqkv_quantizer"] = dQKV_CP_quantizer_per_step[
                            i
                        ]
                    dq_, dk_, dv_, dbias_ = fused_attn_bwd(
                        ctx.max_seqlen_q,
                        ctx.max_seqlen_kv,
                        cu_seqlens_q_per_step[cp_size - i - 1],
                        cu_seqlens_kv_per_step[cp_size - i - 1],
                        q_part,
                        k_part,
                        v_part,
                        out_part,
                        dout_part,
                        dout_dtype,
                        fused_attn_dqkv_dtype,
                        aux_ctx_tensors,
                        fused_attn_backend,
                        cu_seqlens_q_padded=cu_seqlens_q_padded,
                        cu_seqlens_kv_padded=cu_seqlens_kv_padded,
                        attn_scale=ctx.softmax_scale,
                        dropout=ctx.dropout_p,
                        qkv_layout=qkv_layout,
                        attn_mask_type=ctx.attn_mask_type,
                        attn_bias_type=ctx.attn_bias_type,
                        deterministic=ctx.deterministic,
                        **fp8_meta_kwargs,
                    )

                    if ctx.fp8:
                        dq_ = dq_._data
                        dk_ = dk_._data
                        dv_ = dv_._data

                else:
                    dq_ = torch.empty_like(q)
                    dkv_ = torch.empty_like(kv)
                    fa_backward_args_thd = get_fa_args(
                        False,
                        ctx.use_flash_attn_3,
                        ctx.qkv_format,
                        cu_seqlens_q=cu_seqlens_q_per_step[cp_size - i - 1],
                        cu_seqlens_kv=cu_seqlens_kv_per_step[cp_size - i - 1],
                        max_seqlen_q=ctx.max_seqlen_q,
                        max_seqlen_kv=ctx.max_seqlen_kv,
                        dq=dq_,
                        dk=(
                            dkv_[..., 0, :, :]
                            if ctx.qkv_format in ["bshd", "sbhd"]
                            else dkv_[0]
                        ),
                        dv=(
                            dkv_[..., 1, :, :]
                            if ctx.qkv_format in ["bshd", "sbhd"]
                            else dkv_[1]
                        ),
                    )
                    if ctx.use_flash_attn_3 or (
                        fa_utils.v2_3_plus and not fa_utils.v2_7_0_plus
                    ):
                        fa_backward_kwargs["window_size"] = (-1, -1)
                    elif fa_utils.v2_7_0_plus:
                        fa_backward_kwargs["window_size_left"] = -1
                        fa_backward_kwargs["window_size_right"] = -1
                    if not ctx.use_flash_attn_3:
                        fa_backward_kwargs["rng_state"] = rng_states[cp_size - i - 1]
                    # Need to add MLA support once Flash Attention supports MLA
                    flash_attn_bwd(
                        dout,
                        q,
                        (
                            kv[..., 0, :, :]
                            if ctx.qkv_format in ["bshd", "sbhd"]
                            else kv[0]
                        ),
                        (
                            kv[..., 1, :, :]
                            if ctx.qkv_format in ["bshd", "sbhd"]
                            else kv[1]
                        ),
                        out,
                        softmax_lse,
                        *fa_backward_args_thd,
                        causal=False,
                        **fa_backward_kwargs,
                    )

            if ctx.fp8:
                dq = dq_fp8[(rank + i + 1) % cp_size]
            if (
                causal
                and ctx.qkv_format in ["bshd", "sbhd"]
                and i >= (cp_size - rank - 1)
            ):
                # [b, sq, np, hn] -> [b, 2, sq//2, np, hn] or
                # [sq, b, np, hn] -> [2, sq//2, b, np, hn]
                dq_ = dq_.view(*dq.shape)

            if ctx.fp8:
                if i >= (cp_size - rank - 1) or not causal:
                    dq.copy_(dq_)
                else:
                    if ctx.qkv_format == "bshd":
                        dq[:, 0, ...].fill_(0)
                        dq[:, 1, ...].copy_(dq_)
                    elif ctx.qkv_format == "sbhd":
                        dq[0].fill_(0)
                        dq[1].copy_(dq_)
            elif causal:
                if i > (cp_size - rank - 1):
                    dq.add_(dq_)
                elif i == (cp_size - rank - 1):
                    if rank == (cp_size - 1):
                        dq.copy_(dq_)
                    else:
                        if ctx.qkv_format == "bshd":
                            dq[:, 0, ...].copy_(dq_[:, 0, ...])
                            dq[:, 1, ...].add_(dq_[:, 1, ...])
                        elif ctx.qkv_format == "sbhd":
                            dq[0].copy_(dq_[0])
                            dq[1].add_(dq_[1])
                        elif ctx.qkv_format == "thd":
                            tex.thd_grad_correction(
                                dq, dq_, cu_seqlens_q_padded, "copy", "add"
                            )
                elif i > 0:
                    if ctx.qkv_format == "bshd":
                        dq[:, 1, ...].add_(dq_)
                    elif ctx.qkv_format == "sbhd":
                        dq[1].add_(dq_)
                    elif ctx.qkv_format == "thd":
                        tex.thd_grad_correction(
                            dq, dq_, cu_seqlens_q_padded, "none", "add"
                        )
                else:
                    if ctx.qkv_format == "bshd":
                        dq[:, 1, ...].copy_(dq_)
                    elif ctx.qkv_format == "sbhd":
                        dq[1].copy_(dq_)
                    elif ctx.qkv_format == "thd":
                        tex.thd_grad_correction(
                            dq, dq_, cu_seqlens_q_padded, "none", "copy"
                        )
            else:
                if i == 0:
                    dq.copy_(dq_)
                else:
                    dq.add_(dq_)

            if attn_dbias is not None:
                idx = (rank + i + 1) % cp_size
                if i == (cp_size - 1) or not causal:
                    # [b, np, sq, sk//cp] -> [b, np, sq, 2, sk//(2*cp)]
                    dbias_ = dbias_.view(*dbias_.shape[:-1], 2, dbias_.shape[-1] // 2)
                    attn_dbias[..., idx, :].copy_(dbias_[..., 0, :])
                    attn_dbias[..., (2 * cp_size - idx - 1), :].copy_(dbias_[..., 1, :])
                elif i >= (cp_size - rank - 1):
                    # [b, np, sq, sk//(2*cp)]
                    attn_dbias[..., idx, :].copy_(dbias_)
                else:
                    # [b, np, sq//2, sk//cp] -> [b, np, sq//2, 2, sk//(2*cp)]
                    dbias_ = dbias_.view(*dbias_.shape[:-1], 2, dbias_.shape[-1] // 2)
                    attn_dbias_[..., 1, :, idx, :].copy_(dbias_[..., 0, :])
                    attn_dbias_[..., 1, :, (2 * cp_size - idx - 1), :].copy_(
                        dbias_[..., 1, :]
                    )

            # wait until dKV is received
            for req in send_recv_reqs:
                req.wait()

            if ctx.fp8:
                if i < cp_size - 1:
                    dkv = dkv_fp8_[(rank + i + 1) % cp_size]
                else:
                    dkv = dkv_fp8[(rank + i + 1) % cp_size]
            else:
                dkv = p2p_comm_buffers[(i + 1) % 2][1]
            if ctx.use_fused_attention:
                if ctx.enable_mla:
                    dkv_ = None
                elif ctx.qkv_format in ["bshd", "sbhd"]:
                    dkv_ = combine_tensors([dk_, dv_], -2)
                elif ctx.qkv_format == "thd":
                    dkv_ = torch.cat(
                        (dk_.unsqueeze(0), dv_.unsqueeze(0)), dim=0
                    )  # pylint: disable=used-before-assignment
            if not ctx.enable_mla and ctx.qkv_format in ["bshd", "sbhd"]:
                # [b, 2, sk//2, 2, np, hn] -> [2, b, 2, sk//2, np, hn] or
                # [2, sk//2, b, 2, np, hn] -> [2, 2, sk//2, b, np, hn]
                # dkv is a buffer, so we do not need to transpose it, but only need to reshape it.
                dkv = dkv.view(2, *dkv.shape[0:-3], *dkv.shape[-2:])
                dkv_ = dkv_.movedim(-3, 0)
                if causal and (i < (cp_size - rank - 1) or i == (cp_size - 1)):
                    # [2, b, sk, np, hn] -> [2, b, 2, sk//2, np, hn] or
                    # [2, sk, b, np, hn] -> [2, 2, sk//2, b, np, hn]
                    dkv_ = dkv_.view(*dkv.shape)

            if ctx.enable_mla:
                # [b, 2, sk//2, np, hn] or
                # [2, sk//2, b, np, hn]
                dk = dkv[: ctx.k_numel].view(*ctx.k_shape)
                dv = dkv[ctx.k_numel :].view(*ctx.v_shape)
                if causal and (i < (cp_size - rank - 1) or i == (cp_size - 1)):
                    dk_ = dk_.view(*ctx.k_shape)
                    dv_ = dv_.view(*ctx.v_shape)

                if ctx.fp8:
                    # enable_mla and fp8
                    if causal and i >= (cp_size - rank - 1) and i != (cp_size - 1):
                        if ctx.qkv_format == "bshd":
                            dk[:, 0, ...].copy_(dk_)
                            dk[:, 1, ...].fill_(0)
                            dv[:, 0, ...].copy_(dv_)
                            dv[:, 1, ...].fill_(0)
                        elif ctx.qkv_format == "sbhd":
                            dk[0].copy_(dk_)
                            dk[1].fill_(0)
                            dv[0].copy_(dv_)
                            dv[1].fill_(0)
                        else:
                            dk.copy_(dk_)
                            dv.copy_(dv_)
                elif causal:
                    # enable_mla and not fp8 and causal
                    if i == (cp_size - 1):
                        if rank == 0:
                            if ctx.qkv_format == "bshd":
                                dk[:, 0, ...].add_(dk_[:, 0, ...])
                                dk[:, 1, ...].copy_(dk_[:, 1, ...])
                                dv[:, 0, ...].add_(dv_[:, 0, ...])
                                dv[:, 1, ...].copy_(dv_[:, 1, ...])
                            elif ctx.qkv_format == "sbhd":
                                dk[0, ...].add_(dk_[0, ...])
                                dk[1, ...].copy_(dk_[1, ...])
                                dv[0, ...].add_(dv_[0, ...])
                                dv[1, ...].copy_(dv_[1, ...])
                            elif ctx.qkv_format == "thd":
                                tex.thd_grad_correction(
                                    dk, dk_, cu_seqlens_kv_padded, "add", "copy"
                                )
                                tex.thd_grad_correction(
                                    dv, dv_, cu_seqlens_kv_padded, "add", "copy"
                                )
                        else:
                            dk.add_(dk_)
                            dv.add_(dv_)
                    elif i >= (cp_size - rank - 1):
                        if i == 0 and rank == (cp_size - 1):
                            if ctx.qkv_format == "bshd":
                                dk[:, 0, ...].copy_(dk_)
                                dv[:, 0, ...].copy_(dv_)
                            elif ctx.qkv_format == "sbhd":
                                dk[0, ...].copy_(dk_)
                                dv[0, ...].copy_(dv_)
                            elif ctx.qkv_format == "thd":
                                tex.thd_grad_correction(
                                    dk, dk_, cu_seqlens_kv_padded, "copy", "none"
                                )
                                tex.thd_grad_correction(
                                    dv, dv_, cu_seqlens_kv_padded, "copy", "none"
                                )
                        else:
                            if ctx.qkv_format == "bshd":
                                dk[:, 0, ...].add_(dk_)
                                dv[:, 0, ...].add_(dv_)
                            elif ctx.qkv_format == "sbhd":
                                dk[0, ...].add_(dk_)
                                dv[0, ...].add_(dv_)
                            elif ctx.qkv_format == "thd":
                                tex.thd_grad_correction(
                                    dk, dk_, cu_seqlens_kv_padded, "add", "none"
                                )
                                tex.thd_grad_correction(
                                    dv, dv_, cu_seqlens_kv_padded, "add", "none"
                                )
                    elif i > 0:
                        dk.add_(dk_)
                        dv.add_(dv_)
                    else:  # i == 0
                        dk.copy_(dk_)
                        dv.copy_(dv_)
                else:
                    # enable_mla and not fp8 and not causal
                    if i == 0:
                        dk.copy_(dk_)
                        dv.copy_(dv_)
                    else:  # i > 0
                        dk.add_(dk_)
                        dv.add_(dv_)
            else:
                if ctx.fp8:
                    # fp8
                    if causal and i >= (cp_size - rank - 1) and i != (cp_size - 1):
                        if ctx.qkv_format == "bshd":
                            dkv[:, :, 0, ...].copy_(dkv_)
                            dkv[:, :, 1, ...].fill_(0)
                        elif ctx.qkv_format == "sbhd":
                            dkv[:, 0, ...].copy_(dkv_)
                            dkv[:, 1, ...].fill_(0)
                    else:
                        dkv.copy_(dkv_)
                elif causal:
                    # not fp8 and causal
                    if i == (cp_size - 1):
                        if rank == 0:
                            if ctx.qkv_format == "bshd":
                                dkv[:, :, 0, ...].add_(dkv_[:, :, 0, ...])
                                dkv[:, :, 1, ...].copy_(dkv_[:, :, 1, ...])
                            elif ctx.qkv_format == "sbhd":
                                dkv[:, 0, ...].add_(dkv_[:, 0, ...])
                                dkv[:, 1, ...].copy_(dkv_[:, 1, ...])
                            elif ctx.qkv_format == "thd":
                                tex.thd_grad_correction(
                                    dkv, dkv_, cu_seqlens_kv_padded, "add", "copy"
                                )
                        else:
                            dkv.add_(dkv_)
                    elif i >= (cp_size - rank - 1):
                        if i == 0 and rank == (cp_size - 1):
                            if ctx.qkv_format == "bshd":
                                dkv[:, :, 0, ...].copy_(dkv_)
                            elif ctx.qkv_format == "sbhd":
                                dkv[:, 0, ...].copy_(dkv_)
                            elif ctx.qkv_format == "thd":
                                tex.thd_grad_correction(
                                    dkv, dkv_, cu_seqlens_kv_padded, "copy", "none"
                                )
                        else:
                            if ctx.qkv_format == "bshd":
                                dkv[:, :, 0, ...].add_(dkv_)
                            elif ctx.qkv_format == "sbhd":
                                dkv[:, 0, ...].add_(dkv_)
                            elif ctx.qkv_format == "thd":
                                tex.thd_grad_correction(
                                    dkv, dkv_, cu_seqlens_kv_padded, "add", "none"
                                )
                    elif i > 0:
                        dkv.add_(dkv_)
                    else:  # i == 0
                        dkv.copy_(dkv_)
                else:
                    # not fp8 and not causal
                    if i == 0:
                        dkv.copy_(dkv_)
                    else:  # i > 0
                        dkv.add_(dkv_)

        if ctx.fp8 and ctx.use_fused_attention:
            amax_cp_bwd = amax_per_step.amax(dim=1)
            ctx.dP_quantizer.amax.copy_(amax_cp_bwd[0])
            ctx.dQKV_CP_quantizer.amax.copy_(amax_cp_bwd[1])
            dq = ctx.dQKV_CP_quantizer.create_tensor_from_data(
                dq_fp8, fake_dtype=torch.float32, internal=True
            )

            if ctx.enable_mla:
                # [cp, b, 2, sk//2, np, hn] or [cp, 2, sk//2, b, np, hn]
                dk_fp8 = dkv_fp8[:, : ctx.k_numel].view(cp_size, *ctx.k_shape)
                dv_fp8 = dkv_fp8[:, ctx.k_numel :].view(cp_size, *ctx.v_shape)
                dk = ctx.dQKV_CP_quantizer.create_tensor_from_data(
                    dk_fp8, fake_dtype=torch.float32, internal=True
                )
                dv = ctx.dQKV_CP_quantizer.create_tensor_from_data(
                    dv_fp8, fake_dtype=torch.float32, internal=True
                )
                dq, dk, dv = [x.dequantize(dtype=torch.float32) for x in [dq, dk, dv]]
                dq, dk, dv = [x.sum(dim=0).to(dout_dtype) for x in [dq, dk, dv]]
            else:
                if ctx.qkv_format in ["bshd", "sbhd"]:
                    # [cp, b, 2, sk//2, 2, np, hn] -> [cp, 2, b, 2, sk//2, np, hn] or
                    # [cp, 2, sk//2, b, 2, np, hn] -> [cp, 2, 2, sk//2, b, np, hn]
                    dkv_fp8 = dkv_fp8.view(
                        cp_size, 2, *dkv_fp8.shape[1:-3], *dkv_fp8.shape[-2:]
                    )
                dkv = ctx.dQKV_CP_quantizer.create_tensor_from_data(
                    dkv_fp8, fake_dtype=torch.float32, internal=True
                )
                dq, dkv = [x.dequantize(dtype=torch.float32) for x in [dq, dkv]]
                dq, dkv = [x.sum(dim=0).to(dout_dtype) for x in [dq, dkv]]

        if causal:
            if ctx.qkv_format == "bshd":
                # [b, 2, sq//2, np, hn] -> [b, sq, np, hn]
                dq = dq.view(dq.shape[0], -1, *dq.shape[-2:])
                if ctx.enable_mla:
                    # [b, 2, sk//2, np, hn] -> [b, sk, np, hn]
                    dk = dk.view(dk.shape[0], -1, *dk.shape[-2:])
                    dv = dv.view(dv.shape[0], -1, *dv.shape[-2:])
                else:
                    # [2, b, 2, sk//2, np, hn] -> [2, b, sk, np, hn]
                    dkv = dkv.view(*dkv.shape[0:2], -1, *dkv.shape[-2:])
            elif ctx.qkv_format == "sbhd":
                # [2, sq//2, b, np, hn] -> [sq, b, np, hn]
                dq = dq.view(-1, *dq.shape[-3:])
                if ctx.enable_mla:
                    # [2, sk//2, b, np, hn] -> [sk, b, np, hn]
                    dk = dk.view(-1, *dk.shape[-3:])
                    dv = dv.view(-1, *dv.shape[-3:])
                else:
                    # [2, 2, sk//2, b, np, hn] -> [2, sk, b, np, hn]
                    dkv = dkv.view(dkv.shape[0], -1, *dkv.shape[-3:])

        if ctx.qkv_format == "thd" and not ctx.use_fused_attention:
            dq[cu_seqlens_q_padded[-1] :].fill_(0)
            if ctx.enable_mla:
                dk[cu_seqlens_kv_padded[-1] :].fill_(0)
                dv[cu_seqlens_kv_padded[-1] :].fill_(0)
            else:
                dkv[:, cu_seqlens_kv_padded[-1] :].fill_(0)

        if ctx.fp8 and ctx.is_input_fp8:
            assert torch.uint8 not in [dq.dtype, dkv.dtype]
            if ctx.enable_mla:
                dq, dk, dv = [ctx.dQKV_quantizer(x)._data for x in [dq, dk, dv]]
            else:
                dq, dkv = [ctx.dQKV_quantizer(x)._data for x in [dq, dkv]]
        if not ctx.enable_mla:
            dk, dv = dkv[0], dkv[1]

        if cp_size_a2a > 1:
            chunk_ids_for_a2a = get_seq_chunk_ids_for_reordering_after_attn(
                cp_size_a2a, q.device
            )
            dq, dk, dv = flash_attn_a2a_communicate(
                [dq, dk, dv],
                chunk_ids_for_a2a,
                seq_dim,
                cp_size_a2a,
                ctx.cp_group_a2a,
                ctx.cp_stream,
                False,
            )
            if ctx.qkv_format == "bshd":
                dq, dk, dv = [
                    x.view(ctx.batch_size, -1, *x.shape[-2:]) for x in [dq, dk, dv]
                ]
            elif ctx.qkv_format == "sbhd":
                dq, dk, dv = [
                    x.view(-1, ctx.batch_size, *x.shape[-2:]) for x in [dq, dk, dv]
                ]

        if attn_dbias is not None:
            # [b, np, sq, 2*cp, sk//(2*cp)] -> [b, np, sq, sk]
            attn_dbias = attn_dbias.view(*attn_dbias.shape[:-2], -1)
        # converting torch.uint8 to float8tensor
        if ctx.fp8 and ctx.is_input_fp8:
            dq = ctx.dQKV_quantizer.create_tensor_from_data(dq, fake_dtype=dout_dtype)
            dk = ctx.dQKV_quantizer.create_tensor_from_data(dk, fake_dtype=dout_dtype)
            dv = ctx.dQKV_quantizer.create_tensor_from_data(dv, fake_dtype=dout_dtype)
        nvtx_range_pop("transformer_engine.AttnFuncWithCPAndKVP2P.backward")

        return (
            None,
            dq,
            dk,
            dv,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            attn_dbias,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


te_cp.AttnFuncWithCPAndKVP2P = ProfiledAttnFuncWithCPAndKVP2P


class TEDotProductAttentionForTest(CommonOpsForTest, TEDotProductAttention):
    def __init__(
        self,
        config: TransformerConfig,
        layer_number,  # layer_number: layer number of the current `DotProductAttention` when multiple such modules are concatenated, for instance in consecutive transformer blocks.
        attn_mask_type: AttnMaskType = AttnMaskType.causal,
        attention_type: str = "self",  # default self for above 95% llm's attn type is self
        pg_collection: Optional[ProcessGroupCollection] = None,
        hook_activation: bool = False,
    ):
        TEDotProductAttention.__init__(
            self,
            config,
            layer_number,
            attn_mask_type,
            attention_type,
            pg_collection=pg_collection,
        )
        CommonOpsForTest.__init__(
            self,
            hook_activation=hook_activation,
            module_name="TEDotProductAttention",
        )
        self.config = config

    @nvtx_decorator(message="TEDotProductAttention forward")
    def _forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor,
        attn_mask_type: AttnMaskType,
        attention_bias: Tensor = None,
        packed_seq_params: PackedSeqParams = None,
    ):
        """Forward."""
        packed_seq_kwargs = (
            {
                key: getattr(packed_seq_params, key)
                for key in self.kept_packed_seq_params
            }
            if packed_seq_params is not None
            else {}
        )
        qkv_format = packed_seq_kwargs.get("qkv_format", self.qkv_format)
        attention_bias_kwargs = {}
        if attention_bias is not None:
            assert is_te_min_version("1.2.0"), (
                f"Transformer-Engine v{get_te_version()} must be >= 1.2.0 to support"
                "`attention_bias`."
            )
            attention_bias_kwargs = dict(
                core_attention_bias_type="post_scale_bias",
                core_attention_bias=attention_bias,
            )

        # this is for inference, so commented
        # if attn_mask_type == AttnMaskType.no_mask and self.config.window_size is not None:
        #     if (qkv_format == "bshd" and query.size(1) == 1) or (
        #         qkv_format == "sbhd" and query.size(0) == 1
        #     ):
        #         #  need to change mask type for SWA inference decode stage.
        #         attn_mask_type = AttnMaskType.causal_bottom_right
        if self.te_forward_mask_type:
            if qkv_format == "thd" and is_te_min_version("1.7.0"):
                # thd format uses flash attention with cuDNN kernel which requires is_padding=True,
                # so the only acceptable mask types are `padding_causal` and `padding`. These do not
                # necessarily indicate there are padded tokens in the sequence.
                if attn_mask_type == AttnMaskType.causal:
                    attn_mask_type = AttnMaskType.padding_causal
                elif attn_mask_type == AttnMaskType.no_mask:
                    attn_mask_type = AttnMaskType.padding

            # 这里的调用方法是对的，下面也一样，就是调用TEDotProductionAttention的父类的forward方法
            core_attn_out = super(TEDotProductAttention, self).forward(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type.name,
                **attention_bias_kwargs,
                **packed_seq_kwargs,
            )
        else:
            core_attn_out = super(TEDotProductAttention, self).forward(
                query,
                key,
                value,
                attention_mask,
                **attention_bias_kwargs,
                **packed_seq_kwargs,
            )

        return core_attn_out

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor,
        packed_seq_params: PackedSeqParams = None,
        attn_mask_type: AttnMaskType = AttnMaskType.causal,  # 先设置默认值，未来可能修改
        attention_bias: Tensor = None,
    ) -> Tensor:
        self.activation_hook.clear()
        with torch.autograd.graph.saved_tensors_hooks(
            self.activation_hook.save_hook, self.activation_hook.load_hook
        ):
            ret = self._forward(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type,
                attention_bias,
                packed_seq_params,
            )
            return ret
