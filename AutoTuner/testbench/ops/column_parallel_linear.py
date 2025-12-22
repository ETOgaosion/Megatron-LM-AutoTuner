import logging
from functools import reduce
from operator import mul as multiply_op
from typing import Optional, Tuple, Union

import torch
import transformer_engine_torch as tex
from megatron.core.extensions.transformer_engine import TELayerNormColumnParallelLinear
from megatron.core.inference.contexts.base_context import BaseInferenceContext
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import (
    WrappedTensor,
    nvtx_decorator,
    nvtx_range_pop,
    nvtx_range_push,
)
from torch import Tensor
from transformer_engine.pytorch.constants import dist_group_type
from transformer_engine.pytorch.cpp_extensions.gemm import general_gemm
from transformer_engine.pytorch.cpu_offload import (
    is_cpu_offload_enabled,
    mark_activation_offload,
)
from transformer_engine.pytorch.distributed import (
    _fsdp_gather_tensors,
    _fsdp_scatter_tensors,
    allreduce,
    gather_along_first_dim,
    get_distributed_world_size,
    in_fp8_activation_recompute_phase,
    reduce_scatter_along_first_dim,
    symmetric_all_reduce,
)
from transformer_engine.pytorch.export import is_in_onnx_export_mode
from transformer_engine.pytorch.fp8 import FP8GlobalStateManager
from transformer_engine.pytorch.graph import is_graph_capturing
from transformer_engine.pytorch.jit import no_torch_dynamo
from transformer_engine.pytorch.module._common import (
    WeightGradStore,
    apply_normalization,
    noop_cat,
)
from transformer_engine.pytorch.module.base import (
    _2X_ACC_DGRAD,
    _2X_ACC_FPROP,
    _2X_ACC_WGRAD,
    TransformerEngineBaseModule,
    fill_userbuffers_buffer_for_all_gather,
    get_dummy_wgrad,
    get_ub,
    get_workspace,
)
from transformer_engine.pytorch.tensor._internal.float8_blockwise_tensor_base import (
    Float8BlockwiseQTensorBase,
)
from transformer_engine.pytorch.tensor._internal.mxfp8_tensor_base import (
    MXFP8TensorBase,
)
from transformer_engine.pytorch.tensor.float8_blockwise_tensor import (
    Float8BlockQuantizer,
)
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
from transformer_engine.pytorch.tensor.quantized_tensor import (
    QuantizedTensorBase,
    Quantizer,
    prepare_for_saving,
    restore_from_saved,
)
from transformer_engine.pytorch.utils import (
    assert_dim_for_fp8_exec,
    cast_if_needed,
    clear_tensor_data,
    needs_quantized_gemm,
    nvtx_range_pop,
    nvtx_range_push,
    requires_grad,
)

try:
    import transformer_engine as te

    HAVE_TE = True
except ImportError:
    from unittest.mock import MagicMock

    te = MagicMock()
    HAVE_TE = False

from .common import CommonOpsForTest


class _LayerNormLinearHack(torch.autograd.Function):
    """LayerNormLinear semi-top level module
    Calls custom cuda extensions.
    """

    @staticmethod
    def forward(
        ctx,
        inp: torch.Tensor,
        ln_weight: torch.Tensor,
        ln_bias: Union[torch.Tensor, None],
        weight: torch.Tensor,
        bias: torch.Tensor,
        eps: float,
        is_first_microbatch: Union[bool, None],
        fp8: bool,
        fp8_calibration: bool,
        wgrad_store: WeightGradStore,
        fuse_wgrad_accumulation: bool,
        input_quantizer: Optional[Quantizer],
        weight_quantizer: Optional[Quantizer],
        output_quantizer: Optional[Quantizer],
        grad_input_quantizer: Optional[Quantizer],
        grad_weight_quantizer: Optional[Quantizer],
        grad_output_quantizer: Optional[Quantizer],
        cpu_offloading: bool,
        tp_group: Union[dist_group_type, None],
        tp_size: int,
        sequence_parallel: bool,
        tensor_parallel: bool,
        activation_dtype: torch.dtype,
        parallel_mode: Union[str, None],
        return_layernorm_output: bool,
        return_layernorm_output_gathered: bool,
        is_grad_enabled: bool,
        fwd_ln_sm_margin: int,
        bwd_ln_sm_margin: int,
        zero_centered_gamma: bool,
        normalization: str,
        ub_overlap_ag_fprop: bool,
        ub_overlap_rs_fprop: bool,
        ub_overlap_ag_dgrad: bool,
        ub_overlap_rs_dgrad: bool,
        ub_bulk_wgrad: bool,
        ub_bulk_dgrad: bool,
        ub_name: str,
        fsdp_group: Union[dist_group_type, None],
        module: torch.nn.Module,
        skip_fp8_weight_update: bool,
        symmetric_ar_type: str,
        debug: Optional[bool] = False,
    ) -> Union[Tuple[torch.Tensor, ...], torch.Tensor]:
        # pylint: disable=missing-function-docstring

        # NVTX label for profiling
        nvtx_label = "transformer_engine._LayerNormLinear.forward"
        if ub_name is not None:
            nvtx_label = f"{nvtx_label}.{ub_name}"

        # Make sure input dimensions are compatible
        out_features, in_features = weight.shape
        inp_shape = inp.shape
        inp_requires_grad = inp.requires_grad
        assert inp_shape[-1] == in_features, "GEMM not possible"
        inp = inp.view((-1, in_features))
        inputmat = inp
        if fp8:
            assert_dim_for_fp8_exec(inputmat, weight)

        # Cast for native AMP
        nvtx_range_push(f"{nvtx_label}.norm_input_cast")
        inputmat = cast_if_needed(inputmat, activation_dtype)
        ln_weight = cast_if_needed(ln_weight, activation_dtype)
        if ln_bias is not None:
            ln_bias = cast_if_needed(ln_bias, activation_dtype)
        nvtx_range_pop(f"{nvtx_label}.norm_input_cast")

        tp_world_size = get_distributed_world_size(tp_group)

        weight_requires_grad = weight.requires_grad
        backward_needs_input = is_grad_enabled and weight_requires_grad
        with_input_all_gather = parallel_mode == "column" and sequence_parallel

        # Configure Userbuffers communication (comm+GEMM overlap)
        nvtx_range_push(f"{nvtx_label}.ub_config")
        if debug:  # turn off userbuffers in debug mode
            ub_overlap_ag_fprop = False
            ub_overlap_rs_fprop = False
            ub_overlap_ag_dgrad = False
            ub_overlap_rs_dgrad = False
            ub_bulk_wgrad = False
            ub_bulk_dgrad = False
        ub_obj = None
        ub_type = None
        ub_overlap_ag_fprop = (
            ub_overlap_ag_fprop and is_grad_enabled and not return_layernorm_output
        )
        if ub_overlap_rs_fprop:
            ub_obj = get_ub(ub_name + "_fprop", fp8)
            ub_type = tex.CommOverlapType.RS
        elif ub_overlap_ag_fprop:
            ub_obj = get_ub(ub_name + "_fprop", fp8)
            ub_type = tex.CommOverlapType.AG
        nvtx_range_pop(f"{nvtx_label}.ub_config")

        # Configure quantizer for norm output
        if fp8:
            if input_quantizer is None:
                raise ValueError("Missing quantizer for input tensor")
            input_quantizer.set_usage(rowwise=True, columnwise=backward_needs_input)
            if (
                with_input_all_gather
                and input_quantizer.supports_only_rowwise_all_gather()
            ):
                # All-gather is not supported with FP8 column-wise data
                input_quantizer.set_usage(columnwise=False)

        # Avoid quantized norm kernel if norm output will be returned
        # or if a gather of ln_out must be in high precision.
        with_quantized_norm = (
            fp8
            and not debug
            and not return_layernorm_output
            and not return_layernorm_output_gathered
        )

        # Apply normalization
        nvtx_range_push(f"{nvtx_label}.norm")
        ln_out, mu, rsigma = apply_normalization(
            inputmat,
            None,  # ln_out
            ln_weight,
            ln_bias,
            eps,
            input_quantizer if with_quantized_norm else None,
            inputmat.dtype,
            normalization,
            fwd_ln_sm_margin,
            zero_centered_gamma,
        )
        nvtx_range_pop(f"{nvtx_label}.norm")

        # Store unquantized layer norm output if we need to return it
        ln_out_return = None
        if return_layernorm_output or return_layernorm_output_gathered:
            ln_out_return = ln_out

        # ------------------------------------------------------
        # Prepare GEMM input tensor
        # Note: Cast to expected dtype and perform tensor-parallel communication
        # ------------------------------------------------------
        nvtx_range_push(f"{nvtx_label}.gemm_input_cast_comm")
        ln_out_total = None
        if with_input_all_gather:
            if return_layernorm_output_gathered:
                # Perform all-gather in high precision if gathered
                # norm output will be returned
                ln_out_total, _ = gather_along_first_dim(ln_out, tp_group)
                ln_out_return = ln_out_total
                if fp8 or debug:
                    ln_out = input_quantizer(ln_out)
                    input_quantizer.set_usage(rowwise=True, columnwise=False)
                    if isinstance(input_quantizer, Float8BlockQuantizer):
                        input_quantizer.all_gather_usage = False
                    ln_out_total = input_quantizer(ln_out_total)
            else:
                quantizer = None
                if fp8 or debug:
                    quantizer = input_quantizer
                    if not with_quantized_norm:
                        ln_out = quantizer(ln_out)
                    quantizer.set_usage(rowwise=True, columnwise=False)
                if ub_overlap_ag_fprop:  # Initialize Userbuffers all-gather
                    ln_out_total, _ = fill_userbuffers_buffer_for_all_gather(
                        ub_obj,
                        ln_out,
                        quantizer,
                        tp_group,
                    )
                else:  # Perform NCCL all-gather
                    ln_out_total, _ = gather_along_first_dim(
                        ln_out,
                        tp_group,
                        quantizer=quantizer,
                    )
        else:
            if (fp8 or debug) and not with_quantized_norm:
                ln_out = input_quantizer(ln_out)
            ln_out_total = ln_out
        nvtx_range_pop(f"{nvtx_label}.gemm_input_cast_comm")
        # ------------------------------------------------------
        # GEMM input tensor is ready...
        # ------------------------------------------------------

        # ------------------------------------------------------
        # Prepare weight tensor
        # ------------------------------------------------------
        weightmat = weight
        quantized_weight = False
        if fp8 or debug:
            quantized_weight = not isinstance(weight, QuantizedTensorBase)

            # Configure quantizer
            if weight_quantizer is not None:
                weight_quantizer.set_usage(rowwise=True, columnwise=is_grad_enabled)

            # Get quantized weight
            update_workspace = is_first_microbatch is None or is_first_microbatch
            weightmat = module.get_weight_workspace(
                tensor=weight,
                quantizer=weight_quantizer,
                cache_name=(None if is_first_microbatch is None else "weight"),
                update_workspace=update_workspace,
                skip_update_flag=skip_fp8_weight_update,
                fsdp_group=fsdp_group,
                workspace_dtype=activation_dtype,
            )
            weightmat.update_usage(rowwise_usage=True)

        else:
            weightmat = cast_if_needed(weightmat, activation_dtype)  # Cast for AMP
        # ------------------------------------------------------
        # Weight tensor is ready for GEMM...
        # ------------------------------------------------------

        # Cast bias to expected dtype
        bias_dtype = activation_dtype
        if needs_quantized_gemm(ln_out_total) and activation_dtype == torch.float32:
            # cuBLAS does not support FP8 GEMM with FP32 bias, so we cast to BF16
            bias_dtype = torch.bfloat16
        bias = cast_if_needed(bias, bias_dtype) if bias is not None else bias

        # Calibrate quantizers if needed
        if not fp8 and fp8_calibration:
            if input_quantizer is not None:
                input_quantizer.calibrate(ln_out_total)
            if weight_quantizer is not None:
                weight_quantizer.calibrate(weight)

        # Choose whether to use GEMM kernel with split accumulator
        use_split_accumulator = _2X_ACC_FPROP
        if fp8:
            recipe = FP8GlobalStateManager.get_fp8_recipe()
            if hasattr(recipe, "fp8_gemm_fprop"):
                use_split_accumulator = recipe.fp8_gemm_fprop.use_split_accumulator

        # Configure output quantizer
        if output_quantizer is not None:
            output_quantizer.set_usage(rowwise=True, columnwise=False)

        # Output buffer for Userbuffers reduce-scatter
        reduce_scatter_out = None
        if ub_overlap_rs_fprop:
            out_shape = list(inp_shape)
            out_shape[0] //= tp_world_size
            out_shape[-1] = out_features
            reduce_scatter_out = torch.empty(
                out_shape, dtype=activation_dtype, device=inp.device
            )

        # ------------------------------------------------------
        # Forward GEMM
        # Note: y = x * w^T
        # ------------------------------------------------------
        nvtx_range_push(f"{nvtx_label}.gemm")
        gemm_out, *_, reduce_scatter_out = general_gemm(
            weightmat,
            ln_out_total,
            get_workspace(),
            quantization_params=output_quantizer,
            out_dtype=activation_dtype,
            bias=bias,
            use_split_accumulator=use_split_accumulator,
            ub=ub_obj,
            ub_type=ub_type,
            extra_output=reduce_scatter_out,
        )
        nvtx_range_pop(f"{nvtx_label}.gemm")
        # ------------------------------------------------------
        # Finished forward GEMM...
        # ------------------------------------------------------

        # Deallocate GEMM input tensor if no longer needed
        if not weight.requires_grad and not return_layernorm_output:
            ln_out = ln_out_total = None
            clear_tensor_data(ln_out, ln_out_total)

        # ------------------------------------------------------
        # Prepare output tensor
        # Note: Perform tensor-parallel communication
        # ------------------------------------------------------
        out = None
        if ub_overlap_rs_fprop:
            out = reduce_scatter_out
        elif parallel_mode == "row" and tp_size > 1:
            nvtx_range_push(f"{nvtx_label}.row_parallel_comm")
            out = gemm_out
            if sequence_parallel:
                out, _ = reduce_scatter_along_first_dim(out, tp_group)
            elif tensor_parallel:
                if symmetric_ar_type is not None:
                    out, _ = symmetric_all_reduce(
                        out, tp_group, all_reduce_type=symmetric_ar_type
                    )
                else:
                    out, _ = allreduce(out, tp_group)
            nvtx_range_pop(f"{nvtx_label}.row_parallel_comm")
        else:
            out = gemm_out
        out = out.view(-1, *inp_shape[1:-1], out_features)
        # ------------------------------------------------------
        # Output tensor is ready to return...
        # ------------------------------------------------------

        # ------------------------------------------------------
        # Cache state for backward pass
        # ------------------------------------------------------

        if is_grad_enabled:
            ctx.weight_quantizer = weight_quantizer
            ctx.ln_out_needs_gather = (
                weight.requires_grad and parallel_mode == "column" and sequence_parallel
            )

            # Input with column-wise usage is needed for wgrad GEMM.
            if backward_needs_input:
                if isinstance(ln_out, QuantizedTensorBase):
                    # For sequence parallel in vanilla FP8, rowwise data is
                    # to gather the input. For MXFP8, columnwise only data
                    # can be allgathered.
                    if (
                        isinstance(
                            ln_out, (MXFP8TensorBase, Float8BlockwiseQTensorBase)
                        )
                        or not ctx.ln_out_needs_gather
                    ):
                        ln_out.update_usage(rowwise_usage=False)

            # Weight with column-wise usage is needed for dgrad GEMM.
            if isinstance(weightmat, QuantizedTensorBase):
                weightmat.update_usage(columnwise_usage=True)

            if cpu_offloading:
                mark_activation_offload(inputmat, mu, rsigma, ln_out)

            # Scatter intermediate/activation tensors saved for the backward pass
            # NOTE: weight_fp8 = weight when ctx.fp8 == False and torch.disttributed.FSDP already
            #       shards/unshards the base weights so we don't do it ourselves
            nvtx_range_push(f"{nvtx_label}.fsdp_scatter")
            ctx.fsdp_group = fsdp_group
            ctx.fsdp_shapes = _fsdp_scatter_tensors(
                fsdp_group,
                mu,
                rsigma,
                weightmat if quantized_weight else None,
                ln_out if weight.requires_grad else None,
            )
            nvtx_range_pop(f"{nvtx_label}.fsdp_scatter")

            if cpu_offloading:
                ctx.grad_added_to_main_grad = hasattr(weight, "grad_added_to_main_grad")

                if ctx.grad_added_to_main_grad:
                    # If you are passing torch.nn.Parameter through the Torch hooks, you will
                    # get back torch.Tensor. Torch rips off the Parameter wrapper.
                    # You need to preserve the weight object to have all the attributes user
                    # sets for the weights. Because of this, it is not recommended to offload
                    # weights if weights are externally touched outside this module
                    ctx.weight_object = weight

            tensors_to_save, tensor_objects = prepare_for_saving(
                inputmat,
                weightmat,
                weight,
                bias,
                ln_weight,
                ln_out,
                mu,
                rsigma,
            )
            ctx.save_for_backward(*tensors_to_save)
            ctx.tensor_objects = tensor_objects
            ctx.requires_dgrad = inp_requires_grad
            ctx.requires_wgrad = weight.requires_grad
            ctx.quantized_weight = quantized_weight
            if fuse_wgrad_accumulation and weight.requires_grad:
                # This check is needed to ensure that main_grad is not created
                # during the forward pass when using MCore FSDP as it creates
                # the main_grad buffer lazily before backprop
                if hasattr(weight, "__fsdp_param__"):
                    # MCore FSDP creates main_grad lazily before backward
                    ctx.main_grad_func = weight.get_main_grad
                else:
                    ctx.main_grad_func = lambda: weight.main_grad
            ctx.grad_input_quantizer = grad_input_quantizer
            ctx.grad_weight_quantizer = grad_weight_quantizer
            ctx.grad_output_quantizer = grad_output_quantizer
            ctx.input_quantizer = input_quantizer
            ctx.owns_input = inputmat is not inp
            ctx.weight = weight
            ctx.activation_dtype = activation_dtype
            ctx.fp8 = fp8
            ctx.fp8_recipe = FP8GlobalStateManager.get_fp8_recipe() if fp8 else None
            ctx.fuse_wgrad_accumulation = fuse_wgrad_accumulation
            ctx.cpu_offloading = cpu_offloading
            ctx.is_first_microbatch = is_first_microbatch
            ctx.use_bias = bias is not None
            ctx.sequence_parallel = sequence_parallel
            ctx.tensor_parallel = tensor_parallel
            ctx.inp_shape = inp_shape
            ctx.parallel_mode = parallel_mode
            ctx.tp_group = tp_group
            ctx.tp_size = tp_size
            ctx.return_layernorm_output = return_layernorm_output
            ctx.return_layernorm_output_gathered = return_layernorm_output_gathered
            ctx.bwd_ln_sm_margin = bwd_ln_sm_margin
            ctx.zero_centered_gamma = zero_centered_gamma
            ctx.ub_overlap_ag = ub_overlap_ag_dgrad
            ctx.ub_overlap_rs_dgrad = ub_overlap_rs_dgrad
            ctx.ub_bulk_wgrad = ub_bulk_wgrad
            ctx.ub_bulk_dgrad = ub_bulk_dgrad
            ctx.ub_name = ub_name
            ctx.requires_dgrad = inp_requires_grad
            ctx.normalization = normalization
            ctx.reduce_and_update_bwd_fp8_tensors = False
            if ctx.fp8 and requires_grad(inp, ln_weight, ln_bias, weight, bias):
                _first_fp8_module = FP8GlobalStateManager.IS_FIRST_FP8_MODULE
                ctx.reduce_and_update_bwd_fp8_tensors = (
                    FP8GlobalStateManager.is_first_fp8_module()
                )
                if in_fp8_activation_recompute_phase():
                    FP8GlobalStateManager.IS_FIRST_FP8_MODULE = _first_fp8_module
            ctx.wgrad_store = wgrad_store
            ctx.debug = debug

        # ------------------------------------------------------
        # Cached state for backward pass is ready...
        # ------------------------------------------------------

        if return_layernorm_output:
            if return_layernorm_output_gathered:
                shape = list(inp_shape)
                shape[0] *= tp_size if with_input_all_gather else 1
                return out, ln_out_return.view(shape)
            return out, ln_out_return.view(inp_shape)
        return out

    @staticmethod
    def backward(
        ctx, *grad_outputs: Tuple[torch.Tensor, ...]
    ) -> Tuple[Union[torch.Tensor, None], ...]:
        # pylint: disable=missing-function-docstring
        # NVTX label for profiling
        nvtx_label = "transformer_engine._LayerNormLinear.backward"
        if ctx.ub_name is not None:
            nvtx_label = f"{nvtx_label}.{ctx.ub_name}"

        with torch.cuda.nvtx.range("_LayerNormLinear_backward"):
            saved_tensors = ctx.saved_tensors
            (  # pylint: disable=unbalanced-tuple-unpacking
                inputmat,
                weight,
                origin_weight,
                bias,
                ln_weight,
                ln_out,
                mu,
                rsigma,
            ) = restore_from_saved(ctx.tensor_objects, saved_tensors)
            # Delete the references to tensor objects once they've been consumed
            # by the `restore_from_saved` method to construct back the actual tensors.
            ctx.tensor_objects = None

            # Since main_grad can be modified inplace, it should not be a part of saved_tensors
            main_grad = (
                ctx.main_grad_func()
                if weight is not None
                and ctx.fuse_wgrad_accumulation
                and ctx.requires_wgrad
                else None
            )

            # Gather intermediate/activation tensors if needed
            # NOTE: weight_fp8 = weight when ctx.fp8 == False and torch.disttributed.FSDP already
            #       shards/unshards the base weights so we don't do it ourselves
            nvtx_range_push(f"{nvtx_label}.fsdp_gather")
            _fsdp_gather_tensors(
                ctx.fsdp_group,
                ctx.fsdp_shapes,
                mu,
                rsigma,
                weight if ctx.fp8 and ctx.quantized_weight else None,
                ln_out,
            )
            nvtx_range_pop(f"{nvtx_label}.fsdp_gather")

            # For CPU offloading, we offloaded weight and weight.main_grad to different tensors,
            # we need to connect them into one.
            if ctx.cpu_offloading:
                if ctx.grad_added_to_main_grad:
                    origin_weight = ctx.weight_object
                if ctx.requires_wgrad and ctx.fuse_wgrad_accumulation:
                    origin_weight.main_grad = main_grad

            # Configure Userbuffers communication (comm+GEMM overlap)
            nvtx_range_push(f"{nvtx_label}.ub_config")
            ctx.ub_obj_gradout = None
            ub_obj_dgrad = None
            ub_obj_wgrad = None
            ub_type_dgrad = None
            ub_type_wgrad = None
            dgrad_shape = [reduce(multiply_op, ctx.inp_shape[:-1]), ctx.inp_shape[-1]]
            if ctx.ub_overlap_ag:
                # Overlap grad_output all-gather with dgrad compute
                ctx.ub_obj_gradout = get_ub(ctx.ub_name + "_dgrad", ctx.fp8)
                ub_obj_dgrad = ctx.ub_obj_gradout
                ub_type_dgrad = tex.CommOverlapType.AG
            elif ctx.ub_overlap_rs_dgrad:
                # Overlap dgrad reduce-scatter with dgrad compute
                ctx.ub_obj_gradout = get_ub(ctx.ub_name + "_dgrad", ctx.fp8)
                ub_obj_dgrad = ctx.ub_obj_gradout
                ub_type_dgrad = tex.CommOverlapType.RS
            else:
                if ctx.ub_bulk_dgrad:
                    # Overlap inputmat all-gather with dgrad compute
                    ctx.ub_obj_gradout = get_ub(ctx.ub_name + "_dgrad", ctx.fp8)
                    ub_obj_dgrad = ctx.ub_obj_gradout
                    ub_type_dgrad = tex.CommOverlapType.AG
                if ctx.ub_bulk_wgrad:
                    # Overlap dgrad reduce-scatter with wgrad compute
                    ub_obj_wgrad = get_ub(ctx.ub_name + "_wgrad", ctx.fp8)
                    ub_type_wgrad = tex.CommOverlapType.RS
            nvtx_range_pop(f"{nvtx_label}.ub_config")

            # --------------------------------------------------
            # Prepare grad output tensor
            # Note: Cast to expected dtype and perform tensor-parallel communication
            # --------------------------------------------------

            # Configure quantizer for grad output tensor
            # Note: dgrad GEMM requires row-wise usage, wgrad GEMM
            # requires column-wise usage
            if ctx.grad_output_quantizer is not None:
                quantizer = ctx.grad_output_quantizer
                quantizer.set_usage(rowwise=True, columnwise=True)
                if ctx.ub_overlap_ag:
                    # Userbuffers only supports communication for one
                    # tensor usage at a time. Configure quantizer with
                    # usage for only dgrad GEMM.
                    quantizer.set_usage(columnwise=False)

            # Prepare grad output tensor
            # Note: Cast to expected dtype and perform tensor-parallel communication
            nvtx_range_push(f"{nvtx_label}.grad_output_preprocess")
            (
                grad_output,
                grad_bias,
            ) = TransformerEngineBaseModule.grad_output_preprocess(
                ctx,
                grad_outputs[0],
                ctx.parallel_mode == "row",
                ctx.grad_output_quantizer,
            )
            nvtx_range_pop(f"{nvtx_label}.grad_output_preprocess")

            # --------------------------------------------------
            # Grad output tensor is ready for computing grad input...
            # --------------------------------------------------

            # --------------------------------------------------
            # Prepare GEMM input tensor
            # Note: Input tensor is needed for wgrad GEMM.
            # Tensor-parallel communication is overlapped with dgrad
            # GEMM.
            # --------------------------------------------------
            ln_out_total = None
            ln_out_total_work = None
            nvtx_range_push(f"{nvtx_label}.gemm_input_prep")
            if ctx.ln_out_needs_gather:
                quantizer = None
                if ctx.input_quantizer is not None:
                    quantizer = ctx.input_quantizer
                    if quantizer.supports_only_rowwise_all_gather():
                        # If data is in FP8, we compute FP8 transposes manually
                        quantizer.set_usage(rowwise=True, columnwise=False)
                    else:
                        # wgrad GEMM requires input with column-wise usage
                        quantizer.set_usage(rowwise=False, columnwise=True)
                if ctx.ub_bulk_dgrad:
                    ln_out_total, _ = fill_userbuffers_buffer_for_all_gather(
                        ub_obj_dgrad,
                        ln_out,
                        quantizer,
                        ctx.tp_group,
                    )
                else:
                    nvtx_range_push(f"{nvtx_label}.column_parallel_comm_input")
                    ln_out_total, ln_out_total_work = gather_along_first_dim(
                        ln_out,
                        ctx.tp_group,
                        async_op=True,
                        quantizer=quantizer,
                    )
                    nvtx_range_pop(f"{nvtx_label}.column_parallel_comm_input")
            else:
                ln_out_total = ln_out
            nvtx_range_pop(f"{nvtx_label}.gemm_input_prep")
            # --------------------------------------------------
            # Input tensor is ready for computing grad weight...
            # --------------------------------------------------

            # --------------------------------------------------
            # Compute grad input tensor
            # Note: Gradient w.r.t. GEMM input (i.e. norm output).
            # --------------------------------------------------

            # Make sure required data is available
            if isinstance(grad_output, QuantizedTensorBase):
                grad_output.update_usage(rowwise_usage=True)
            if ctx.weight_quantizer is not None and isinstance(
                weight, QuantizedTensorBase
            ):
                weight.update_usage(columnwise_usage=True)

            # Choose whether to use GEMM kernel with split accumulator
            use_split_accumulator = _2X_ACC_DGRAD
            if ctx.fp8:
                recipe = ctx.fp8_recipe
                if hasattr(recipe, "fp8_gemm_dgrad"):
                    use_split_accumulator = recipe.fp8_gemm_dgrad.use_split_accumulator

            # Update grad input quantizer
            if ctx.grad_input_quantizer is not None:
                ctx.grad_input_quantizer.set_usage(rowwise=True, columnwise=False)

            # Output buffers for Userbuffers reduce-scatter
            gemm_out = None
            reduce_scatter_out = None
            if ctx.ub_overlap_rs_dgrad:
                reduce_scatter_out = torch.empty(
                    dgrad_shape,
                    dtype=ctx.activation_dtype,
                    device=grad_outputs[0].device,
                )
            elif ctx.ub_bulk_wgrad:
                gemm_out = ub_obj_wgrad.get_buffer(local_chunk=False)

            # dgrad GEMM
            # Note: dx = dy * w
            nvtx_range_push(f"{nvtx_label}.dgrad_gemm")
            gemm_out, *_, reduce_scatter_out = general_gemm(
                weight,
                grad_output,
                get_workspace(),
                layout="NN",
                grad=True,
                quantization_params=ctx.grad_input_quantizer,
                out=gemm_out,
                out_dtype=ctx.activation_dtype,
                use_split_accumulator=use_split_accumulator,
                ub=ub_obj_dgrad,
                ub_type=ub_type_dgrad,
                extra_output=reduce_scatter_out,
                bulk_overlap=ctx.ub_bulk_dgrad,
            )
            nvtx_range_pop(f"{nvtx_label}.dgrad_gemm")

            # Prepare grad input tensor
            # Note: Perform tensor-parallel communication
            dgrad = None
            dgrad_work = None
            if ctx.ub_overlap_rs_dgrad:
                dgrad = reduce_scatter_out
            elif ctx.ub_bulk_wgrad:
                dgrad = ub_obj_wgrad.get_buffer(local_chunk=True)
            elif ctx.parallel_mode == "column" and ctx.tp_size > 1:
                nvtx_range_push(f"{nvtx_label}.column_parallel_comm_dgrad")
                dgrad = gemm_out
                if ctx.sequence_parallel:
                    dgrad, dgrad_work = reduce_scatter_along_first_dim(
                        dgrad,
                        ctx.tp_group,
                        async_op=True,
                    )
                else:
                    dgrad, dgrad_work = allreduce(dgrad, ctx.tp_group, async_op=True)
                nvtx_range_pop(f"{nvtx_label}.column_parallel_comm_dgrad")
            else:
                dgrad = gemm_out

            # --------------------------------------------------
            # Grad input tensor has been computed...
            # --------------------------------------------------

            # --------------------------------------------------
            # Compute grad weight
            # --------------------------------------------------

            wgrad = None
            if ctx.requires_wgrad:
                # Prepare grad output tensor
                # Note: Synchronize tensor-parallel communication and
                # make sure required data is available
                nvtx_range_push(f"{nvtx_label}.mxfp8_special_overlap")
                if ctx.ub_overlap_ag and isinstance(
                    ctx.grad_output_quantizer, MXFP8Quantizer
                ):
                    # UB does not support pipelined overlapping grad output
                    # all-gather with wgrad GEMM. Also, we can't
                    # convert row-scaled MXFP8 to column-scaled, so we
                    # can't reuse the grad output that was gathered
                    # for the dgrad GEMM. We work around by explicitly
                    # overlapping the AG operation with the dgrad GEMM.

                    # Get the communication stream from the dgrad GEMM to use for the AG
                    dgrad_send_stream, dgrad_recv_stream = (
                        ub_obj_dgrad.get_communication_stream()
                    )

                    # This object is separate from the ub_obj_wgrad object which is passed to the GEMM
                    ub_obj_overlap_wgrad = get_ub(ctx.ub_name + "_wgrad", ctx.fp8)

                    ctx.grad_output_quantizer.set_usage(rowwise=False, columnwise=True)

                    # We use the send stream to copy into the userbuffers.
                    # This is the same stream that we will use to access the data in the AG,
                    # so we dont need to add any syncs yet.
                    with torch.cuda.stream(dgrad_send_stream):
                        grad_output, _ = fill_userbuffers_buffer_for_all_gather(
                            ub_obj_overlap_wgrad,
                            grad_outputs[0],
                            ctx.grad_output_quantizer,
                            ctx.tp_group,
                        )

                    # Allgather grad_outputs[0] using the dgrad streams so we can overlap with the fc2_dgrad gemm
                    tex.bulk_overlap_ag_with_external_gemm(
                        ub_obj_overlap_wgrad, dgrad_send_stream, dgrad_recv_stream
                    )
                nvtx_range_pop(f"{nvtx_label}.mxfp8_special_overlap")

                # Prepare input tensor
                # Note: Synchronize tensor-parallel communication and
                # make sure required data is available
                nvtx_range_push(f"{nvtx_label}.wgrad_prep_and_gemm")
                if ln_out_total_work is not None:
                    ln_out_total_work.wait()
                    ln_out_total_work = None
                if ctx.fp8 or ctx.debug:
                    if isinstance(ln_out_total, QuantizedTensorBase):
                        ln_out_total.update_usage(columnwise_usage=True)
                    else:
                        ctx.input_quantizer.set_usage(rowwise=False, columnwise=True)
                        ln_out_total = ctx.input_quantizer(ln_out_total)

                if ctx.fp8 or ctx.debug:
                    if isinstance(grad_output, QuantizedTensorBase):
                        grad_output.update_usage(columnwise_usage=True)
                    else:
                        ctx.grad_output_quantizer.set_usage(
                            rowwise=False, columnwise=True
                        )
                        grad_output = ctx.grad_output_quantizer(grad_output)
                nvtx_range_pop(f"{nvtx_label}.wgrad_prep_and_gemm")

                # Figure out whether to use split accumulator
                use_split_accumulator = _2X_ACC_WGRAD
                if ctx.fp8:
                    recipe = ctx.fp8_recipe
                    if hasattr(recipe, "fp8_gemm_wgrad"):
                        use_split_accumulator = (
                            recipe.fp8_gemm_wgrad.use_split_accumulator
                        )

                # Figure out whether to output wgrad GEMM directly into main grad
                if ctx.is_first_microbatch is not None:
                    accumulate_wgrad_into_param_main_grad = (
                        ctx.fuse_wgrad_accumulation and not ctx.is_first_microbatch
                    )
                else:
                    accumulate_wgrad_into_param_main_grad = ctx.fuse_wgrad_accumulation

                # Output buffer for overlapping FP8 grad input
                # reduce-scatter with wgrad GEMM
                reduce_scatter_out = None
                if ctx.ub_bulk_wgrad and ub_obj_wgrad.is_fp8_ubuf():
                    reduce_scatter_out = torch.empty(
                        dgrad_shape,
                        dtype=ctx.activation_dtype,
                        device=grad_outputs[0].device,
                    )

                # Arguments to include in wgrad GEMM closure
                wgrad_gemm_kwargs = {
                    "workspace": get_workspace(),
                    "out_dtype": (
                        main_grad.dtype
                        if ctx.fuse_wgrad_accumulation
                        else ctx.activation_dtype
                    ),
                    "quantization_params": ctx.grad_weight_quantizer,
                    "accumulate": accumulate_wgrad_into_param_main_grad,
                    "layout": "NT",
                    "out": main_grad if ctx.fuse_wgrad_accumulation else None,
                    "bias": (bias if (grad_bias is None and not ctx.fp8) else None),
                    "use_split_accumulator": use_split_accumulator,
                    "grad": True,
                    "ub": ub_obj_wgrad,
                    "ub_type": ub_type_wgrad,
                    "extra_output": reduce_scatter_out,
                    "bulk_overlap": ctx.ub_bulk_wgrad,
                }

                def wgrad_gemm(
                    x: torch.Tensor,
                    dy: torch.Tensor,
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                    """Perform wgrad GEMM: dw = dy^T * x

                    May be fused with bgrad computation.

                    May be called outside of this function to enable
                    some advanced communication/compute overlapping.

                    """
                    nvtx_range_push(f"{nvtx_label}.wgrad_gemm")
                    dw, db, *_ = general_gemm(x, dy, **wgrad_gemm_kwargs)
                    nvtx_range_pop(f"{nvtx_label}.wgrad_gemm")
                    return dw, db

                # Choose whether to call wgrad GEMM now or delay
                if (
                    ctx.wgrad_store is not None
                    and ctx.wgrad_store.delay_wgrad_compute()
                ):
                    if (
                        wgrad_gemm_kwargs["ub"] is not None
                        or wgrad_gemm_kwargs["ub_type"] is not None
                        or wgrad_gemm_kwargs["extra_output"] is not None
                        or wgrad_gemm_kwargs["bulk_overlap"]
                    ):
                        raise NotImplementedError(
                            "Delayed weight grad computation is not supported "
                            "with Userbuffers (tensor-parallel communication overlapping)"
                        )
                    ctx.wgrad_store.put([ln_out_total, grad_output], wgrad_gemm)
                else:

                    # Call wgrad GEMM now
                    wgrad, grad_bias_ = wgrad_gemm(ln_out_total, grad_output)

                    # Update grad bias if needed
                    if grad_bias is None:
                        grad_bias = grad_bias_
                    del grad_bias_

                    # Deallocate input tensor if permitted
                    if not ctx.return_layernorm_output:
                        clear_tensor_data(ln_out_total)

                # Update grad input if overlapping reduce-scatter with wgrad GEMM
                if ctx.ub_bulk_wgrad:
                    nvtx_range_push(f"{nvtx_label}.wgrad_postprocess")
                    if ub_obj_wgrad.is_fp8_ubuf():
                        dgrad = reduce_scatter_out
                    else:
                        dgrad = ub_obj_wgrad.get_buffer(local_chunk=True).clone()
                    nvtx_range_pop(f"{nvtx_label}.wgrad_postprocess")
            # --------------------------------------------------
            # Grad weight has been computed...
            # --------------------------------------------------

            # Don't return grad bias if not needed
            if not ctx.use_bias:
                grad_bias = None

            # Synchronize tensor parallel communication
            nvtx_range_push(f"{nvtx_label}.comm_sync")
            if ln_out_total_work is not None:
                ln_out_total_work.wait()
                ln_out_total_work = None
            if dgrad_work is not None:
                dgrad_work.wait()
                dgrad_work = None
            nvtx_range_pop(f"{nvtx_label}.comm_sync")

            # Residual gradient
            dgrad = dgrad.view(inputmat.shape)
            if ctx.return_layernorm_output and not ctx.return_layernorm_output_gathered:
                dgrad = dgrad + grad_outputs[1].view_as(dgrad)

            # Norm gradient
            dgamma = None
            dbeta = None
            nvtx_range_push(f"{nvtx_label}.norm")
            if ctx.normalization == "LayerNorm":
                dgrad, dgamma, dbeta = tex.layernorm_bwd(
                    dgrad,
                    inputmat,
                    mu,
                    rsigma,
                    ln_weight,
                    ctx.bwd_ln_sm_margin,
                    ctx.zero_centered_gamma,
                )
                dgrad = dgrad.reshape(inputmat.size())
            elif ctx.normalization == "RMSNorm":
                dgrad, dgamma = tex.rmsnorm_bwd(
                    dgrad,
                    inputmat,
                    rsigma,
                    ln_weight,
                    ctx.bwd_ln_sm_margin,
                    ctx.zero_centered_gamma,
                )
                dgrad = dgrad.reshape(inputmat.size())
                dbeta = None
            nvtx_range_pop(f"{nvtx_label}.norm")
            clear_tensor_data(mu)
            clear_tensor_data(rsigma)

        if ctx.requires_wgrad:
            # Handle custom DDP from mcore.
            if ctx.fuse_wgrad_accumulation and hasattr(
                origin_weight, "grad_added_to_main_grad"
            ):
                origin_weight.grad_added_to_main_grad = True
                if getattr(origin_weight, "zero_out_wgrad", False):
                    wgrad = get_dummy_wgrad(
                        list(origin_weight.main_grad.shape),
                        origin_weight.dtype,
                        zero=True,
                    )
                else:
                    wgrad = get_dummy_wgrad(
                        list(origin_weight.main_grad.shape),
                        origin_weight.dtype,
                    )
            elif ctx.fuse_wgrad_accumulation:
                wgrad = None
        else:
            wgrad = None

        if ctx.reduce_and_update_bwd_fp8_tensors and not is_graph_capturing():
            nvtx_range_push(f"{nvtx_label}.reduce_and_update_fp8_tensors")
            FP8GlobalStateManager.reduce_and_update_fp8_tensors(forward=False)
            nvtx_range_pop(f"{nvtx_label}.reduce_and_update_fp8_tensors")

        # Scatter fp8 weight buffers
        # if ctx.fp8 and not isinstance(weight, QuantizedTensorBase):
        #    _fsdp_scatter_tensors(ctx.fsdp_group, weight_fp8)

        return (
            dgrad.view(ctx.inp_shape) if ctx.requires_dgrad else None,
            dgamma,
            dbeta,
            wgrad,
            grad_bias,
            None,  # eps
            None,  # is_first_microbatch
            None,  # fp8
            None,  # fp8_calibration
            None,  # wgrad_store
            None,  # fuse_wgrad_accumulation
            None,  # input_quantizer
            None,  # weight_quantizer
            None,  # output_quantizer
            None,  # grad_input_quantizer
            None,  # grad_weight_quantizer
            None,  # grad_output_quantizer
            None,  # cpu_offloading
            None,  # tp_group
            None,  # tp_size
            None,  # sequence_parallel
            None,  # tensor_parallel
            None,  # activation_dtype
            None,  # parallel_mode
            None,  # return_layernorm_output
            None,  # return_layernorm_output_gathered
            None,  # is_grad_enabled
            None,  # fwd_ln_sm_margin
            None,  # bwd_ln_sm_margin
            None,  # zero_centered_gamma
            None,  # normalization
            None,  # ub_overlap_ag_fprop
            None,  # ub_overlap_rs_fprop
            None,  # ub_overlap_ag_dgrad
            None,  # ub_overlap_rs_dgrad
            None,  # ub_bulk_dgrad
            None,  # ub_bulk_wgrad
            None,  # ub_name
            None,  # fsdp_group
            None,  # debug
            None,  # module
            None,  # skip_fp8_weight_update
            None,  # symmetric_ar_type
        )


@no_torch_dynamo()
def hack_forward(
    self,
    inp: torch.Tensor,
    is_first_microbatch: Optional[bool] = None,
    fp8_output: Optional[bool] = False,
    fp8_grad: Optional[bool] = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
    if is_in_onnx_export_mode():
        return self.onnx_forward(inp, fp8_output)

    debug = self.is_debug_iter()

    if FP8GlobalStateManager.fp8_graph_capturing():
        skip_fp8_weight_update = (
            FP8GlobalStateManager.get_skip_fp8_weight_update_tensor()
        )
    else:
        skip_fp8_weight_update = None
    if skip_fp8_weight_update is not None:
        is_first_microbatch = False

    if self.ub_overlap_rs_fprop:
        if get_ub(
            self.ub_name + "_fprop", FP8GlobalStateManager.is_fp8_enabled()
        ).is_fp8_ubuf():
            fp8_output = True
    if self.ub_overlap_rs_dgrad:
        if get_ub(
            self.ub_name + "_dgrad", FP8GlobalStateManager.is_fp8_enabled()
        ).is_fp8_ubuf():
            fp8_grad = True

    with (
        torch.cuda.device(getattr(self, list(self.named_parameters())[0][0]).device),
        self.prepare_forward(
            inp, allow_non_contiguous=False  # removed .contiguous from inside the layer
        ) as inp,
    ):

        # Get concatenated weight and bias tensors
        weight_tensor, bias_tensor = self._get_weight_and_bias_tensors()

        quantizers = (
            self._get_quantizers(fp8_output, fp8_grad)
            if not debug
            else self._get_debug_quantizers(fp8_output, fp8_grad)
        )
        if debug:
            if self.no_debug_features_active(quantizers):
                debug = False
                quantizers = self._get_quantizers(fp8_output, fp8_grad)

        (
            input_quantizer,
            weight_quantizer,
            output_quantizer,
            grad_input_quantizer,
            grad_weight_quantizer,
            grad_output_quantizer,
        ) = quantizers

        if torch.is_grad_enabled():
            fwd_fn = _LayerNormLinearHack.apply
            args = []
        else:
            fwd_fn = _LayerNormLinearHack.forward
            args = [None]
        args += (
            inp,
            self.layer_norm_weight,
            self.layer_norm_bias,
            weight_tensor,
            bias_tensor if self.apply_bias and not self.gemm_bias_unfused_add else None,
            self.eps,
            is_first_microbatch,
            self.fp8,
            self.fp8_calibration,
            self.wgrad_store,
            self.fuse_wgrad_accumulation,
            input_quantizer,
            weight_quantizer,
            output_quantizer,
            grad_input_quantizer,
            grad_weight_quantizer,
            grad_output_quantizer,
            is_cpu_offload_enabled(),
            self.tp_group,
            self.tp_size,
            self.sequence_parallel,
            self.tp_size > 1,
            self.activation_dtype,
            self.parallel_mode,
            self.return_layernorm_output,
            self.return_layernorm_output_gathered,
            torch.is_grad_enabled(),
            self.fwd_ln_sm_margin if torch.is_grad_enabled() else self.inf_ln_sm_margin,
            self.bwd_ln_sm_margin,
            self.zero_centered_gamma,
            self.normalization,
            self.ub_overlap_ag_fprop,
            self.ub_overlap_rs_fprop,
            self.ub_overlap_ag_dgrad,
            self.ub_overlap_rs_dgrad,
            self.ub_bulk_wgrad,
            self.ub_bulk_dgrad,
            self.ub_name,
            self.fsdp_group,
            self,
            skip_fp8_weight_update,
            self.symmetric_ar_type,
            debug,
        )
        out = fwd_fn(*args)

    if self.return_layernorm_output:
        out, ln_out = out

    if self.gemm_bias_unfused_add:
        out = out + cast_if_needed(bias_tensor, self.activation_dtype)

    if self.return_bias:
        if self.return_layernorm_output:
            return out, cast_if_needed(bias_tensor, self.activation_dtype), ln_out
        return out, cast_if_needed(bias_tensor, self.activation_dtype)
    if self.return_layernorm_output:
        return out, ln_out
    return out


te.pytorch.LayerNormLinear.forward = hack_forward


class ColumnParallelLinearForTest(CommonOpsForTest, TELayerNormColumnParallelLinear):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        tf_config: TransformerConfig,
        init_method: Optional[callable] = None,
        gather_output: bool = True,
        bias: bool = True,
        skip_bias_add: bool = False,
        is_expert: bool = False,
        skip_weight_param_allocation: bool = False,
        tp_comm_buffer_name: Optional[str] = None,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
        hook_activation=False,
    ):
        TELayerNormColumnParallelLinear.__init__(
            self,
            input_size=input_size,
            output_size=output_size,
            config=tf_config,
            init_method=init_method,
            gather_output=gather_output,
            bias=bias,
            skip_bias_add=skip_bias_add,
            is_expert=is_expert,
            # skip_weight_param_allocation=skip_weight_param_allocation,
            tp_comm_buffer_name=tp_comm_buffer_name,
            tp_group=tp_group,
        )
        CommonOpsForTest.__init__(
            self,
            hook_activation=hook_activation,
            module_name="TELayerNormColumnParallelLinear",
            logging_level=logging.INFO,
        )

    @nvtx_decorator(message="TELayerNormColumnParallelLinear forward")
    def _forward(self, x: Tensor) -> Tensor:
        out = super().forward(x)
        return out

    def forward(
        self,
        hidden_states: Union[Tensor, WrappedTensor],
        attention_mask: Optional[Tensor],
        rotary_pos_emb: Optional[Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[Tensor] = None,
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        rotary_pos_cos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        inference_context: Optional[BaseInferenceContext] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
        dynamic_inference_decode_only: Optional[bool] = None,
    ) -> Tensor:
        self.activation_hook.clear()
        with torch.autograd.graph.saved_tensors_hooks(
            self.activation_hook.save_hook, self.activation_hook.load_hook
        ):
            ret = self._forward(hidden_states)
        return ret
