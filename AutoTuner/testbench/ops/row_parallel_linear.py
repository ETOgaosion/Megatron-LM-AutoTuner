from typing import Callable, Optional, Tuple, Union

import torch
from megatron.core.extensions.transformer_engine import TERowParallelLinear
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.utils import nvtx_decorator, nvtx_range_pop, nvtx_range_push

from .common import CommonOpsForTest

try:
    import transformer_engine as te

    HAVE_TE = True
except ImportError as im_err:
    raise ImportError(
        "Transformer Engine is required for this module, but it is not installed.\n"
        "Please install it following the document of this repository"
    ) from im_err

from functools import reduce
from operator import mul as multiply_op

import transformer_engine_torch as tex
from transformer_engine.pytorch.constants import dist_group_type
from transformer_engine.pytorch.cpp_extensions import (
    general_gemm,
)
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
    is_fp8_activation_recompute_enabled,
    reduce_scatter_along_first_dim,
    symmetric_all_reduce,
)
from transformer_engine.pytorch.export import is_in_onnx_export_mode
from transformer_engine.pytorch.fp8 import FP8GlobalStateManager
from transformer_engine.pytorch.graph import is_graph_capturing
from transformer_engine.pytorch.jit import no_torch_dynamo
from transformer_engine.pytorch.module._common import WeightGradStore
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
from transformer_engine.pytorch.tensor.float8_tensor import (
    Float8CurrentScalingQuantizer,
    Float8Quantizer,
)
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
from transformer_engine.pytorch.tensor.quantized_tensor import (
    QuantizedTensor,
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


class _Linear(torch.autograd.Function):
    """Linear semi-top level module
    Calls custom cuda extensions.
    """

    @staticmethod
    def forward(
        ctx,
        weight: torch.Tensor,
        inp: torch.Tensor,
        bias: Optional[torch.Tensor],
        is_first_microbatch: Union[bool, None],
        fp8: bool,
        fp8_calibration: bool,
        wgrad_store: WeightGradStore,
        input_quantizer: Optional[Quantizer],
        weight_quantizer: Optional[Quantizer],
        output_quantizer: Optional[Quantizer],
        grad_input_quantizer: Optional[Quantizer],
        grad_weight_quantizer: Optional[Quantizer],
        grad_output_quantizer: Optional[Quantizer],
        fuse_wgrad_accumulation: bool,
        cpu_offloading: bool,
        tp_group: Union[dist_group_type, None],
        tp_size: int,
        sequence_parallel: bool,
        tensor_parallel: bool,
        activation_dtype: torch.dtype,
        parallel_mode: Union[str, None],
        is_grad_enabled: bool,
        ub_overlap_rs_fprop: bool,
        ub_overlap_ag_dgrad: bool,
        ub_overlap_ag_fprop: bool,
        ub_overlap_rs_dgrad: bool,
        ub_bulk_dgrad: bool,
        ub_bulk_wgrad: bool,
        ub_name: str,
        fp8_output: bool,  # pylint: disable=unused-argument
        fsdp_group: Union[dist_group_type, None],
        module: torch.nn.Module,
        skip_fp8_weight_update: bool,
        symmetric_ar_type: str,
        save_original_input: bool = False,
        debug: Optional[bool] = False,
    ) -> torch.Tensor:
        # pylint: disable=missing-function-docstring
        # NVTX label for profiling
        nvtx_label = "transformer_engine._Linear.forward"
        if ub_name is not None:
            nvtx_label = f"{nvtx_label}.{ub_name}"

        # Make sure input dimensions are compatible
        out_features, in_features = weight.shape
        assert inp.shape[-1] == in_features, "GEMM not possible"

        # Configure tensor-parallel communication
        tp_world_size = get_distributed_world_size(tp_group)
        backward_needs_input = is_grad_enabled and weight.requires_grad
        with_input_all_gather_nccl = (
            parallel_mode == "column" and sequence_parallel and not ub_overlap_ag_fprop
        )

        # Configure Userbuffers communication (comm+GEMM overlap)
        if debug:  # turn off userbuffers in debug mode
            ub_overlap_rs_fprop = False
            ub_overlap_ag_fprop = False
            ub_overlap_rs_dgrad = False
            ub_bulk_wgrad = False
            ub_bulk_dgrad = False
        ub_obj = None
        ub_type = None

        nvtx_range_push(f"{nvtx_label}.configure_user_buffers")
        if ub_overlap_rs_fprop:
            ub_obj = get_ub(ub_name + "_fprop")
            ub_type = tex.CommOverlapType.RS
        elif ub_overlap_ag_fprop:
            ub_obj = get_ub(ub_name + "_fprop")
            ub_type = tex.CommOverlapType.AG
        nvtx_range_pop(f"{nvtx_label}.configure_user_buffers")
        # ------------------------------------------------------
        # Prepare input tensor
        # Note: Cast to expected dtype and perform tensor-parallel communication
        # ------------------------------------------------------
        nvtx_range_push(f"{nvtx_label}.input_cast_comm")
        inputmat = inp  # Input tensor to save for backward (maybe sharded)
        inputmat_total = None  # Input tensor to pass to GEMM (gathered)
        own_quantized_input = False
        if fp8:
            assert_dim_for_fp8_exec(inputmat, weight)
            if save_original_input:
                assert not isinstance(
                    input_quantizer, Float8Quantizer
                ), "DelayedScaling recipe is not supported with save_original_input"

        if with_input_all_gather_nccl or ub_overlap_ag_fprop:  # All-gather input tensor
            nvtx_range_push(f"{nvtx_label}.all_gather_inputs")
            # Cast local input tensor if needed
            if fp8 or debug:
                if input_quantizer is None:
                    raise ValueError("Missing quantizer for input tensor")
                if not isinstance(inputmat, QuantizedTensorBase):
                    own_quantized_input = True
                    input_quantizer.set_usage(
                        rowwise=True, columnwise=backward_needs_input
                    )
                    if isinstance(
                        input_quantizer,
                        (Float8Quantizer, Float8CurrentScalingQuantizer),
                    ):
                        # All-gather is not supported with FP8 column-wise data
                        input_quantizer.set_usage(columnwise=False)
                    if save_original_input:
                        # No need for column-wise data since this
                        # tensor will not be cached for backward pass
                        input_quantizer.set_usage(columnwise=False)
                        own_quantized_input = False
                    inputmat = input_quantizer(inputmat)
            else:
                inputmat = cast_if_needed(inp, activation_dtype)  # Cast for AMP

            # Initialize gathered input tensor
            quantizer = None
            if fp8 or debug:
                quantizer = input_quantizer
                quantizer.set_usage(rowwise=True, columnwise=False)
            if with_input_all_gather_nccl:  # Perform NCCL all-gather
                # this will never be used in 'row' parallel mode, so no instrumentation is added
                inputmat_total, _ = gather_along_first_dim(
                    inputmat,
                    tp_group,
                    quantizer=quantizer,
                )
            elif ub_overlap_ag_fprop:  # Initialize Userbuffers all-gather
                inputmat_total, _ = fill_userbuffers_buffer_for_all_gather(
                    ub_obj,
                    inputmat,
                    quantizer,
                    tp_group,
                )
            nvtx_range_pop(f"{nvtx_label}.all_gather_inputs")
        else:  # Do not all-gather input tensor
            if fp8 or debug:
                if isinstance(inputmat, QuantizedTensorBase):
                    inputmat.update_usage(rowwise_usage=True)
                else:
                    if input_quantizer is None:
                        raise ValueError("Missing quantizer for input tensor")
                    input_quantizer.set_usage(
                        rowwise=True,
                        columnwise=backward_needs_input and not save_original_input,
                    )
                    inputmat = input_quantizer(inputmat)
                    own_quantized_input = True
            else:
                inputmat = cast_if_needed(inp, activation_dtype)  # Cast for AMP
            inputmat_total = inputmat
        nvtx_range_pop(f"{nvtx_label}.input_cast_comm")
        # ------------------------------------------------------
        # Input tensor is ready for GEMM...
        # ------------------------------------------------------

        # ------------------------------------------------------
        # Prepare weight tensor
        # ------------------------------------------------------
        nvtx_range_push(f"{nvtx_label}.prepare_weight_tensors")
        weightmat = weight
        if fp8 or debug:
            # Configure quantizer
            if weight_quantizer is not None:
                columnwise_usage = is_grad_enabled and inp.requires_grad
                if not columnwise_usage:
                    columnwise_usage = (
                        is_fp8_activation_recompute_enabled()
                        and not in_fp8_activation_recompute_phase()
                    )
                weight_quantizer.set_usage(rowwise=True, columnwise=columnwise_usage)

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
        nvtx_range_pop(f"{nvtx_label}.prepare_weight_tensors")
        # ------------------------------------------------------
        # Weight tensor is ready for GEMM...
        # ------------------------------------------------------

        # Cast bias to expected dtype
        bias_dtype = activation_dtype
        if needs_quantized_gemm(inputmat_total) and activation_dtype == torch.float32:
            # cuBLAS does not support FP8 GEMM with FP32 bias, so we cast to BF16
            bias_dtype = torch.bfloat16
        bias = cast_if_needed(bias, bias_dtype) if bias is not None else bias

        # Calibrate quantizers if needed
        if not fp8 and fp8_calibration:
            if input_quantizer is not None:
                input_quantizer.calibrate(inputmat_total)
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
            out_shape = list(inp.shape)
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
            inputmat_total,
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
        # ------------------------------------------------------
        # Output tensor is ready to return...
        # ------------------------------------------------------

        # ------------------------------------------------------
        # Cache state for backward pass
        # ------------------------------------------------------

        if is_grad_enabled:
            nvtx_range_push(f"{nvtx_label}.cache_state_for_bwd")
            if save_original_input:
                inputmat = inp

            ctx.weight_quantizer = weight_quantizer

            ctx.backward_input_needs_gather = (
                weight.requires_grad and parallel_mode == "column" and sequence_parallel
            )

            # Discard unneeded data in input tensor
            if (
                backward_needs_input
                and own_quantized_input
                and isinstance(inputmat, QuantizedTensorBase)
            ):
                if (
                    ctx.backward_input_needs_gather
                    and weight_quantizer.supports_only_rowwise_all_gather()
                ):
                    # ctx's backward_input_needs_gather is false for parallel mode row, so this will not be used
                    # All-gather is not supported with FP8 column-wise data
                    inputmat.update_usage(rowwise_usage=True, columnwise_usage=False)
                else:
                    # Discard row-wise data since it is not needed in backward pass
                    inputmat.update_usage(rowwise_usage=False, columnwise_usage=True)

            # Cached input tensor
            saved_inputmat = None
            if backward_needs_input:
                saved_inputmat = inputmat

            # Weight with column-wise usage is needed for dgrad GEMM.
            if inp.requires_grad:
                if isinstance(weightmat, QuantizedTensorBase):
                    weightmat.update_usage(columnwise_usage=True)

            if cpu_offloading and saved_inputmat is not None:
                mark_activation_offload(saved_inputmat)

            # Scatter intermediate/activation tensors saved for the backward pass
            # NOTE: FSDP sharding is not valid for models initialized with primary Fp8 weights
            nvtx_range_push(f"{nvtx_label}.fsdp_scatter")
            ctx.fsdp_group = fsdp_group
            ctx.fsdp_shapes = _fsdp_scatter_tensors(
                fsdp_group,
                saved_inputmat,
                (
                    weightmat
                    if fp8 and not isinstance(weight, QuantizedTensorBase)
                    else None
                ),
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

            # TODO(ksivamani): Check memory usage
            tensors_to_save, tensor_objects = prepare_for_saving(
                saved_inputmat,
                weightmat,
                weight,
                bias,
            )
            ctx.save_for_backward(*tensors_to_save)
            ctx.tensor_objects = tensor_objects

            ctx.activation_dtype = activation_dtype
            ctx.fp8 = fp8
            ctx.fp8_recipe = FP8GlobalStateManager.get_fp8_recipe() if fp8 else None
            ctx.input_quantizer = input_quantizer
            ctx.grad_input_quantizer = grad_input_quantizer
            ctx.grad_weight_quantizer = grad_weight_quantizer
            ctx.grad_output_quantizer = grad_output_quantizer
            ctx.fuse_wgrad_accumulation = fuse_wgrad_accumulation
            if fuse_wgrad_accumulation and weight.requires_grad:
                # This check is needed to ensure that main_grad is not created
                # during the forward pass when using MCore FSDP as it creates
                # the main_grad buffer lazily before backprop
                if hasattr(weight, "__fsdp_param__"):
                    # MCore FSDP creates main_grad lazily before backward
                    ctx.main_grad_func = weight.get_main_grad
                else:
                    ctx.main_grad_func = lambda: weight.main_grad

            ctx.debug = debug
            ctx.cpu_offloading = cpu_offloading
            ctx.is_first_microbatch = is_first_microbatch
            ctx.use_bias = bias is not None
            ctx.sequence_parallel = sequence_parallel
            ctx.tensor_parallel = tensor_parallel
            ctx.inp_shape = inp.shape
            ctx.parallel_mode = parallel_mode
            ctx.tp_group = tp_group
            ctx.ub_overlap_ag = ub_overlap_ag_dgrad
            ctx.ub_overlap_rs_dgrad = ub_overlap_rs_dgrad
            ctx.ub_bulk_dgrad = ub_bulk_dgrad
            ctx.ub_bulk_wgrad = ub_bulk_wgrad
            ctx.ub_name = ub_name
            ctx.tp_size = tp_size
            ctx.requires_dgrad = inp.requires_grad
            ctx.requires_wgrad = weight.requires_grad
            ctx.reduce_and_update_bwd_fp8_tensors = False

            ctx.owns_input = saved_inputmat is not inp
            if ctx.fp8 and requires_grad(inp, weight, bias):
                _first_fp8_module = FP8GlobalStateManager.IS_FIRST_FP8_MODULE
                ctx.reduce_and_update_bwd_fp8_tensors = (
                    FP8GlobalStateManager.is_first_fp8_module()
                )
                if in_fp8_activation_recompute_phase():
                    FP8GlobalStateManager.IS_FIRST_FP8_MODULE = _first_fp8_module
            ctx.wgrad_store = wgrad_store
            nvtx_range_pop(f"{nvtx_label}.cache_state_for_bwd")
        # ------------------------------------------------------
        # Cached state for backward pass is ready...
        # ------------------------------------------------------

        return out

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> Tuple[Union[torch.Tensor, None], ...]:
        # pylint: disable=missing-function-docstring
        # NVTX label for profiling
        nvtx_label = "transformer_engine._Linear.backward"
        if ctx.ub_name is not None:
            nvtx_label = f"{nvtx_label}.{ctx.ub_name}"

        with torch.cuda.nvtx.range("_Linear_backward"):
            saved_tensors = ctx.saved_tensors
            (
                inputmat,
                weight_fp8,
                weight,
                bias,
            ) = restore_from_saved(  # pylint: disable=unbalanced-tuple-unpacking
                ctx.tensor_objects, saved_tensors
            )

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

            if ctx.cpu_offloading:
                if ctx.grad_added_to_main_grad:
                    weight = ctx.weight_object
                if ctx.requires_wgrad and ctx.fuse_wgrad_accumulation:
                    weight.main_grad = main_grad

            # Gather intermediate/activation tensors if needed
            # NOTE: weight_fp8 = weight when ctx.fp8 == False and torch.disttributed.FSDP already
            #       shards/unshards the base weights so we don't do it ourselves
            nvtx_range_push(f"{nvtx_label}.fsdp_gather")
            _fsdp_gather_tensors(
                ctx.fsdp_group,
                ctx.fsdp_shapes,
                inputmat,
                weight_fp8,
            )
            nvtx_range_pop(f"{nvtx_label}.fsdp_gather")

            # Configure Userbuffers communication (comm+GEMM overlap)
            nvtx_range_push(f"{nvtx_label}.configure_user_buffers")
            ctx.ub_obj_gradout = None
            ub_obj_dgrad = None
            ub_obj_wgrad = None
            ub_type_dgrad = None
            ub_type_wgrad = None
            dgrad_shape = [reduce(multiply_op, ctx.inp_shape[:-1]), ctx.inp_shape[-1]]
            if ctx.ub_overlap_ag:
                # Overlap grad_output all-gather with dgrad compute
                ctx.ub_obj_gradout = get_ub(ctx.ub_name + "_dgrad")
                ub_obj_dgrad = ctx.ub_obj_gradout
                ub_type_dgrad = tex.CommOverlapType.AG
            elif ctx.ub_overlap_rs_dgrad:
                # Overlap dgrad reduce-scatter with dgrad compute
                ctx.ub_obj_gradout = get_ub(ctx.ub_name + "_dgrad")
                ub_obj_dgrad = ctx.ub_obj_gradout
                ub_type_dgrad = tex.CommOverlapType.RS
            else:
                if ctx.ub_bulk_dgrad:
                    # Overlap inputmat all-gather with dgrad compute
                    ctx.ub_obj_gradout = get_ub(ctx.ub_name + "_dgrad")
                    ub_obj_dgrad = ctx.ub_obj_gradout
                    ub_type_dgrad = tex.CommOverlapType.AG
                if ctx.ub_bulk_wgrad:
                    # Overlap dgrad reduce-scatter with wgrad compute
                    ub_obj_wgrad = get_ub(ctx.ub_name + "_wgrad")
                    ub_type_wgrad = tex.CommOverlapType.RS
            nvtx_range_pop(f"{nvtx_label}.configure_user_buffers")

            # --------------------------------------------------
            # Prepare grad output tensor
            # Note: Cast to expected dtype and perform tensor-parallel communication
            # --------------------------------------------------

            # Unmodified grad output tensor
            grad_output_arg = grad_output

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

            # Adjust the quantization direction approach depending
            # on whether wgrad calculations will be performed.
            # NOTE: If requires_dgrad is False, disabling `rowwise` quantization and keeping `columnwise` quantization
            #       results in `Assertion failed: output_tensor->has_data(). Quantizing in only the columnwise direction not supported yet!`
            # NOTE: For `ctx.bias is True`, selected quantize kernel errors with
            #       `cast_kernels.cuh:1322 in function fp8_quantize_arch_l_100: Not implemented scaling mode or fusion: NVTE_DELAYED_TENSOR_SCALING or IS_DBIAS=true on GPU with compute capability < 10.0.`
            if (
                not ctx.use_bias
                and not ctx.requires_wgrad
                and ctx.grad_output_quantizer is not None
            ):
                ctx.grad_output_quantizer.set_usage(columnwise=False)

            # Prepare grad output tensor
            nvtx_range_push(f"{nvtx_label}.grad_output_preprocess")
            (
                grad_output,
                grad_bias,
            ) = TransformerEngineBaseModule.grad_output_preprocess(
                ctx,
                grad_output,
                ctx.parallel_mode == "row",
                ctx.grad_output_quantizer,
            )
            nvtx_range_pop(f"{nvtx_label}.grad_output_preprocess")

            # --------------------------------------------------
            # Grad output tensor is ready for computing grad input...
            # --------------------------------------------------

            # --------------------------------------------------
            # Prepare input tensor
            # Note: Input tensor is needed for wgrad GEMM.
            # Tensor-parallel communication is overlapped with dgrad
            # GEMM.
            # --------------------------------------------------
            inputmat_total = None
            inputmat_total_work = None
            if ctx.requires_wgrad:
                if ctx.fp8 or ctx.debug:
                    if isinstance(inputmat, QuantizedTensorBase):
                        # Input tensor is already quantized
                        pass
                    elif ctx.debug:
                        # Debug quantizer will be applied immediately before wgrad GEMM
                        pass
                    else:
                        # Quantize input tensor
                        quantizer = ctx.input_quantizer
                        if quantizer.supports_only_rowwise_all_gather():
                            # All-gather is not supported with FP8 column-wise data
                            quantizer.set_usage(
                                rowwise=True,
                                columnwise=not ctx.backward_input_needs_gather,
                            )
                        else:
                            quantizer.set_usage(rowwise=False, columnwise=True)
                        inputmat = quantizer(inputmat)
                else:
                    if isinstance(inputmat, QuantizedTensorBase):
                        inputmat = inputmat.dequantize(dtype=ctx.activation_dtype)
                    else:
                        inputmat = cast_if_needed(inputmat, ctx.activation_dtype)
            if ctx.backward_input_needs_gather:
                quantizer = None
                if ctx.fp8 or ctx.debug:
                    quantizer = ctx.input_quantizer
                    if quantizer.supports_only_rowwise_all_gather():
                        # If data is in FP8, we compute FP8 transposes manually
                        quantizer.set_usage(rowwise=True, columnwise=False)
                    else:
                        # wgrad GEMM requires input with column-wise usage
                        quantizer.set_usage(rowwise=False, columnwise=True)
                if ctx.ub_bulk_dgrad:
                    inputmat_total, _ = fill_userbuffers_buffer_for_all_gather(
                        ub_obj_dgrad,
                        inputmat,
                        quantizer,
                        ctx.tp_group,
                    )
                else:
                    nvtx_range_push(f"{nvtx_label}.column_parallel_comm_input")
                    inputmat_total, inputmat_total_work = gather_along_first_dim(
                        inputmat,
                        ctx.tp_group,
                        async_op=True,
                        quantizer=quantizer,
                    )
                    nvtx_range_pop(f"{nvtx_label}.column_parallel_comm_input")
            else:
                inputmat_total = inputmat
            # --------------------------------------------------
            # Input tensor is ready for computing grad weight...
            # --------------------------------------------------

            # --------------------------------------------------
            # Compute grad input tensor
            # --------------------------------------------------

            dgrad = None
            dgrad_work = None
            if ctx.requires_dgrad:

                # Make sure required data is available
                if isinstance(grad_output, QuantizedTensorBase):
                    grad_output.update_usage(rowwise_usage=True)
                if ctx.weight_quantizer is not None and isinstance(
                    weight_fp8, QuantizedTensorBase
                ):
                    weight_fp8.update_usage(columnwise_usage=True)

                # Choose whether to use GEMM kernel with split accumulator
                use_split_accumulator = _2X_ACC_DGRAD
                if ctx.fp8:
                    recipe = ctx.fp8_recipe
                    if hasattr(recipe, "fp8_gemm_dgrad"):
                        use_split_accumulator = (
                            recipe.fp8_gemm_dgrad.use_split_accumulator
                        )

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
                        device=grad_output_arg.device,
                    )
                elif ctx.ub_bulk_wgrad:
                    gemm_out = ub_obj_wgrad.get_buffer(local_chunk=False)

                # dgrad GEMM
                # Note: dx = dy * w
                nvtx_range_push(f"{nvtx_label}.dgrad_gemm")
                gemm_out, *_, reduce_scatter_out = general_gemm(
                    weight_fp8,
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
                        dgrad, dgrad_work = allreduce(
                            dgrad, ctx.tp_group, async_op=True
                        )
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

                # Prepare input tensor
                # Note: Synchronize tensor-parallel communication and
                # make sure required data is available
                if inputmat_total_work is not None:
                    inputmat_total_work.wait()
                    inputmat_total_work = None
                if ctx.fp8 or ctx.debug:
                    if isinstance(inputmat_total, QuantizedTensorBase):
                        inputmat_total.update_usage(columnwise_usage=True)
                    else:
                        ctx.input_quantizer.set_usage(rowwise=False, columnwise=True)
                        inputmat_total = ctx.input_quantizer(inputmat_total)

                # Prepare grad output tensor
                # Note: Synchronize tensor-parallel communication and
                # make sure required data is available
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
                    ub_obj_overlap_wgrad = get_ub(ctx.ub_name + "_wgrad")

                    ctx.grad_output_quantizer.set_usage(rowwise=False, columnwise=True)

                    # We use the send stream to copy into the userbuffers.
                    # This is the same stream that we will use to access the data in the AG,
                    # so we dont need to add any syncs yet.
                    with torch.cuda.stream(dgrad_send_stream):
                        grad_output, _ = fill_userbuffers_buffer_for_all_gather(
                            ub_obj_overlap_wgrad,
                            grad_output_arg,
                            ctx.grad_output_quantizer,
                            ctx.tp_group,
                        )

                    # Allgather grad_outputs[0] using the dgrad streams so we can overlap with the fc2_dgrad gemm
                    tex.bulk_overlap_ag_with_external_gemm(
                        ub_obj_overlap_wgrad, dgrad_send_stream, dgrad_recv_stream
                    )

                if ctx.fp8 or ctx.debug:
                    if isinstance(grad_output, QuantizedTensorBase):
                        grad_output.update_usage(columnwise_usage=True)
                    else:
                        ctx.grad_output_quantizer.set_usage(
                            rowwise=False, columnwise=True
                        )
                        grad_output = ctx.grad_output_quantizer(grad_output)

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
                        device=grad_output_arg.device,
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
                    ctx.wgrad_store.put([inputmat_total, grad_output], wgrad_gemm)
                else:

                    # Call wgrad GEMM now
                    wgrad, grad_bias_ = wgrad_gemm(inputmat_total, grad_output)

                    # Update grad bias if needed
                    if grad_bias is None:
                        grad_bias = grad_bias_
                    del grad_bias_

                    # Deallocate input tensor if permitted
                    if ctx.owns_input:
                        clear_tensor_data(inputmat_total)

                # Update grad input if overlapping reduce-scatter with wgrad GEMM
                if ctx.ub_bulk_wgrad:
                    if ub_obj_wgrad.is_fp8_ubuf():
                        dgrad = reduce_scatter_out
                    else:
                        dgrad = ub_obj_wgrad.get_buffer(local_chunk=True).clone()

            # --------------------------------------------------
            # Grad weight has been computed...
            # --------------------------------------------------

            # Don't return grad bias if not needed
            if not ctx.use_bias:
                grad_bias = None

            # Make sure all tensor-parallel communication is finished
            if inputmat_total_work is not None:
                inputmat_total_work.wait()
                inputmat_total_work = None
            if dgrad_work is not None:
                dgrad_work.wait()
                dgrad_work = None

        if ctx.requires_wgrad:
            # Handle custom DDP from mcore.
            if (
                ctx.fuse_wgrad_accumulation
                and weight is not None
                and hasattr(weight, "grad_added_to_main_grad")
            ):
                weight.grad_added_to_main_grad = True
                if getattr(weight, "zero_out_wgrad", False):
                    wgrad = get_dummy_wgrad(
                        list(weight.main_grad.shape),
                        weight.dtype,
                        zero=True,
                    )
                else:
                    wgrad = get_dummy_wgrad(
                        list(weight.main_grad.shape),
                        weight.dtype,
                    )
            elif ctx.fuse_wgrad_accumulation:
                wgrad = None
        else:
            wgrad = None

        # Update FP8 scaling factors if needed
        if ctx.reduce_and_update_bwd_fp8_tensors and not is_graph_capturing():
            nvtx_range_push(f"{nvtx_label}.reduce_and_update_fp8_tensors")
            FP8GlobalStateManager.reduce_and_update_fp8_tensors(forward=False)
            nvtx_range_pop(f"{nvtx_label}.reduce_and_update_fp8_tensors")

        # Scatter fp8 weight buffers
        if ctx.fp8 and not isinstance(weight, QuantizedTensorBase):
            _fsdp_scatter_tensors(ctx.fsdp_group, weight_fp8)
        return (
            wgrad,
            dgrad.view(ctx.inp_shape) if ctx.requires_dgrad else None,
            grad_bias,
            None,  # is_first_microbatch
            None,  # fp8
            None,  # fp8_calibration
            None,  # wgrad_store
            None,  # input_quantizer
            None,  # weight_quantizer
            None,  # output_quantizer
            None,  # grad_input_quantizer
            None,  # grad_weight_quantizer
            None,  # grad_output_quantizer
            None,  # fuse_wgrad_accumulation
            None,  # cpu_offloading
            None,  # tp_group
            None,  # tp_size
            None,  # sequence_parallel
            None,  # tensor_parallel
            None,  # activation_dtype
            None,  # parallel_mode
            None,  # is_grad_enabled
            None,  # ub_overlap_rs_fprop
            None,  # ub_overlap_ag_dgrad
            None,  # ub_overlap_ag_fprop
            None,  # ub_overlap_rs_dgrad
            None,  # ub_bulk_dgrad
            None,  # ub_bulk_wgrad
            None,  # ub_name
            None,  # fp8_output
            None,  # fsdp_group
            None,  # module
            None,  # skip_fp8_weight_update
            None,  # symmetric_ar_type
            None,  # save_original_input
            None,  # debug
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
        skip_fp8_weight_update = FP8GlobalStateManager.get_skip_fp8_weight_update_tensor()
    else:
        skip_fp8_weight_update = None
    if skip_fp8_weight_update is not None:
        is_first_microbatch = False

    if self.ub_overlap_rs_fprop:
        if get_ub(self.ub_name + "_fprop").is_fp8_ubuf():
            fp8_output = True
    if self.ub_overlap_rs_dgrad:
        if get_ub(self.ub_name + "_dgrad").is_fp8_ubuf():
            fp8_grad = True

    with (
        torch.cuda.device(getattr(self, list(self.named_parameters())[0][0]).device),
        self.prepare_forward(
            inp,
            allow_non_contiguous=isinstance(inp, QuantizedTensor),
        ) as inp,
    ):

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
            linear_fn = _Linear.apply
            args = []
        else:
            linear_fn = _Linear.forward
            args = [None]
        args += (
            weight_tensor,
            inp,
            bias_tensor if (self.apply_bias and not self.gemm_bias_unfused_add) else None,
            is_first_microbatch,
            self.fp8,
            self.fp8_calibration,
            self.wgrad_store,
            input_quantizer,
            weight_quantizer,
            output_quantizer,
            grad_input_quantizer,
            grad_weight_quantizer,
            grad_output_quantizer,
            self.fuse_wgrad_accumulation,
            is_cpu_offload_enabled(),
            self.tp_group,
            self.tp_size,
            self.sequence_parallel,
            self.tp_size > 1,
            self.activation_dtype,
            self.parallel_mode,
            torch.is_grad_enabled(),
            self.ub_overlap_rs_fprop,
            self.ub_overlap_ag_dgrad,
            self.ub_overlap_ag_fprop,
            self.ub_overlap_rs_dgrad,
            self.ub_bulk_dgrad,
            self.ub_bulk_wgrad,
            self.ub_name,
            fp8_output,
            self.fsdp_group,
            self,
            skip_fp8_weight_update,
            self.symmetric_ar_type,
            self.save_original_input,
            debug,
        )
        out = linear_fn(*args)
    if self.gemm_bias_unfused_add:
        out = out + cast_if_needed(bias_tensor, self.activation_dtype)

    if self.return_bias:
        return out, cast_if_needed(bias_tensor, self.activation_dtype)
    return out


te.pytorch.Linear.forward = hack_forward


class TERowParallelLinearForTest(CommonOpsForTest, TERowParallelLinear):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config: ModelParallelConfig,
        init_method: Callable,
        bias: bool,
        input_is_parallel: bool = True,
        skip_bias_add: bool = True,
        is_expert: bool = False,
        tp_comm_buffer_name: Optional[str] = "fc2",
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
        hook_activation: bool = False,
    ):
        TERowParallelLinear.__init__(
            self,
            input_size=input_size,
            output_size=output_size,
            config=config,
            init_method=init_method,
            bias=bias,
            input_is_parallel=input_is_parallel,
            skip_bias_add=skip_bias_add,
            is_expert=is_expert,
            tp_comm_buffer_name=tp_comm_buffer_name,
            tp_group=tp_group,
        )
        CommonOpsForTest.__init__(
            self, hook_activation=hook_activation, module_name="TERowParallelLinear"
        )

    @nvtx_decorator(message="TERowParallelLinearForward")
    def _forward(self, x: torch.Tensor):
        """Forward."""
        # the instrumentation is in the code monkey patched by us
        res = super().forward(x)
        return res

    def forward(self, x: torch.Tensor):
        self.activation_hook.clear()
        with torch.autograd.graph.saved_tensors_hooks(
            self.activation_hook.save_hook, self.activation_hook.load_hook
        ):
            ret = self._forward(x)
        return ret
