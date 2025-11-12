import warnings

import megatron.core.parallel_state as mpu
from megatron.core.utils import get_te_version, is_te_min_version


def initialize_tp_communicators(
    tp_comm_overlap_cfg: str = None,
    seq_length: int = None,
    micro_batch_size: int = None,
    hidden_size: int = None,
    fp8: bool = False,
    first_last_layers_bf16: bool = False,
    num_layers_at_start_in_bf16: int = 0,
    num_layers_at_end_in_bf16: int = 0,
    tp_comm_bootstrap_backend: str = "nccl",
):
    # Copy from Megatron-LM/megatron/training/initialize.py: _initialize_tp_communicators

    """initializing the communicators with user buffers for high-performance tensor-model-parallel
    communication overlap"""

    try:
        import transformer_engine
        import yaml
        from transformer_engine.pytorch import module as te_module

    except ImportError:
        raise RuntimeError(
            "Tensor Parallel Communication/GEMM Overlap optimization needs 'yaml' and "
            "'transformer_engine' packages"
        )

    if tp_comm_overlap_cfg is not None:
        with open(tp_comm_overlap_cfg, "r") as stream:
            ub_cfgs = yaml.safe_load(stream)
    else:
        ub_cfgs = {}

    input_shape = [
        (seq_length * micro_batch_size) // mpu.get_context_parallel_world_size(),
        hidden_size,
    ]

    if is_te_min_version("2.7.0"):
        UserBufferQuantizationMode = te_module.base.UserBufferQuantizationMode
        quantization_modes = [
            UserBufferQuantizationMode.FP8 if fp8 else UserBufferQuantizationMode.NONE
        ]
        if (
            fp8
            and first_last_layers_bf16
            and (num_layers_at_start_in_bf16 > 0 or num_layers_at_end_in_bf16 > 0)
        ):
            quantization_modes.append(UserBufferQuantizationMode.NONE)
        # The process group with the target bootstrap backend is created in Transformer Engine.
        te_module.base.initialize_ub(
            shape=input_shape,
            tp_size=mpu.get_tensor_model_parallel_world_size(),
            quantization_modes=quantization_modes,
            ub_cfgs=ub_cfgs,
            bootstrap_backend=tp_comm_bootstrap_backend,
        )
    elif is_te_min_version("1.9.0"):
        # The process group with the target bootstrap backend is created in Transformer Engine.
        te_module.base.initialize_ub(
            shape=input_shape,
            tp_size=mpu.get_tensor_model_parallel_world_size(),
            use_fp8=(fp8 is not None),
            ub_cfgs=ub_cfgs,
            bootstrap_backend=tp_comm_bootstrap_backend,
        )
    else:
        if tp_comm_bootstrap_backend != "mpi":
            warnings.warn(
                f"Transformer Engine v{get_te_version()} supports only MPI bootstrap backend."
            )
        # Create a MPI process group to help with TP communication overlap bootstrap.
        mpu.create_group(backend="mpi", group_desc="TP_BOOTSTRAP_GROUP_MPI")

        te_module.base.initialize_ub(
            shape=input_shape,
            tp_size=mpu.get_tensor_model_parallel_world_size(),
            use_fp8=(fp8 is not None),
            ub_cfgs=ub_cfgs,
        )
