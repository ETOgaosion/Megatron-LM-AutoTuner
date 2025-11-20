import traceback
import warnings

import megatron.core.parallel_state as mpu
from megatron.core.utils import get_te_version, is_te_min_version


def initialize_tp_communicators(
    tp_comm_overlap_cfg: str = None,
    tokens: int = None,
    seq_length: int = None,
    micro_batch_size: int = None,
    hidden_size: int = None,
    fp8: bool = None,
    first_last_layers_bf16: bool = False,
    num_layers_at_start_in_bf16: int = 0,
    num_layers_at_end_in_bf16: int = 0,
    tp_comm_bootstrap_backend: str = "nccl",
):
    # Copy from Megatron-LM/megatron/training/initialize.py: _initialize_tp_communicators

    """initializing the communicators with user buffers for high-performance tensor-model-parallel
    communication overlap"""

    assert (tokens is not None and tokens > 0) ^ (
        seq_length is not None
        and seq_length > 0
        and micro_batch_size is not None
        and micro_batch_size > 0
    ), "Either tokens or (seq_length, micro_batch_size) must be provided and greater than 0"

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

    if seq_length is not None:
        input_shape = [
            (seq_length * micro_batch_size * mpu.get_tensor_model_parallel_world_size())
            // mpu.get_context_parallel_world_size(),
            hidden_size,
        ]
    else:
        input_shape = [
            tokens
            * mpu.get_tensor_model_parallel_world_size()
            // mpu.get_context_parallel_world_size(),
            hidden_size,
        ]
    print(f"Initializing TP Communicators with User Buffers for shape {input_shape}...")

    # The process group with the target bootstrap backend is created in Transformer Engine.
    te_module.base.initialize_ub(
        shape=input_shape,
        tp_size=mpu.get_tensor_model_parallel_world_size(),
        bootstrap_backend=tp_comm_bootstrap_backend,
    )


def destroy_ub():
    """Destroy the user buffers created by Transformer Engine."""
    try:
        from transformer_engine.pytorch import module as te_module

        te_module.base.destroy_ub()
    except ImportError:
        warnings.warn(
            "Transformer Engine is not installed, skip destroying user buffers."
        )
        traceback.print_exc()
