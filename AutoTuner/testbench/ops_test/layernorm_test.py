import os
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
from megatron.core import parallel_state as mpu
from megatron.core import tensor_parallel
from megatron.core.transformer.transformer_config import TransformerConfig
from tensordict import TensorDict
from transformers import PretrainedConfig
from typing_extensions import override

from AutoTuner.utils.memory import MemoryTracker, MemoryTrackerContext, get_memory_str
from AutoTuner.utils.model_inputs import get_thd_model_input_from_bshd
from AutoTuner.utils.nested_dict import NestedDict
from AutoTuner.utils.structs import InputTestCase
from AutoTuner.utils.timing import Timer, TimerContext

from ..ops.layernorm import LayerNormForTest
from ..profile.configs.config_struct import ProfileMode
from .common import TestCommon

os.environ["NVTE_NVTX_ENABLED"] = "1"


class TestLayerNorm(TestCommon):
    def __init__(
        self,
        tf_config: TransformerConfig,
        hf_config: PretrainedConfig,
        profile_mode: int = 0,
        warmup_iters: int = 2,
        profile_iters: int = 2,
        theoretical_flops: bool = False,
        theoretical_activations: bool = False,
        tp_comm_overlap_cfg: str = None,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        super().__init__(
            hf_config=hf_config,
            tf_config=tf_config,
            profile_mode=profile_mode,
            warmup_iters=warmup_iters,
            profile_iters=profile_iters,
            theoretical_flops=theoretical_flops,
            theoretical_activations=theoretical_activations,
            tp_comm_overlap_cfg=tp_comm_overlap_cfg,
        )
        self.module_name = "LayerNorm"

        if profile_mode == ProfileMode.collect_data:
            with MemoryTrackerContext(self.module_name) as memory_tracker_ctx:
                self.op = LayerNormForTest(
                    tf_config,
                    hf_config,
                    hook_activation=(profile_mode == ProfileMode.collect_data),
                )

            detailed_mem_report = memory_tracker_ctx.get_result()
            hidden_size = hf_config.hidden_size
            if tf_config.normalization == "RMSNorm":
                num_params = hidden_size
            else:
                num_params = 2 * hidden_size
            dtype = self.op.norm.weight.dtype
            bytes_per_param = torch.tensor([], dtype=dtype).element_size()
            total_param_bytes = num_params * bytes_per_param
            estimated_weight_mem_str = get_memory_str(
                total_param_bytes, human_readable=True
            )
            detailed_mem_report["estimated_peak_mem_diff"] = estimated_weight_mem_str
            self.memory_db["weights"][self.module_name] = detailed_mem_report

        else:
            self.op = LayerNormForTest(
                tf_config,
                hf_config,
                hook_activation=False,
            )

    @override
    def prepare_input(self, test_case: InputTestCase, micro_batch: TensorDict):
        device = torch.cuda.current_device()
        dtype = torch.float16
        hidden_size = self.hf_config.hidden_size
        if test_case.shape == "thd":
            if "cu_seqlens" in micro_batch:
                cu_seqlens = micro_batch["cu_seqlens"].to(device)
                num_tokens = cu_seqlens[-1].item() - cu_seqlens[0].item()
            else:
                num_tokens = test_case.batch_size * test_case.seqlen

            hidden_states = torch.randn(
                num_tokens, hidden_size, device=device, dtype=dtype, requires_grad=True
            )
        elif test_case.shape == "bshd":
            hidden_states = torch.randn(
                test_case.micro_batch_size,
                test_case.seqlen,
                hidden_size,
                device=device,
                dtype=dtype,
                requires_grad=True,
            )
            hidden_states = hidden_states.transpose(0, 1).contiguous()
        else:
            raise ValueError(
                f"Unsupported shape format: {test_case.shape}. "
                f"Supported: 'thd', 'bshd'"
            )

        return (hidden_states,)

    @override
    def calculate_tokens(
        self, test_case: InputTestCase, micro_batch: TensorDict, inputs: Any
    ) -> int:
        hidden_states = inputs[0]

        if test_case.shape == "thd":
            num_tokens = hidden_states.size(0)

        elif test_case.shape == "bshd":
            if test_case.sequence_parallel_enabled:
                seq_len_local = hidden_states.size(0)
                batch_size = hidden_states.size(1)
                num_tokens = seq_len_local * batch_size
            else:
                seq_len = hidden_states.size(0)
                batch_size = hidden_states.size(1)
                num_tokens = seq_len * batch_size
        else:
            raise ValueError(f"Unsupported shape format: {test_case.shape}")
        if test_case.context_parallel_size > 1:
            num_tokens = num_tokens // test_case.context_parallel_size
        return num_tokens

    @override
    def calc_theoretical_memory(self, test_case: InputTestCase) -> Dict[str, int]:
        return {"activations": {"activations": 0}}

    @override
    def calc_theoretical_flops(self, test_case: InputTestCase) -> Dict[str, float]:
        """
        Calculate theoretical FLOPS from the perspective of a single rank.
        """
        hidden_size = self.hf_config.hidden_size
        num_tokens = test_case.micro_batch_size * test_case.seqlen

        if test_case.context_parallel_size > 1:
            num_tokens = num_tokens // test_case.context_parallel_size

        if self.tf_config.normalization == "RMSNorm":
            forward_flops = 2 * num_tokens * hidden_size
            # recompute
            backward_flops = 6 * num_tokens * hidden_size
        else:
            forward_flops = 4 * num_tokens * hidden_size
            # recompute
            backward_flops = 12 * num_tokens * hidden_size
        return {"forward": forward_flops, "backward": backward_flops}
