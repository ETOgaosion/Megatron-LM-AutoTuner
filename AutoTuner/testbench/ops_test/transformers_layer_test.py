from typing import Any, Dict

import os
import torch
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from transformers import PretrainedConfig
from typing_extensions import override
from typing import Optional
from contextlib import nullcontext

from AutoTuner.testbench.ops.transformer_layer import TransformerLayerForTest
from AutoTuner.testbench.profile.configs.config_struct import ProfileMode
from AutoTuner.utils.structs import InputTestCase
from AutoTuner.utils.memory import MemoryTrackerContext, get_memory_str
from AutoTuner.utils.hidden_status_gen import HiddenStatusGenerator

from megatron.core.transformer import TransformerLayerSubmodules
from megatron.core import parallel_state
from megatron.core import tensor_parallel
from megatron.core.process_groups_config import ProcessGroupCollection
from AutoTuner.testbench.ops_test.test_with_hiddens import TestWithHiddenInputs
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)

from .common import TestCommon


os.environ["NVTE_NVTX_ENABLED"] = "1"


class TestTransformerLayer(TestWithHiddenInputs):
    def __init__(
        self,
        tf_config: TransformerConfig,
        hf_config: PretrainedConfig,
        profile_mode: int = 0,
        warmup_iters: int = 2,
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
            tp_group=tp_group,
            theoretical_flops=theoretical_flops,
            theoretical_activations=theoretical_activations,
            tp_comm_overlap_cfg=tp_comm_overlap_cfg,
        )
        self.module_name = "TransformerLayer"
        self.tp_group = tp_group if tp_group is not None else parallel_state.get_tensor_model_parallel_group()

        if profile_mode == ProfileMode.collect_data:
            with MemoryTrackerContext(self.module_name) as memory_tracker_ctx:
                # Resolve layer submodules: prefer explicit config, else use GPT layer spec
                if hasattr(tf_config, "layer_submodules") and tf_config.layer_submodules is not None:
                    layer_submodules = tf_config.layer_submodules
                else:
                    use_te = getattr(tf_config, "transformer_impl", "local") == "transformer_engine"
                    try:
                        if use_te:
                            spec = get_gpt_layer_with_transformer_engine_spec()
                        else:
                            spec = get_gpt_layer_local_spec()
                        layer_submodules = spec.submodules
                    except Exception:
                        layer_submodules = None

                self.op = TransformerLayerForTest(
                    tf_config,
                    submodules=layer_submodules,
                    layer_number=tf_config.num_layers,
                    hidden_dropout=tf_config.hidden_dropout if tf_config.hidden_dropout is not None else 0.1,
                    pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
                    vp_stage=parallel_state.get_virtual_pipeline_model_parallel_rank(),
                    hook_activation=(profile_mode == ProfileMode.collect_data),
                )

            detailed_mem_report = memory_tracker_ctx.get_result()

            self.memory_db["weights"][self.module_name] = detailed_mem_report
        else:
            if hasattr(tf_config, "layer_submodules") and tf_config.layer_submodules is not None:
                layer_submodules = tf_config.layer_submodules
            else:
                use_te = getattr(tf_config, "transformer_impl", "local") == "transformer_engine"
                try:
                    if use_te:
                        spec = get_gpt_layer_with_transformer_engine_spec()
                    else:
                        spec = get_gpt_layer_local_spec()
                    layer_submodules = spec.submodules
                except Exception:
                    layer_submodules = None
            self.op = TransformerLayerForTest(
                tf_config,
                submodules=layer_submodules,
                layer_number=tf_config.num_layers,
                hidden_dropout=tf_config.hidden_dropout if tf_config.hidden_dropout is not None else 0.1,
                pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
                vp_stage=parallel_state.get_virtual_pipeline_model_parallel_rank(),
                hook_activation=(profile_mode == ProfileMode.collect_data),
            )

    @override
    def calculate_tokens(self, test_case: InputTestCase, micro_batch: Any, inputs: Any) -> int:
            return 0

    @override
    def calc_theoretical_memory(self, test_case: InputTestCase) -> Dict[str, int]:
        return {"activations": {"activations": 0}}

    @override
    def calc_theoretical_flops(self, test_case: InputTestCase) -> Dict[str, float]:
        return {"forward": 0, "backward": 0}
