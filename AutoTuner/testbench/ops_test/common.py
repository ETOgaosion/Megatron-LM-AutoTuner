import abc
import os
from abc import ABC
from typing import Any, Iterator, List, Optional, Tuple

import megatron.core.parallel_state as mpu
import torch
from megatron.core import tensor_parallel
from megatron.core.transformer.transformer_config import TransformerConfig
from tensordict import TensorDict
from transformers import PretrainedConfig

from AutoTuner.utils.batch import average_microbatch_metric
from AutoTuner.utils.gpu_info import GPU_PEAK_FLOPS
from AutoTuner.utils.memory import MemoryTracker, MemoryTrackerContext, get_memory_str
from AutoTuner.utils.model_inputs import get_thd_model_input_from_bshd
from AutoTuner.utils.nested_dict import NestedDict
from AutoTuner.utils.nvtx import nvtx_decorator, nvtx_range_pop, nvtx_range_push
from AutoTuner.utils.structs import InputTestCase
from AutoTuner.utils.timing import Timer, TimerContext
from AutoTuner.utils.tp_overlap import destroy_ub, initialize_tp_communicators

from ..ops.common import CommonOpsForTest
from ..profile.configs.config_struct import ProfileMode
from .theoretical_base import TheoreticalCalculation

os.environ["NVTE_NVTX_ENABLED"] = "1"


class TestCommon(TheoreticalCalculation):
    def __init__(
        self,
        tf_config: TransformerConfig,
        hf_config: PretrainedConfig,
        profile_mode: int = 0,
        warmup_iters: int = 2,
        theoretical_flops: bool = False,
        theoretical_activations: bool = False,
        tp_comm_overlap_cfg: str = None,
    ):
        super().__init__()
        self.op: CommonOpsForTest = None
        self.module_name = "common"
        if profile_mode == ProfileMode.collect_data:
            self.timing_db = NestedDict()
            self.memory_db = {"weights": {}, "activations": NestedDict()}
            self.micro_batch_results = []
        self.tf_config = tf_config
        self.hf_config = hf_config
        self.profile_mode = profile_mode
        self.warmup_iters = warmup_iters
        self.theoretical_flops = theoretical_flops
        self.theoretical_activations = theoretical_activations
        self.tp_comm_overlap_cfg = tp_comm_overlap_cfg

        """
        timing_db structure:
        {
            "Embedding":
            {
                "forward":
                {
                    InputTestCase(batch_size=..., micro_batch_size=..., seqlen=..., max_token_len=..., shape='thd', system='megatron') : time_in_seconds,
                    ...
                },
                "backward":
                {
                    InputTestCase(batch_size=..., micro_batch_size=..., seqlen=..., max_token_len=..., shape='thd', system='megatron') : time_in_seconds,
                    ...
                }
            },
            ...
        }
        """
        """memory_db structure:
        {
            "weights": {
                "Embedding": memory_in_bytes,
                ...
            },
            "activations": {
                "Embedding":
                {
                    InputTestCase(batch_size=..., micro_batch_size=..., seqlen=..., max_token_len=..., shape='thd', system='megatron'): memory_in_bytes,
                    ...
                },
                ...
            }
        }
        """

    @abc.abstractmethod
    def prepare_input(self, test_case: InputTestCase, micro_batch: TensorDict):
        raise NotImplementedError

    @abc.abstractmethod
    def calculate_tokens(
        self, test_case: InputTestCase, micro_batch: TensorDict, inputs: Any
    ) -> int:
        raise NotImplementedError

    def run_micro_batch(self, test_case: InputTestCase, inputs: List[Any], tokens: int):
        if (
            mpu.get_tensor_model_parallel_world_size() > 1
            and self.tf_config.tp_comm_overlap
            and test_case.shape == "thd"
            and tokens is not None
            and tokens > 0
        ):
            initialize_tp_communicators(
                tp_comm_overlap_cfg=self.tp_comm_overlap_cfg,
                tokens=tokens,
                hidden_size=self.hf_config.hidden_size,
            )
        if self.profile_mode == ProfileMode.nsys_profile:
            """
            When using nsys profile
            """

            # Warmup iterations
            torch.cuda.profiler.stop()
            if self.warmup_iters > 0:
                for _ in range(self.warmup_iters):
                    self.op(*inputs)

            # Call forward function - force output to require grad
            torch.cuda.profiler.start()
            output = self.op(*inputs)
            torch.cuda.profiler.stop()

            # Call backward function - force output to require grad
            if isinstance(output, tuple):
                output = output[0]
            output.requires_grad_(True)
            torch.cuda.profiler.start()
            nvtx_range_push("backward")
            output.sum().backward()
            nvtx_range_pop("backward")
            torch.cuda.profiler.stop()
        elif self.profile_mode == ProfileMode.torch_profiler:
            """
            When using torch profiler, we do warmup outside
            """

            # Forward pass
            output = self.op(*inputs)

            # Backward pass
            if isinstance(output, tuple):
                output = output[0]
            output.requires_grad_(True)
            nvtx_range_push("backward")
            output.sum().backward()
            nvtx_range_pop("backward")
        else:
            """
            When collecting data
            """
            self.micro_batch_results.append({})

            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            if self.warmup_iters > 0:
                for _ in range(self.warmup_iters):
                    self.op(*inputs)
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            # Call forward function - force output to require grad
            with TimerContext() as timer_ctx:
                output = self.op(*inputs)
            self.micro_batch_results[-1]["forward"] = timer_ctx.elapsed_time

            # Notice that an operator may have multiple outputs
            if isinstance(output, tuple):
                output = output[0]

            # Call backward function - force output to require grad
            output.requires_grad_(True)
            with TimerContext() as timer_ctx:
                # Create a dummy loss tensor and call backward
                loss = output.sum()
                loss.backward()
            self.micro_batch_results[-1]["backward"] = timer_ctx.elapsed_time
            self.micro_batch_results[-1][
                "activation_memory"
            ] = self.op.activation_hook.get_activation_memory()
        if (
            mpu.get_tensor_model_parallel_world_size() > 1
            and self.tf_config.tp_comm_overlap
            and test_case.shape == "thd"
            and tokens is not None
            and tokens > 0
        ):
            destroy_ub()

    def run_test(self, test_case: InputTestCase, batch_data_generator: Iterator):
        for batch_data in batch_data_generator:
            inputs = self.prepare_input(test_case, batch_data)
            tokens = self.calculate_tokens(test_case, batch_data, inputs)
            self.run_micro_batch(test_case, inputs, tokens)

        if self.profile_mode == ProfileMode.collect_data:
            avg_forward_time = average_microbatch_metric(
                self.micro_batch_results, "forward"
            )
            avg_backward_time = average_microbatch_metric(
                self.micro_batch_results, "backward"
            )
            if self.theoretical_flops:
                theo_flops = self.calc_theoretical_flops(test_case)
                forward_flops = theo_flops.get("forward", 0)
                backward_flops = theo_flops.get("backward", 0)

                # forward
                forward_leaf = {
                    "real": f"avg {avg_forward_time:.6f}s",
                    "estimated_flops": forward_flops,
                    "estimated_time": (
                        forward_flops / GPU_PEAK_FLOPS if forward_flops > 0 else 0
                    ),
                }
                # backward
                backward_leaf = {
                    "real": f"avg {avg_backward_time:.6f}s",
                    "estimated_flops": backward_flops,
                    "estimated_time": (
                        backward_flops / GPU_PEAK_FLOPS if backward_flops > 0 else 0
                    ),
                }

                self.timing_db[self.module_name]["forward"] = test_case.set_nested_dict(
                    self.timing_db[self.module_name]["forward"], forward_leaf
                )
                self.timing_db[self.module_name]["backward"] = (
                    test_case.set_nested_dict(
                        self.timing_db[self.module_name]["backward"], backward_leaf
                    )
                )
            else:
                self.timing_db[self.module_name]["forward"] = test_case.set_nested_dict(
                    self.timing_db[self.module_name]["forward"],
                    f"avg {avg_forward_time:.6f}s",
                )
                self.timing_db[self.module_name]["backward"] = (
                    test_case.set_nested_dict(
                        self.timing_db[self.module_name]["backward"],
                        f"avg {avg_backward_time:.6f}s",
                    )
                )
            avg_activation_bytes = average_microbatch_metric(
                self.micro_batch_results, "activation_memory"
            )
            if self.theoretical_activations:

                theo_mem = self.calc_theoretical_memory(test_case)
                estimated_activations = theo_mem.get("activations", {}).get(
                    "activations", 0
                )

                leaf_data_full = {
                    self.module_name: {
                        "real": f"avg {get_memory_str(avg_activation_bytes, human_readable=True)}",
                        "estimated": get_memory_str(
                            estimated_activations, human_readable=True
                        ),
                    }
                }
                temp_db_full = test_case.set_nested_dict(NestedDict(), leaf_data_full)
                self.memory_db["activations"].merge(temp_db_full)
            else:
                leaf_data = {
                    self.module_name: f"avg {get_memory_str(avg_activation_bytes, human_readable=True)}"
                }
                temp_db = test_case.set_nested_dict(NestedDict(), leaf_data)
                self.memory_db["activations"].merge(temp_db)
            # Clear micro batch results
            self.micro_batch_results = []

    def get_results(self) -> Tuple[dict, dict]:
        assert (
            not self.profile_mode
        ), f"Nothing to return when not using data collection mode"
        return self.timing_db, self.memory_db
