import abc
import os
from abc import ABC
from typing import Any, Iterator, List, Optional, Tuple

import torch
from megatron.core import tensor_parallel
from megatron.core.transformer.transformer_config import TransformerConfig
from transformers import PretrainedConfig

from AutoTuner.utils.memory import MemoryTracker, MemoryTrackerContext
from AutoTuner.utils.model_inputs import get_thd_model_input_from_bshd
from AutoTuner.utils.nested_dict import NestedDict
from AutoTuner.utils.nvtx import nvtx_decorator, nvtx_range_pop, nvtx_range_push
from AutoTuner.utils.structs import InputTestCase
from AutoTuner.utils.timing import Timer, TimerContext

from ..ops.common import CommonOpsForTest
from ..profile.configs.config_struct import ProfileMode

os.environ["NVTE_NVTX_ENABLED"] = "1"


class TestCommon(ABC):
    def __init__(
        self,
        hf_config: PretrainedConfig,
        profile_mode: int = 0,
        warmup_iters: int = 2,
    ):
        self.op: CommonOpsForTest = None
        self.module_name = "common"

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
        if profile_mode == ProfileMode.collect_data:
            self.timing_db = NestedDict()
            self.memory_db = {"weights": {}, "activations": NestedDict()}
        self.model_config = hf_config
        self.profile_mode = profile_mode
        self.warmup_iters = warmup_iters

    @abc.abstractclassmethod
    def prepare_input(self, test_case: InputTestCase, batch_data_generator: Iterator):
        pass

    def run_test(self, test_case: InputTestCase, batch_data_generator: Iterator):
        inputs = self.prepare_input(
            test_case=test_case, batch_data_generator=batch_data_generator
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
            output.requires_grad_(True)
            nvtx_range_push("backward")
            output.sum().backward()
            nvtx_range_pop("backward")
        else:
            """
            When collecting data
            """
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
            name = f"{self.module_name} forward {test_case}"
            with TimerContext(name) as timer_ctx:
                output = self.op(*inputs)
            self.timing_db[self.module_name]["forward"] = test_case.set_nested_dict(
                self.timing_db[self.module_name]["forward"], timer_ctx.result
            )

            # Call backward function - force output to require grad
            output.requires_grad_(True)
            name = f"{self.module_name} backward {test_case}"
            with TimerContext(name) as timer_ctx:
                # Create a dummy loss tensor and call backward
                loss = output.sum()
                loss.backward()
            self.timing_db[self.module_name]["backward"] = test_case.set_nested_dict(
                self.timing_db[self.module_name]["backward"], timer_ctx.result
            )

            self.memory_db["activations"] = test_case.set_nested_dict(
                self.memory_db["activations"],
                {self.module_name: self.op.get_activation_memory()},
            )

    def get_results(self) -> Tuple[dict, dict]:
        assert (
            not self.profile_mode
        ), f"Nothing to return when not using data collection mode"
        return self.timing_db, self.memory_db
