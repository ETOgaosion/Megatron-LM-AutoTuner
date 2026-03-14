import abc
import logging
import os
from abc import ABC
from typing import Dict

import torch

from AutoTuner.utils.memory import ActivationHook, MemoryTracker, get_memory_str


class CommonOpsForTest(ABC):
    def __init__(
        self,
        hook_activation: bool = False,
        module_name: str = "common_ops",
        logging_level: int = logging.INFO,
        online: bool = False,  # Control the hook's activation memory calculation logic: when True, calculate memory online to avoid issues with cleared tensors
    ):
        self.activation_hook = ActivationHook(
            enable=hook_activation,
            module_name=module_name,
            logging_level=logging_level,
            online=online,
        )
        self.module_name = module_name

    def get_activation_memory(self) -> Dict[str, int]:
        return {"activations": self.activation_hook.get_activation_memory()}

    def debug_stage_sync(self, stage: str):
        if os.getenv("AUTOTUNER_STAGE_SYNC_DEBUG") != "1":
            return

        rank = 0
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
        print(f"[rank{rank}] stage-sync begin: {self.module_name}.{stage}", flush=True)
        torch.cuda.synchronize()
        print(f"[rank{rank}] stage-sync end: {self.module_name}.{stage}", flush=True)
