import abc
import logging
from abc import ABC
from typing import Dict

from AutoTuner.utils.memory import ActivationHook, MemoryTracker, get_memory_str


class CommonOpsForTest(ABC):
    def __init__(
        self,
        hook_activation: bool = False,
        module_name: str = "common_ops",
        logging_level: int = logging.INFO,
    ):
        self.activation_hook = ActivationHook(
            enable=hook_activation, module_name=module_name, logging_level=logging_level
        )

    def get_activation_memory(self) -> Dict[str, int]:
        return {"activations": self.activation_hook.get_activation_memory()}
