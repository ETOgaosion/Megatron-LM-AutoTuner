import abc
from typing import Dict

from AutoTuner.utils.structs import InputTestCase


class TheoreticalCalculation(abc.ABC):
    """
    An interface for calculating the theoretical performance of an operator.

    Test classes for operators should inherit this and implement its methods.
    """

    @abc.abstractmethod
    def calc_theoretical_memory(self, test_case: InputTestCase) -> Dict[str, int]:
        """
        Calculates the theoretical memory usage in bytes.

        Args:
            test_case: An object containing operator parameters.

        Returns:
            A dict with memory usage, e.g., {"weights": 8e6, "activations": 16e6}.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def calc_theoretical_flops(self, test_case: InputTestCase) -> Dict[str, float]:
        """
        Calculates the theoretical Floating Point Operations (FLOPs).

        Args:
            test_case: An object containing operator parameters.

        Returns:
            A dict with FLOPs, e.g., {"forward": 1.2e9, "backward": 2.4e9}.
        """
        raise NotImplementedError
