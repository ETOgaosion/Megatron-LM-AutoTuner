from dataclasses import dataclass, field
from typing import Any, Tuple, Optional

from .nested_dict import NestedDict


@dataclass
class InputTestCase:
    batch_size: int  # mini batch size
    micro_batch_size: int  # micro batch size
    seqlen: int
    max_token_len: int | None = None
    shape: str = "thd"  # or "bshd"
    system: str = "megatron"  # or "fsdp"

    def __str__(self):
        return f"batch_size={self.batch_size}, micro_batch_size={self.micro_batch_size}, seqlen={self.seqlen}, max_token_len={self.max_token_len}, shape={self.shape}, system={self.system}"

    def __hash__(self):
        return hash(
            (
                self.batch_size,
                self.micro_batch_size,
                self.seqlen,
                self.max_token_len,
                self.shape,
                self.system,
            )
        )

    tensor_model_parallel_size: int = field(default=1)
    pipeline_model_parallel_size: int = field(default=1)
    virtual_pipeline_model_parallel_size: Optional[int] = field(default=None)
    context_parallel_size: int = field(default=1)
    expert_parallel_size: int = field(default=1)
    expert_tensor_parallel_size: int = field(default=1)
    sequence_parallel_enabled: bool = field(default=False)

    def set_nested_dict(self, nested_dict: NestedDict, value: Any) -> NestedDict:
        """
        Set the value in the nested_dict according to the attributes of the InputTestCase.

        Args:
            nested_dict (NestedDict): The NestedDict to set the value in.
            value (Any): The value to set.
        Returns:
            NestedDict: The updated NestedDict.
        """

        if self.shape == "thd":
            nested_dict[f"batch_size={self.batch_size}"][f"seqlen={self.seqlen}"][
                f"shape={self.shape}"
            ][f"max_token_len={self.max_token_len}"][f"system={self.system}"] = value
            return nested_dict
        else:
            nested_dict[f"batch_size={self.batch_size}"][f"seqlen={self.seqlen}"][
                f"shape={self.shape}"
            ][f"micro_batch_size={self.micro_batch_size}"][
                f"system={self.system}"
            ] = value
            return nested_dict
