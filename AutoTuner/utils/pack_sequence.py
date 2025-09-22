from typing import Tuple

import torch
from megatron.core.packed_seq_params import PackedSeqParams
from torch import Tensor

from verl.models.mcore.model_forward import gptmodel_forward
from verl.models.mcore.util import (
    postprocess_packed_seqs,
    preprocess_packed_seqs,
    recover_left_padding,
    remove_left_padding,
)


def generate_thd_input(
    input_ids: Tensor,
    attention_mask: Tensor,
) -> Tuple[Tensor, PackedSeqParams]:
    """Generate text sequence with thread-level parallelism"""
    input_ids_rmpad, packed_seq_params = preprocess_packed_seqs(
        input_ids, attention_mask, pre_process=True
    )
    input_ids_rmpad = input_ids_rmpad.contiguous()
    return input_ids_rmpad, packed_seq_params
