import copy
from typing import Any, Iterable, Tuple

import megatron.core.parallel_state as mpu
import tensordict
import torch
from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
from megatron.core.packed_seq_params import PackedSeqParams
from tensordict import TensorDict
from transformers import PretrainedConfig

from verl.utils.megatron.pipeline_parallel import make_batch_generator
from verl.utils.model import compute_position_id_with_mask, create_random_mask
from verl.utils.seqlen_balancing import rearrange_micro_batches

from .pack_sequence import generate_thd_input
from .structs import InputTestCase

"""
    Follow some convinient implementation in verl

    Return:
    - input_ids: (b, s) or (1, total_nnz) if thd
    - attention_mask: (b, s) or None if thd
    - position_ids: (b, s) or (1, total_nnz) if
    - packed_seq_params: Any, only for thd
"""


def _get_one_model_input_bshd(
    model_config: PretrainedConfig, batch_size: int, seqlen: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_ids = torch.randint(
        low=0, high=model_config.vocab_size, size=(batch_size, seqlen), device="cuda"
    )
    attention_mask = create_random_mask(
        input_ids=input_ids,
        max_ratio_of_left_padding=0,
        max_ratio_of_valid_token=0.9,
        min_ratio_of_valid_token=0.8,
    )
    position_ids = compute_position_id_with_mask(attention_mask)
    return input_ids, attention_mask, position_ids, None


def _get_one_model_input_fsdp_thd(
    model_config: PretrainedConfig, batch_size: int, seqlen: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    input_ids, attention_mask, position_ids = _get_one_model_input_bshd(
        model_config, batch_size, seqlen
    )
    input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)
    input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)
    position_ids_rmpad = index_first_axis(
        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
    ).transpose(0, 1)
    return input_ids_rmpad, attention_mask, position_ids_rmpad, None


def _get_one_model_input_megatron_thd(
    model_config: PretrainedConfig, batch_size: int, seqlen: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    input_ids, attention_mask, position_ids = _get_one_model_input_bshd(
        model_config, batch_size, seqlen
    )
    input_ids_rmpad, packed_seq_params = generate_thd_input(
        input_ids=input_ids, attention_mask=attention_mask
    )
    return input_ids_rmpad, attention_mask, position_ids, packed_seq_params


def get_one_model_input(
    model_config: PretrainedConfig,
    batch_size: int,
    seqlen: int,
    shape: str = "bshd",
    system: str = "megatron",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any]:
    if shape == "thd":
        if system == "megatron":
            return _get_one_model_input_megatron_thd(model_config, batch_size, seqlen)
        elif system == "fsdp":
            return _get_one_model_input_fsdp_thd(model_config, batch_size, seqlen)
    else:
        return _get_one_model_input_bshd(model_config, batch_size, seqlen)


def get_thd_model_input_from_bshd(
    batch: TensorDict, system: str = "megatron"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any]:
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    attention_mask = attention_mask.to(bool)
    position_ids = batch["position_ids"]
    batch_size, seqlen = attention_mask.shape[:2]
    if system == "megatron":
        input_ids_rmpad, packed_seq_params = generate_thd_input(
            input_ids=input_ids, attention_mask=attention_mask
        )
        return input_ids_rmpad, attention_mask, position_ids, packed_seq_params
    elif system == "fsdp":
        input_ids_rmpad, indices, *_ = unpad_input(
            input_ids.unsqueeze(-1), attention_mask
        )
        input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)
        position_ids_rmpad = index_first_axis(
            rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
        ).transpose(0, 1)
        return input_ids_rmpad, attention_mask, position_ids_rmpad, None
    else:
        raise ValueError(f"system {system} not supported")


class DataSets:
    """
    For real RLHF training, sequence packing shall use this simulated dataset to prepare inputs.

    This class prepares sequence balanced data
    """

    def __init__(
        self,
        model_config: PretrainedConfig,
        test_cases: list[InputTestCase],
        use_dynamic_bsz_balance: bool = True,
        vpp_size: int | None = None,
    ):
        self.model_config = model_config
        self.test_cases = test_cases
        self.use_dynamic_bsz_balance = use_dynamic_bsz_balance
        self.vpp_size = vpp_size

        self.data = {}
        self.data_batch_generators = {}
        if torch.distributed.get_rank() == 0:
            for test_case in self.test_cases:
                batch_size = test_case.batch_size
                micro_batch_size = test_case.micro_batch_size
                seqlen = test_case.seqlen
                max_token_len = test_case.max_token_len
                shape = test_case.shape
                system = test_case.system

                input_ids, attention_mask, position_ids, packed_seq_params = (
                    _get_one_model_input_bshd(model_config, batch_size, seqlen)
                )
                batch = TensorDict(
                    {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "position_ids": position_ids,
                        "packed_seq_params": packed_seq_params,
                    },
                    batch_size=batch_size,
                    # device=torch.cuda.current_device(),
                    device="cpu",
                )
                if shape == "bshd":
                    micro_batches = batch.split(micro_batch_size)
                    self.data[test_case] = micro_batches
                else:
                    assert shape == "thd", f"shape {shape} not supported"
                    micro_batches, _ = rearrange_micro_batches(
                        batch,
                        max_token_len=max_token_len,
                        num_batches_divided_by=self.vpp_size,
                        use_dynamic_bsz_balance=self.use_dynamic_bsz_balance,
                        same_micro_num_in_dp=False,
                        # dp_group=mpu.get_data_parallel_group(),
                    )
                    self.data[test_case] = micro_batches
                self.data_batch_generators[test_case] = make_batch_generator(
                    self.data[test_case],
                    vpp_size=self.vpp_size if self.vpp_size is not None else 1,
                )
            test_cases_micro_batches = [
                self.data[test_case] for test_case in self.test_cases
            ]
            torch.distributed.broadcast_object_list(test_cases_micro_batches, src=0)
        else:
            test_cases_micro_batches = [None for _ in self.test_cases]
            torch.distributed.broadcast_object_list(test_cases_micro_batches, src=0)
            for test_case, micro_batches in zip(
                self.test_cases, test_cases_micro_batches
            ):
                self.data[test_case] = micro_batches
                self.data_batch_generators[test_case] = make_batch_generator(
                    self.data[test_case],
                    vpp_size=self.vpp_size if self.vpp_size is not None else 1,
                )

    def get_batch_generator(self, test_case: InputTestCase):
        return copy.deepcopy(self.data_batch_generators[test_case])
