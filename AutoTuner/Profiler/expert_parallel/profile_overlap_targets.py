"""Torchrun entry point to profile MoE overlap windows for EP selection."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import torch
import torch.distributed as dist
from torch.profiler import ProfilerActivity, profile

from AutoTuner.testbench.ops_test.self_attention_test import TestSelfAttention
from AutoTuner.testbench.ops_test.te_grouped_mlp_test import TestTEGroupedMLP
from AutoTuner.testbench.profile.configs.config_struct import ProfileMode
from AutoTuner.utils.config import (
    get_hf_model_config,
    get_mcore_model_config_from_hf_config,
)
from AutoTuner.utils.distributed import destroy_distributed, init_distributed_multi_nodes
from AutoTuner.utils.model_inputs import DataSets
from AutoTuner.utils.structs import InputTestCase


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile attention/MLP overlap windows for MoE EP step 3."
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--tp-size", type=int, required=True)
    parser.add_argument("--cp-size", type=int, required=True)
    parser.add_argument("--seqlen", type=int, required=True)
    parser.add_argument("--max-token-len", type=int, required=True)
    parser.add_argument("--nprocs-per-node", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--micro-batch-size", type=int, default=2)
    parser.add_argument("--shape", type=str, default="thd", choices=["thd", "bshd"])
    parser.add_argument("--system", type=str, default="megatron")
    parser.add_argument("--warmup-iters", type=int, default=2)
    parser.add_argument("--profile-iters", type=int, default=3)
    parser.add_argument("--dtype-bytes", type=int, default=2)
    parser.add_argument("--output-path", required=True)
    return parser.parse_args()


def _build_test_case(args: argparse.Namespace) -> InputTestCase:
    return InputTestCase(
        batch_size=args.batch_size,
        micro_batch_size=args.micro_batch_size,
        seqlen=args.seqlen,
        max_token_len=args.max_token_len,
        shape=args.shape,
        system=args.system,
        tensor_model_parallel_size=args.tp_size,
        context_parallel_size=args.cp_size,
        expert_parallel_size=1,
    )


def _build_configs(args: argparse.Namespace):
    hf_config = get_hf_model_config(args.model_name)
    override_tf_config = {
        "persist_layer_norm": True,
        "bias_activation_fusion": True,
        "apply_rope_fusion": True,
        "moe_permute_fusion": True,
        "deallocate_pipeline_outputs": True,
        "gradients_accumulation_fusion": True,
        "fine_grained_activation_offloading": False,
        "recompute_granularity": None,
        "recompute_modules": None,
        "tensor_model_parallel_size": args.tp_size,
        "context_parallel_size": args.cp_size,
        "expert_model_parallel_size": 1,
        "sequence_parallel": args.tp_size > 1,
    }
    tf_config = get_mcore_model_config_from_hf_config(hf_config, **override_tf_config)
    return hf_config, tf_config


def _build_datasets(hf_config: Any, test_case: InputTestCase) -> DataSets:
    return DataSets(
        hf_config,
        [test_case],
        fix_compute_amount=True,
        use_dynamic_bsz_balance=True,
        vpp_size=None,
    )


def _prepare_inputs(tester: Any, test_case: InputTestCase, datasets: DataSets):
    batch_generator = datasets.get_batch_generator(test_case)
    batch = next(batch_generator)
    with torch.no_grad():
        inputs = tester.prepare_input(test_case, batch)
    return batch, inputs


def _cleanup_iteration(
    tester: Any, batch: Any, inputs: tuple[Any, ...], output: Any | None = None
) -> None:
    if isinstance(tester.op, torch.nn.Module):
        tester.op.zero_grad(set_to_none=True)
    del batch
    for value in inputs:
        if isinstance(value, torch.Tensor):
            del value
    if isinstance(output, torch.Tensor):
        del output
    torch.cuda.empty_cache()


def _run_forward_backward_once(
    tester: Any, test_case: InputTestCase, datasets: DataSets
) -> None:
    batch, inputs = _prepare_inputs(tester, test_case, datasets)
    torch.cuda.synchronize()
    dist.barrier()
    output = tester.op(*inputs)
    if isinstance(output, tuple):
        output = output[0]
    output.requires_grad_(True)
    loss = output.sum()
    loss.backward()
    torch.cuda.synchronize()
    dist.barrier()
    del loss
    _cleanup_iteration(tester, batch, inputs, output)


def _collect_forward_backward_time(
    tester: Any, test_case: InputTestCase, datasets: DataSets
) -> tuple[float, float]:
    batch, inputs = _prepare_inputs(tester, test_case, datasets)
    tokens = tester.calculate_tokens(test_case, batch, inputs)
    tester.run_micro_batch(test_case, list(inputs), tokens)
    result = tester.micro_batch_results.pop()
    _cleanup_iteration(tester, batch, inputs)
    return float(result["forward"]), float(result["backward"])


def _collect_wgrad_time(
    tester: Any, test_case: InputTestCase, datasets: DataSets, profile_iters: int
) -> tuple[float, list[str]]:
    notes: list[str] = []
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        for _ in range(profile_iters):
            _run_forward_backward_once(tester, test_case, datasets)
            prof.step()

    wgrad_cuda_time_us = sum(
        event.self_cuda_time_total
        for event in prof.key_averages()
        if "wgrad" in event.key.lower()
    )
    if wgrad_cuda_time_us <= 0:
        notes.append(
            f"{tester.module_name}: profiler did not expose wgrad ranges; using total backward time as fallback."
        )
    return float(wgrad_cuda_time_us) / 1e6, notes


def _profile_operator(
    tester: Any,
    test_case: InputTestCase,
    datasets: DataSets,
    args: argparse.Namespace,
) -> tuple[float, float, float, list[str]]:
    forward_time_s, backward_time_s = _collect_forward_backward_time(
        tester, test_case, datasets
    )

    for _ in range(args.warmup_iters):
        _run_forward_backward_once(tester, test_case, datasets)

    wgrad_time_s, notes = _collect_wgrad_time(
        tester, test_case, datasets, args.profile_iters
    )
    if wgrad_time_s <= 0:
        wgrad_time_s = backward_time_s
    return forward_time_s, backward_time_s, wgrad_time_s, notes


def main() -> int:
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    init_distributed_multi_nodes(tp=args.tp_size, cp=args.cp_size, ep=1, pp=1)
    try:
        hf_config, tf_config = _build_configs(args)
        test_case = _build_test_case(args)
        datasets = _build_datasets(hf_config, test_case)

        attention_tester = TestSelfAttention(
            tf_config=tf_config,
            hf_config=hf_config,
            tp_group=None,
            profile_mode=ProfileMode.collect_data,
            warmup_iters=args.warmup_iters,
            profile_iters=args.profile_iters,
            theoretical_flops=False,
            theoretical_activations=False,
        )
        mlp_tester = TestTEGroupedMLP(
            tf_config=tf_config,
            hf_config=hf_config,
            tp_group=None,
            profile_mode=ProfileMode.collect_data,
            warmup_iters=args.warmup_iters,
            profile_iters=args.profile_iters,
            theoretical_flops=False,
            theoretical_activations=False,
        )

        attn_fwd, attn_bwd, attn_wgrad, attn_notes = _profile_operator(
            attention_tester, test_case, datasets, args
        )
        mlp_fwd, mlp_bwd, mlp_wgrad, mlp_notes = _profile_operator(
            mlp_tester, test_case, datasets, args
        )

        rank_summary = {
            "rank": dist.get_rank(),
            "attention_forward_time_s": attn_fwd,
            "attention_backward_time_s": attn_bwd,
            "attention_wgrad_time_s": attn_wgrad,
            "mlp_forward_time_s": mlp_fwd,
            "mlp_backward_time_s": mlp_bwd,
            "mlp_wgrad_time_s": mlp_wgrad,
            "notes": attn_notes + mlp_notes,
        }
        gathered = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, rank_summary)

        if dist.get_rank() == 0:
            notes: list[str] = []
            for item in gathered:
                for note in item.get("notes", []):
                    if note not in notes:
                        notes.append(note)

            profile_payload = {
                "model_name": args.model_name,
                "tp_size": args.tp_size,
                "cp_size": args.cp_size,
                "seqlen": args.seqlen,
                "max_token_len": args.max_token_len,
                "nprocs_per_node": args.nprocs_per_node,
                "batch_size": args.batch_size,
                "micro_batch_size": args.micro_batch_size,
                "shape": args.shape,
                "system": args.system,
                "hidden_size": int(hf_config.hidden_size),
                "num_moe_experts": int(getattr(tf_config, "num_moe_experts", 0) or 0),
                "moe_router_topk": int(getattr(tf_config, "moe_router_topk", 1) or 1),
                "dtype_bytes": args.dtype_bytes,
                "profiled_ep_size": 1,
                "attention_forward_time_s": max(
                    item["attention_forward_time_s"] for item in gathered
                ),
                "attention_backward_time_s": max(
                    item["attention_backward_time_s"] for item in gathered
                ),
                "attention_wgrad_time_s": max(
                    item["attention_wgrad_time_s"] for item in gathered
                ),
                "mlp_forward_time_s": max(item["mlp_forward_time_s"] for item in gathered),
                "mlp_backward_time_s": max(
                    item["mlp_backward_time_s"] for item in gathered
                ),
                "mlp_wgrad_time_s": max(item["mlp_wgrad_time_s"] for item in gathered),
                "notes": notes,
            }
            if profile_payload["num_moe_experts"] <= 0:
                raise ValueError("Profiled model config does not expose num_moe_experts.")
            with open(args.output_path, "w") as handle:
                json.dump(profile_payload, handle, indent=2, sort_keys=True)
                handle.write("\n")
        dist.barrier()
    finally:
        destroy_distributed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
