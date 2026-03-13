"""Torchrun entry point for TransformerLayer activation profiling."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.process_groups_config import ProcessGroupCollection

from AutoTuner.Profiler.activations.runner import PROFILED_PARTS
from AutoTuner.testbench.ops.transformer_layer import TransformerLayerForTest
from AutoTuner.utils.config import (
    get_hf_model_config,
    get_mcore_model_config_from_hf_config,
)
from AutoTuner.utils.distributed import destroy_distributed, init_distributed_multi_nodes
from AutoTuner.utils.hidden_status_gen import HiddenStatusGenerator
from AutoTuner.utils.model_inputs import DataSets
from AutoTuner.utils.structs import InputTestCase
from AutoTuner.utils.timing import TimerContext


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile TransformerLayer activation parts for step 3."
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--tp-size", type=int, required=True)
    parser.add_argument("--cp-size", type=int, required=True)
    parser.add_argument("--seqlen", type=int, required=True)
    parser.add_argument("--max-token-len", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--micro-batch-size", type=int, default=2)
    parser.add_argument("--shape", type=str, default="thd", choices=["thd", "bshd"])
    parser.add_argument("--system", type=str, default="megatron")
    parser.add_argument("--warmup-iters", type=int, default=2)
    parser.add_argument("--profile-iters", type=int, default=3)
    parser.add_argument("--output-path", required=True)
    return parser.parse_args()


def _tensor_bytes(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        if value.device.type != "cuda" or value.dtype == torch.bool:
            return 0
        return value.numel() * value.element_size()
    if isinstance(value, (list, tuple)):
        return sum(_tensor_bytes(item) for item in value)
    if isinstance(value, dict):
        return sum(_tensor_bytes(item) for item in value.values())
    return 0


def _resolve_attr(root: object, path: tuple[str, ...]) -> object | None:
    current = root
    for item in path:
        if not hasattr(current, item):
            return None
        current = getattr(current, item)
    return current


def _register_hooks(layer: torch.nn.Module) -> tuple[dict[str, list[int]], list[Any]]:
    samples = {part: [] for part in PROFILED_PARTS}
    handles = []
    module_paths = {
        "attn_norm": [("input_layernorm",)],
        "core_attn": [("self_attention", "core_attention")],
        "attn_proj": [("self_attention", "linear_proj")],
        "mlp_norm": [("pre_mlp_layernorm",)],
        "expert_fc1": [("mlp", "experts", "linear_fc1")],
    }

    for part, candidate_paths in module_paths.items():
        for path in candidate_paths:
            module = _resolve_attr(layer, path)
            if module is None:
                continue

            def _make_hook(part_name: str):
                def _hook(_module: torch.nn.Module, inputs: tuple[Any, ...]) -> None:
                    samples[part_name].append(_tensor_bytes(inputs))

                return _hook

            handles.append(module.register_forward_pre_hook(_make_hook(part)))
            break
    return samples, handles


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
    )


def _build_layer_and_inputs(args: argparse.Namespace):
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
    }
    tf_config = get_mcore_model_config_from_hf_config(hf_config, **override_tf_config)
    spec = get_gpt_layer_with_transformer_engine_spec(
        num_experts=tf_config.num_moe_experts,
        multi_latent_attention=tf_config.multi_latent_attention,
        qk_layernorm=tf_config.qk_layernorm,
        moe_grouped_gemm=tf_config.moe_grouped_gemm,
    )
    layer = TransformerLayerForTest(
        tf_config,
        submodules=spec.submodules,
        layer_number=tf_config.num_layers,
        hidden_dropout=tf_config.hidden_dropout if tf_config.hidden_dropout is not None else 0.1,
        pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
        vp_stage=parallel_state.get_virtual_pipeline_model_parallel_rank(),
        hook_activation=False,
    )
    layer.train()

    test_case = _build_test_case(args)
    datasets = DataSets(
        hf_config,
        [test_case],
        fix_compute_amount=True,
        use_dynamic_bsz_balance=True,
        vpp_size=parallel_state.get_virtual_pipeline_model_parallel_world_size(),
    )
    hidden_status_generator = HiddenStatusGenerator(
        tf_config,
        hf_config,
        tp_group=parallel_state.get_tensor_model_parallel_group(),
    )
    return layer, datasets, hidden_status_generator, test_case


def _run_one_iteration(
    layer: torch.nn.Module,
    datasets: DataSets,
    hidden_status_generator: HiddenStatusGenerator,
    test_case: InputTestCase,
    hook_samples: dict[str, list[int]],
) -> tuple[float, float, dict[str, int]]:
    batch_generator = datasets.get_batch_generator(test_case)
    batch = next(batch_generator)
    with torch.no_grad():
        inputs = hidden_status_generator.prepare_input(test_case, batch)

    for values in hook_samples.values():
        values.clear()

    if isinstance(layer, torch.nn.Module):
        layer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    dist.barrier()

    with TimerContext(name="forward", cuda_sync=True) as forward_timer:
        output = layer(*inputs)
    if isinstance(output, tuple):
        output = output[0]
    output.requires_grad_(True)

    with TimerContext(name="backward", cuda_sync=True) as backward_timer:
        loss = output.sum()
        loss.backward()

    activation_bytes = {
        part: max(values) if values else 0 for part, values in hook_samples.items()
    }

    if isinstance(layer, torch.nn.Module):
        layer.zero_grad(set_to_none=True)
    del batch
    del inputs
    del output
    del loss
    torch.cuda.empty_cache()

    return (
        float(forward_timer.elapsed_time),
        float(backward_timer.elapsed_time),
        activation_bytes,
    )


def _average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> int:
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    init_distributed_multi_nodes(tp=args.tp_size, cp=args.cp_size, pp=1)
    try:
        layer, datasets, hidden_status_generator, test_case = _build_layer_and_inputs(args)
        hook_samples, hook_handles = _register_hooks(layer)

        for _ in range(args.warmup_iters):
            _run_one_iteration(
                layer, datasets, hidden_status_generator, test_case, hook_samples
            )

        forward_times: list[float] = []
        backward_times: list[float] = []
        activation_history = {part: [] for part in PROFILED_PARTS}
        for _ in range(args.profile_iters):
            forward_time, backward_time, activation_bytes = _run_one_iteration(
                layer, datasets, hidden_status_generator, test_case, hook_samples
            )
            forward_times.append(forward_time)
            backward_times.append(backward_time)
            for part in PROFILED_PARTS:
                activation_history[part].append(activation_bytes.get(part, 0))

        for handle in hook_handles:
            handle.remove()

        rank_summary = {
            "rank": dist.get_rank(),
            "forward_time_s": _average(forward_times),
            "backward_time_s": _average(backward_times),
            "total_time_s": _average(forward_times) + _average(backward_times),
            "activation_bytes_by_part": {
                part: max(values) if values else 0
                for part, values in activation_history.items()
            },
        }
        gathered = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, rank_summary)

        if dist.get_rank() == 0:
            profile = {
                "model_name": args.model_name,
                "tp_size": args.tp_size,
                "cp_size": args.cp_size,
                "seqlen": args.seqlen,
                "max_token_len": args.max_token_len,
                "batch_size": args.batch_size,
                "micro_batch_size": args.micro_batch_size,
                "shape": args.shape,
                "system": args.system,
                "forward_time_s": max(item["forward_time_s"] for item in gathered),
                "backward_time_s": max(item["backward_time_s"] for item in gathered),
                "total_time_s": max(item["total_time_s"] for item in gathered),
                "activation_bytes_by_part": {
                    part: max(item["activation_bytes_by_part"].get(part, 0) for item in gathered)
                    for part in PROFILED_PARTS
                },
                "activation_present_by_part": {
                    part: any(
                        item["activation_bytes_by_part"].get(part, 0) > 0 for item in gathered
                    )
                    for part in PROFILED_PARTS
                },
                "rank_summaries": gathered,
            }
            with open(args.output_path, "w") as handle:
                json.dump(profile, handle, indent=2, sort_keys=True)
                handle.write("\n")
        dist.barrier()
    finally:
        destroy_distributed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
