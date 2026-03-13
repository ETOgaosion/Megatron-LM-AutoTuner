from __future__ import annotations

from AutoTuner.Profiler.expert_parallel import (
    AllToAllBandwidthModel,
    ExpertParallelProfileResult,
    ExpertParallelStrategyDecider,
)


def test_decide_from_profile_prefers_fastest_fully_overlapped_ep() -> None:
    profile = ExpertParallelProfileResult(
        model_name="Qwen/Qwen3-30B-A3B",
        tp_size=2,
        cp_size=2,
        seqlen=4096,
        max_token_len=4096,
        nprocs_per_node=8,
        batch_size=128,
        micro_batch_size=2,
        shape="thd",
        system="megatron",
        hidden_size=4096,
        num_moe_experts=8,
        moe_router_topk=2,
        dtype_bytes=2,
        profiled_ep_size=1,
        attention_forward_time_s=0.003,
        attention_backward_time_s=0.006,
        attention_wgrad_time_s=0.0005,
        mlp_forward_time_s=0.004,
        mlp_backward_time_s=0.006,
        mlp_wgrad_time_s=0.003,
    )

    report = ExpertParallelStrategyDecider.decide_from_profile(
        profile=profile,
        profile_path="expert_parallel_profile.json",
        bandwidth_model=AllToAllBandwidthModel(
            intra_node_bandwidth_gbps=300.0,
            inter_node_bandwidth_gbps=50.0,
        ),
    )

    assert report.decision.ep_size == 2
    assert report.decision.can_fully_overlap is True
    assert report.candidate_evaluations[0].ep_size == 1
    assert report.candidate_evaluations[-1].ep_size == 8


def test_decide_from_profile_filters_invalid_custom_candidates() -> None:
    profile = ExpertParallelProfileResult(
        model_name="Qwen/Qwen3-30B-A3B",
        tp_size=1,
        cp_size=1,
        seqlen=2048,
        max_token_len=2048,
        nprocs_per_node=8,
        batch_size=32,
        micro_batch_size=1,
        shape="thd",
        system="megatron",
        hidden_size=2048,
        num_moe_experts=6,
        moe_router_topk=1,
        dtype_bytes=2,
        profiled_ep_size=1,
        attention_forward_time_s=0.002,
        attention_backward_time_s=0.003,
        attention_wgrad_time_s=0.001,
        mlp_forward_time_s=0.006,
        mlp_backward_time_s=0.008,
        mlp_wgrad_time_s=0.004,
    )

    report = ExpertParallelStrategyDecider.decide_from_profile(
        profile=profile,
        profile_path="expert_parallel_profile.json",
        bandwidth_model=AllToAllBandwidthModel(),
        ep_candidates=[1, 2, 4, 6],
    )

    assert [candidate.ep_size for candidate in report.candidate_evaluations] == [1, 2, 6]
