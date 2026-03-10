from AutoTuner.runtime.baseline.simulator import (
    StageParamStats,
    StageTimingStats,
    build_pp_stage_layer_counts,
    simulate_full_iteration,
    simulate_pp_step_durations,
    simulate_train_pipeline_1f1b,
)


def test_build_pp_stage_layer_counts_with_vpp():
    assert build_pp_stage_layer_counts(total_layers=22, pp_size=4, vpp_size=2) == [
        4,
        6,
        6,
        6,
    ]


def test_simulate_pp_step_durations_uses_bottleneck_stage():
    assert simulate_pp_step_durations([5.0, 1.0, 5.0], num_microbatches=2) == [
        5.0,
        5.0,
        5.0,
        5.0,
    ]


def test_simulate_train_pipeline_1f1b_uses_training_step_count():
    result = simulate_train_pipeline_1f1b([1.0, 1.0], [1.0, 1.0], num_microbatches=2)

    assert result["total_time_s"] == 6.0
    assert result["num_scheduled_tasks"] == 8


def test_simulate_full_iteration_scales_pp_stage_time_by_full_layer_count():
    result = simulate_full_iteration(
        observed_iteration_time_s=9.0,
        num_microbatches=2,
        full_stage_layer_counts=[4, 2],
        runtime_stage_layer_counts=[1, 1],
        stage_param_stats=[
            StageParamStats(0, 1, 100, 100),
            StageParamStats(1, 1, 100, 100),
        ],
        stage_timing_stats=[
            StageTimingStats(0, 1, 1.0, 1.0, 2.0, 2.0, 2),
            StageTimingStats(1, 1, 1.0, 1.0, 2.0, 2.0, 2),
        ],
        dp_world_size=1,
        include_runtime_dp_in_observed_time=False,
        bandwidth_gbps=1.0,
        latency_s=0.0,
    )

    assert result["stage_layer_scales"] == [4.0, 2.0]
    assert result["runtime_pp_schedule_time_s"] == 6.0
    assert result["runtime_pp_schedule_scale"] == 1.5
    assert result["full_stage_forward_times_s"] == [6.0, 3.0]
    assert result["full_stage_backward_times_s"] == [6.0, 3.0]
    assert result["simulated_pp_compute_time_s"] == 24.0
    assert result["simulated_time_s"] == 24.0


def test_simulate_full_iteration_adds_dp_allreduce_bottleneck():
    result = simulate_full_iteration(
        observed_iteration_time_s=3.0,
        num_microbatches=1,
        full_stage_layer_counts=[1, 1],
        runtime_stage_layer_counts=[1, 1],
        stage_param_stats=[
            StageParamStats(0, 1, 4_000_000_000, 4_000_000_000),
            StageParamStats(1, 1, 2_000_000_000, 2_000_000_000),
        ],
        stage_timing_stats=[
            StageTimingStats(0, 1, 0.5, 1.0, 0.5, 1.0, 1),
            StageTimingStats(1, 1, 0.5, 1.0, 0.5, 1.0, 1),
        ],
        dp_world_size=4,
        include_runtime_dp_in_observed_time=False,
        bandwidth_gbps=1.0,
        latency_s=0.0,
    )

    assert result["simulated_pp_compute_time_s"] == 3.0
    assert result["simulated_dp_allreduce_time_s"] == 6.0
    assert result["simulated_time_s"] == 9.0
