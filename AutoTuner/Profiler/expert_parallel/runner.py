"""MoE expert-parallel profiling runner and step-3 EP decider."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import asdict, dataclass, field
from typing import Any, Optional, Sequence


EP_COMM_PHASES = (
    "dispatch_forward",
    "combine_backward",
    "combine_forward",
    "dispatch_backward",
)


@dataclass(frozen=True)
class AllToAllBandwidthModel:
    """Bandwidth model for EP all-to-all communication."""

    intra_node_bandwidth_gbps: float = 300.0
    inter_node_bandwidth_gbps: float = 50.0
    latency_seconds: float = 0.0

    def __post_init__(self) -> None:
        if self.intra_node_bandwidth_gbps <= 0:
            raise ValueError("intra_node_bandwidth_gbps must be positive.")
        if self.inter_node_bandwidth_gbps <= 0:
            raise ValueError("inter_node_bandwidth_gbps must be positive.")
        if self.latency_seconds < 0:
            raise ValueError("latency_seconds must be non-negative.")

    def estimate_seconds(self, num_bytes: int, tier: str) -> float:
        if num_bytes <= 0:
            return 0.0
        bandwidth_gbps = (
            self.intra_node_bandwidth_gbps
            if tier == "intra_node"
            else self.inter_node_bandwidth_gbps
        )
        bandwidth_bytes_per_s = bandwidth_gbps * 1e9
        return self.latency_seconds + (num_bytes / bandwidth_bytes_per_s)

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class ExpertParallelProfilingConfig:
    """Inputs required to profile MoE overlap windows and decide EP."""

    model_name: str
    output_dir: str
    tp_size: int
    cp_size: int
    seqlen: int
    max_token_len: int
    nprocs_per_node: int
    batch_size: int = 128
    micro_batch_size: int = 2
    shape: str = "thd"
    system: str = "megatron"
    warmup_iters: int = 2
    profile_iters: int = 3
    dtype_bytes: int = 2
    bandwidth_model: AllToAllBandwidthModel = field(
        default_factory=AllToAllBandwidthModel
    )
    ep_candidates: Optional[list[int]] = None
    profile_script: Optional[str] = None

    def __post_init__(self) -> None:
        if self.tp_size <= 0 or self.cp_size <= 0:
            raise ValueError("tp_size and cp_size must be positive.")
        if self.seqlen <= 0 or self.max_token_len <= 0:
            raise ValueError("seqlen and max_token_len must be positive.")
        if self.nprocs_per_node <= 0:
            raise ValueError("nprocs_per_node must be positive.")
        if self.batch_size <= 0 or self.micro_batch_size <= 0:
            raise ValueError("batch_size and micro_batch_size must be positive.")
        if self.shape not in {"thd", "bshd"}:
            raise ValueError("shape must be either 'thd' or 'bshd'.")
        if self.warmup_iters < 0 or self.profile_iters <= 0:
            raise ValueError("warmup_iters must be >= 0 and profile_iters must be > 0.")
        if self.dtype_bytes <= 0:
            raise ValueError("dtype_bytes must be positive.")
        if self.ep_candidates is not None:
            if not self.ep_candidates:
                raise ValueError("ep_candidates must not be empty.")
            if any(candidate <= 0 for candidate in self.ep_candidates):
                raise ValueError("ep_candidates must contain only positive integers.")

    @property
    def world_size(self) -> int:
        return self.tp_size * self.cp_size

    @property
    def profile_path(self) -> str:
        return os.path.join(self.output_dir, "expert_parallel_profile.json")

    @property
    def decision_path(self) -> str:
        return os.path.join(self.output_dir, "expert_parallel_strategy_report.json")


@dataclass(frozen=True)
class ExpertParallelProfileResult:
    """Normalized overlap-window profile used for EP selection."""

    model_name: str
    tp_size: int
    cp_size: int
    seqlen: int
    max_token_len: int
    nprocs_per_node: int
    batch_size: int
    micro_batch_size: int
    shape: str
    system: str
    hidden_size: int
    num_moe_experts: int
    moe_router_topk: int
    dtype_bytes: int
    profiled_ep_size: int
    attention_forward_time_s: float
    attention_backward_time_s: float
    attention_wgrad_time_s: float
    mlp_forward_time_s: float
    mlp_backward_time_s: float
    mlp_wgrad_time_s: float
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive.")
        if self.num_moe_experts <= 0:
            raise ValueError("num_moe_experts must be positive.")
        if self.moe_router_topk <= 0:
            raise ValueError("moe_router_topk must be positive.")
        if self.dtype_bytes <= 0:
            raise ValueError("dtype_bytes must be positive.")
        if self.profiled_ep_size <= 0:
            raise ValueError("profiled_ep_size must be positive.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExpertParallelProfileResult":
        return cls(
            model_name=str(payload["model_name"]),
            tp_size=int(payload["tp_size"]),
            cp_size=int(payload["cp_size"]),
            seqlen=int(payload["seqlen"]),
            max_token_len=int(payload["max_token_len"]),
            nprocs_per_node=int(payload["nprocs_per_node"]),
            batch_size=int(payload["batch_size"]),
            micro_batch_size=int(payload["micro_batch_size"]),
            shape=str(payload["shape"]),
            system=str(payload["system"]),
            hidden_size=int(payload["hidden_size"]),
            num_moe_experts=int(payload["num_moe_experts"]),
            moe_router_topk=int(payload["moe_router_topk"]),
            dtype_bytes=int(payload["dtype_bytes"]),
            profiled_ep_size=int(payload.get("profiled_ep_size", 1)),
            attention_forward_time_s=float(payload["attention_forward_time_s"]),
            attention_backward_time_s=float(payload["attention_backward_time_s"]),
            attention_wgrad_time_s=float(payload["attention_wgrad_time_s"]),
            mlp_forward_time_s=float(payload["mlp_forward_time_s"]),
            mlp_backward_time_s=float(payload["mlp_backward_time_s"]),
            mlp_wgrad_time_s=float(payload["mlp_wgrad_time_s"]),
            notes=list(payload.get("notes") or []),
        )


@dataclass(frozen=True)
class ExpertParallelCandidateEvaluation:
    """One evaluated EP candidate."""

    ep_size: int
    local_experts: int
    bandwidth_tier: str
    comm_payload_bytes_per_phase: int
    comm_time_s_by_phase: dict[str, float]
    overlap_window_s_by_phase: dict[str, float]
    overlap_slack_s_by_phase: dict[str, float]
    scaled_mlp_forward_time_s: float
    scaled_mlp_backward_time_s: float
    scaled_mlp_wgrad_time_s: float
    exposed_comm_time_s: float
    objective_time_s: float
    can_fully_overlap: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExpertParallelDecision:
    """Selected EP configuration."""

    ep_size: int
    local_experts: int
    bandwidth_tier: str
    can_fully_overlap: bool
    exposed_comm_time_s: float
    objective_time_s: float
    comm_payload_bytes_per_phase: int
    comm_time_s_by_phase: dict[str, float]
    overlap_window_s_by_phase: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExpertParallelStrategyReport:
    """Full EP step-3 output combining the profile and the final decision."""

    profile_path: str
    profile: ExpertParallelProfileResult
    bandwidth_model: AllToAllBandwidthModel
    decision: ExpertParallelDecision
    candidate_evaluations: list[ExpertParallelCandidateEvaluation]
    assumptions: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_path": self.profile_path,
            "profile": self.profile.to_dict(),
            "bandwidth_model": self.bandwidth_model.to_dict(),
            "decision": self.decision.to_dict(),
            "candidate_evaluations": [
                candidate.to_dict() for candidate in self.candidate_evaluations
            ],
            "assumptions": self.assumptions,
        }


class ExpertParallelStrategyDecider:
    """Profiles overlap windows and decides a MoE expert-parallel size."""

    def __init__(self, config: ExpertParallelProfilingConfig):
        self.config = config

    def run(self, skip_profiling: bool = False) -> ExpertParallelStrategyReport:
        os.makedirs(self.config.output_dir, exist_ok=True)

        if skip_profiling:
            if not os.path.exists(self.config.profile_path):
                raise FileNotFoundError(
                    f"Expert-parallel profile not found at {self.config.profile_path}. "
                    "Disable skip_profiling or provide an existing profile."
                )
        else:
            self._run_profiling()

        profile = self._load_profile(self.config.profile_path)
        report = self.decide_from_profile(
            profile=profile,
            profile_path=self.config.profile_path,
            bandwidth_model=self.config.bandwidth_model,
            ep_candidates=self.config.ep_candidates,
        )
        self._save_report(report)
        return report

    def _run_profiling(self) -> None:
        if self.config.profile_script:
            self._run_profile_script(self.config.profile_script)
            return

        cmd = [
            "torchrun",
            f"--nproc_per_node={self.config.world_size}",
            "-m",
            "AutoTuner.Profiler.expert_parallel.profile_overlap_targets",
            "--model-name",
            self.config.model_name,
            "--tp-size",
            str(self.config.tp_size),
            "--cp-size",
            str(self.config.cp_size),
            "--seqlen",
            str(self.config.seqlen),
            "--max-token-len",
            str(self.config.max_token_len),
            "--nprocs-per-node",
            str(self.config.nprocs_per_node),
            "--batch-size",
            str(self.config.batch_size),
            "--micro-batch-size",
            str(self.config.micro_batch_size),
            "--shape",
            self.config.shape,
            "--system",
            self.config.system,
            "--warmup-iters",
            str(self.config.warmup_iters),
            "--profile-iters",
            str(self.config.profile_iters),
            "--dtype-bytes",
            str(self.config.dtype_bytes),
            "--output-path",
            self.config.profile_path,
        ]

        env = os.environ.copy()
        env.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
        env.setdefault("NVTE_FLASH_ATTN", "1")
        env.setdefault("NVTE_FUSED_ATTN", "0")
        env.setdefault("NVTE_NVTX_ENABLED", "1")
        try:
            subprocess.run(cmd, env=env, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:  # pragma: no cover - env dependent
            error = exc.stderr or exc.stdout or str(exc)
            raise RuntimeError(
                f"Expert-parallel overlap profiling failed: {error}"
            ) from exc

    @staticmethod
    def _run_profile_script(script_path: str) -> None:
        try:
            subprocess.run(
                ["bash", script_path], check=True, capture_output=True, text=True
            )
        except subprocess.CalledProcessError as exc:  # pragma: no cover - env dependent
            error = exc.stderr or exc.stdout or str(exc)
            raise RuntimeError(f"EP profiling script failed: {error}") from exc

    @staticmethod
    def _load_profile(path: str) -> ExpertParallelProfileResult:
        with open(path, "r") as handle:
            return ExpertParallelProfileResult.from_dict(json.load(handle))

    @classmethod
    def decide_from_profile(
        cls,
        profile: ExpertParallelProfileResult,
        profile_path: str,
        bandwidth_model: AllToAllBandwidthModel,
        ep_candidates: Optional[Sequence[int]] = None,
    ) -> ExpertParallelStrategyReport:
        candidate_sizes = cls._resolve_ep_candidates(profile, ep_candidates)
        evaluations = [
            cls._evaluate_candidate(profile, bandwidth_model, candidate)
            for candidate in candidate_sizes
        ]

        feasible = [candidate for candidate in evaluations if candidate.can_fully_overlap]
        pool = feasible or evaluations
        best = min(
            pool,
            key=lambda item: (
                item.objective_time_s,
                item.exposed_comm_time_s,
                -item.ep_size,
            ),
        )

        decision = ExpertParallelDecision(
            ep_size=best.ep_size,
            local_experts=best.local_experts,
            bandwidth_tier=best.bandwidth_tier,
            can_fully_overlap=best.can_fully_overlap,
            exposed_comm_time_s=best.exposed_comm_time_s,
            objective_time_s=best.objective_time_s,
            comm_payload_bytes_per_phase=best.comm_payload_bytes_per_phase,
            comm_time_s_by_phase=best.comm_time_s_by_phase,
            overlap_window_s_by_phase=best.overlap_window_s_by_phase,
        )
        assumptions = [
            "Attention overlap windows are treated as EP-invariant.",
            "TEGroupedMLP timings are profiled at EP=1 and scaled linearly by local expert count.",
            "EP dispatch/combine communication payload per phase is approximated as micro_batch_size * seqlen / CP * hidden_size * dtype_bytes * router_topk.",
            "Intra-node bandwidth is used when the full EP group fits inside one node after accounting for TP*CP ranks; otherwise inter-node bandwidth is used.",
        ]
        return ExpertParallelStrategyReport(
            profile_path=profile_path,
            profile=profile,
            bandwidth_model=bandwidth_model,
            decision=decision,
            candidate_evaluations=evaluations,
            assumptions=assumptions,
        )

    @classmethod
    def _resolve_ep_candidates(
        cls,
        profile: ExpertParallelProfileResult,
        ep_candidates: Optional[Sequence[int]],
    ) -> list[int]:
        if ep_candidates is not None:
            candidates = sorted(set(int(candidate) for candidate in ep_candidates))
        else:
            candidates = []
            ep_size = 1
            while ep_size <= profile.num_moe_experts:
                candidates.append(ep_size)
                ep_size *= 2

        valid = [
            candidate
            for candidate in candidates
            if candidate > 0 and profile.num_moe_experts % candidate == 0
        ]
        if not valid:
            raise ValueError(
                "No valid EP candidate divides num_moe_experts="
                f"{profile.num_moe_experts}: {candidates}"
            )
        return valid

    @classmethod
    def _evaluate_candidate(
        cls,
        profile: ExpertParallelProfileResult,
        bandwidth_model: AllToAllBandwidthModel,
        ep_size: int,
    ) -> ExpertParallelCandidateEvaluation:
        local_experts = profile.num_moe_experts // ep_size
        bandwidth_tier = cls._select_bandwidth_tier(profile, ep_size)
        comm_payload_bytes = cls._estimate_comm_payload_bytes(profile)
        comm_time = bandwidth_model.estimate_seconds(comm_payload_bytes, bandwidth_tier)
        comm_time_s_by_phase = {phase: comm_time for phase in EP_COMM_PHASES}

        scaled_mlp_forward = profile.mlp_forward_time_s / ep_size
        scaled_mlp_backward = profile.mlp_backward_time_s / ep_size
        scaled_mlp_wgrad = profile.mlp_wgrad_time_s / ep_size
        overlap_window_s_by_phase = {
            "dispatch_forward": profile.attention_wgrad_time_s,
            "combine_backward": scaled_mlp_forward,
            "combine_forward": scaled_mlp_backward,
            "dispatch_backward": scaled_mlp_wgrad,
        }
        overlap_slack_s_by_phase = {
            phase: overlap_window_s_by_phase[phase] - comm_time_s_by_phase[phase]
            for phase in EP_COMM_PHASES
        }
        exposed_comm_time_s = sum(
            max(-slack, 0.0) for slack in overlap_slack_s_by_phase.values()
        )
        objective_time_s = scaled_mlp_forward + scaled_mlp_backward + exposed_comm_time_s
        return ExpertParallelCandidateEvaluation(
            ep_size=ep_size,
            local_experts=local_experts,
            bandwidth_tier=bandwidth_tier,
            comm_payload_bytes_per_phase=comm_payload_bytes,
            comm_time_s_by_phase=comm_time_s_by_phase,
            overlap_window_s_by_phase=overlap_window_s_by_phase,
            overlap_slack_s_by_phase=overlap_slack_s_by_phase,
            scaled_mlp_forward_time_s=scaled_mlp_forward,
            scaled_mlp_backward_time_s=scaled_mlp_backward,
            scaled_mlp_wgrad_time_s=scaled_mlp_wgrad,
            exposed_comm_time_s=exposed_comm_time_s,
            objective_time_s=objective_time_s,
            can_fully_overlap=all(slack >= 0 for slack in overlap_slack_s_by_phase.values()),
        )

    @staticmethod
    def _estimate_comm_payload_bytes(profile: ExpertParallelProfileResult) -> int:
        tokens_per_rank = profile.micro_batch_size * profile.seqlen
        cp_sharded_tokens = max(1, tokens_per_rank // profile.cp_size)
        return (
            cp_sharded_tokens
            * profile.hidden_size
            * profile.dtype_bytes
            * profile.moe_router_topk
        )

    @staticmethod
    def _select_bandwidth_tier(
        profile: ExpertParallelProfileResult, ep_size: int
    ) -> str:
        ranks_per_ep_replica = profile.tp_size * profile.cp_size
        max_ep_within_node = max(1, profile.nprocs_per_node // ranks_per_ep_replica)
        return "intra_node" if ep_size <= max_ep_within_node else "inter_node"

    def _save_report(self, report: ExpertParallelStrategyReport) -> None:
        with open(self.config.decision_path, "w") as handle:
            json.dump(report.to_dict(), handle, indent=2, sort_keys=True)
            handle.write("\n")
