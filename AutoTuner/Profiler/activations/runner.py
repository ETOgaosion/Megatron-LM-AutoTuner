"""Activation profiling runner and step-3 strategy decider."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Optional

PROFILED_PARTS = (
    "attn_norm",
    "core_attn",
    "attn_proj",
    "mlp_norm",
    "expert_fc1",
    "moe_act",
)


@dataclass(frozen=True)
class TransferBandwidthModel:
    """PCIe transfer model used to estimate activation offload time."""

    bandwidth_gbps: float = 64.0
    overhead_seconds: float = 0.0
    round_trip_factor: float = 2.0

    def __post_init__(self) -> None:
        if self.bandwidth_gbps <= 0:
            raise ValueError("bandwidth_gbps must be positive.")
        if self.overhead_seconds < 0:
            raise ValueError("overhead_seconds must be non-negative.")
        if self.round_trip_factor <= 0:
            raise ValueError("round_trip_factor must be positive.")

    def estimate_seconds(self, num_bytes: int) -> float:
        if num_bytes <= 0:
            return 0.0
        gib = num_bytes / float(1024**3)
        transfer_seconds = (gib / self.bandwidth_gbps) * self.round_trip_factor
        return self.overhead_seconds + transfer_seconds

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class ActivationProfilingConfig:
    """Inputs required to profile a single TransformerLayer for step 3."""

    model_name: str
    output_dir: str
    tp_size: int
    cp_size: int
    seqlen: int
    max_token_len: int
    batch_size: int = 128
    micro_batch_size: int = 2
    shape: str = "thd"
    system: str = "megatron"
    warmup_iters: int = 2
    profile_iters: int = 3
    transfer_model: TransferBandwidthModel = field(
        default_factory=TransferBandwidthModel
    )
    profile_script: Optional[str] = None

    def __post_init__(self) -> None:
        if self.tp_size <= 0 or self.cp_size <= 0:
            raise ValueError("tp_size and cp_size must be positive.")
        if self.seqlen <= 0 or self.max_token_len <= 0:
            raise ValueError("seqlen and max_token_len must be positive.")
        if self.batch_size <= 0 or self.micro_batch_size <= 0:
            raise ValueError("batch_size and micro_batch_size must be positive.")
        if self.shape not in {"thd", "bshd"}:
            raise ValueError("shape must be either 'thd' or 'bshd'.")
        if self.warmup_iters < 0 or self.profile_iters <= 0:
            raise ValueError("warmup_iters must be >= 0 and profile_iters must be > 0.")

    @property
    def world_size(self) -> int:
        return self.tp_size * self.cp_size

    @property
    def profile_path(self) -> str:
        return os.path.join(self.output_dir, "activation_profile.json")

    @property
    def decision_path(self) -> str:
        return os.path.join(self.output_dir, "activation_strategy_report.json")


@dataclass(frozen=True)
class ActivationProfileResult:
    """Normalized activation profile emitted by the torchrun profiler."""

    model_name: str
    tp_size: int
    cp_size: int
    seqlen: int
    max_token_len: int
    batch_size: int
    micro_batch_size: int
    shape: str
    system: str
    forward_time_s: float
    backward_time_s: float
    total_time_s: float
    activation_bytes_by_part: dict[str, int]
    activation_present_by_part: dict[str, bool]
    rank_summaries: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.total_time_s <= 0:
            raise ValueError("total_time_s must be positive.")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["activation_bytes_by_part"] = {
            part: int(self.activation_bytes_by_part.get(part, 0))
            for part in PROFILED_PARTS
        }
        payload["activation_present_by_part"] = {
            part: bool(self.activation_present_by_part.get(part, False))
            for part in PROFILED_PARTS
        }
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ActivationProfileResult":
        activation_bytes_by_part = {
            part: int((payload.get("activation_bytes_by_part") or {}).get(part, 0))
            for part in PROFILED_PARTS
        }
        activation_present_by_part = {
            part: bool((payload.get("activation_present_by_part") or {}).get(part, False))
            for part in PROFILED_PARTS
        }
        return cls(
            model_name=str(payload["model_name"]),
            tp_size=int(payload["tp_size"]),
            cp_size=int(payload["cp_size"]),
            seqlen=int(payload["seqlen"]),
            max_token_len=int(payload["max_token_len"]),
            batch_size=int(payload["batch_size"]),
            micro_batch_size=int(payload["micro_batch_size"]),
            shape=str(payload["shape"]),
            system=str(payload["system"]),
            forward_time_s=float(payload["forward_time_s"]),
            backward_time_s=float(payload["backward_time_s"]),
            total_time_s=float(payload["total_time_s"]),
            activation_bytes_by_part=activation_bytes_by_part,
            activation_present_by_part=activation_present_by_part,
            rank_summaries=list(payload.get("rank_summaries") or []),
        )


@dataclass(frozen=True)
class ActivationOffloadDecision:
    """Recommended activation offload policy."""

    modules: list[str]
    module_times_s: dict[str, float]
    total_time_s: float
    covered_parts: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ActivationRecomputeDecision:
    """Recommended recompute policy after offload is chosen."""

    kind: str
    modules: list[str]
    covered_parts: list[str]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ActivationStrategyReport:
    """Full step-3 output combining the profile and the final strategy."""

    profile_path: str
    profile: ActivationProfileResult
    transfer_model: TransferBandwidthModel
    offload: ActivationOffloadDecision
    recompute: ActivationRecomputeDecision
    assumptions: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_path": self.profile_path,
            "profile": self.profile.to_dict(),
            "transfer_model": self.transfer_model.to_dict(),
            "offload": self.offload.to_dict(),
            "recompute": self.recompute.to_dict(),
            "assumptions": self.assumptions,
        }


@dataclass(frozen=True)
class _OffloadGroup:
    name: str
    parts: tuple[str, ...]
    recompute_module: str


_OFFLOAD_GROUPS = (
    _OffloadGroup(
        name="attention_core",
        parts=("core_attn", "attn_proj"),
        recompute_module="core_attn",
    ),
    _OffloadGroup(
        name="layernorm",
        parts=("attn_norm", "mlp_norm"),
        recompute_module="layernorm",
    ),
    _OffloadGroup(
        name="moe",
        parts=("expert_fc1", "moe_act"),
        recompute_module="moe",
    ),
)


class ActivationStrategyDecider:
    """Profiles a TransformerLayer and decides offload/recompute policies."""

    def __init__(self, config: ActivationProfilingConfig):
        self.config = config

    def run(self, skip_profiling: bool = False) -> ActivationStrategyReport:
        os.makedirs(self.config.output_dir, exist_ok=True)

        if skip_profiling:
            if not os.path.exists(self.config.profile_path):
                raise FileNotFoundError(
                    f"Activation profile not found at {self.config.profile_path}. "
                    "Disable skip_profiling or provide an existing profile."
                )
        else:
            self._run_profiling()

        profile = self._load_profile(self.config.profile_path)
        report = self.decide_from_profile(
            profile=profile,
            profile_path=self.config.profile_path,
            transfer_model=self.config.transfer_model,
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
            "AutoTuner.Profiler.activations.profile_transformer_layer",
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
            raise RuntimeError(f"Activation profiling failed: {error}") from exc

    @staticmethod
    def _run_profile_script(script_path: str) -> None:
        try:
            subprocess.run(
                ["bash", script_path], check=True, capture_output=True, text=True
            )
        except subprocess.CalledProcessError as exc:  # pragma: no cover - env dependent
            error = exc.stderr or exc.stdout or str(exc)
            raise RuntimeError(f"Activation profiling script failed: {error}") from exc

    @staticmethod
    def _load_profile(path: str) -> ActivationProfileResult:
        with open(path, "r") as handle:
            return ActivationProfileResult.from_dict(json.load(handle))

    @classmethod
    def decide_from_profile(
        cls,
        profile: ActivationProfileResult,
        profile_path: str,
        transfer_model: TransferBandwidthModel,
    ) -> ActivationStrategyReport:
        part_times = {
            part: transfer_model.estimate_seconds(profile.activation_bytes_by_part.get(part, 0))
            for part in PROFILED_PARTS
        }

        groups = cls._build_groups(profile, transfer_model)
        selected_groups = []
        selected_group_names = set()
        total_offload_time = 0.0
        for group in sorted(groups, key=lambda item: item["total_bytes"], reverse=True):
            if group["recompute_module"] == "full":
                continue
            candidate_time = total_offload_time + group["total_time_s"]
            if candidate_time <= profile.total_time_s:
                selected_groups.append(group)
                selected_group_names.add(group["name"])
                total_offload_time = candidate_time

        offload_modules = [
            part for group in selected_groups for part in group["present_parts"]
        ]
        offload = ActivationOffloadDecision(
            modules=offload_modules,
            module_times_s={part: part_times[part] for part in offload_modules},
            total_time_s=total_offload_time,
            covered_parts=offload_modules,
        )

        leftover_groups = [
            group for group in groups if group["name"] not in selected_group_names
        ]
        requires_full_recompute = any(
            group["recompute_module"] == "full" for group in leftover_groups
        )
        if requires_full_recompute:
            recompute = ActivationRecomputeDecision(
                kind="full",
                modules=[],
                covered_parts=[
                    part for group in groups for part in group["present_parts"]
                ],
                reason=(
                    "At least one remaining activation part does not map cleanly to a "
                    "selective recompute module, so the fallback is full recompute."
                ),
            )
            offload = ActivationOffloadDecision(
                modules=[],
                module_times_s={},
                total_time_s=0.0,
                covered_parts=[],
            )
        elif leftover_groups:
            recompute_modules = cls._dedupe_preserve_order(
                group["recompute_module"] for group in leftover_groups
            )
            recompute_parts = [
                part for group in leftover_groups for part in group["present_parts"]
            ]
            recompute = ActivationRecomputeDecision(
                kind="selective",
                modules=recompute_modules,
                covered_parts=recompute_parts,
                reason=(
                    "Selective recompute is used for remaining activation groups that do "
                    "not fit in the offload time budget."
                ),
            )
        else:
            recompute = ActivationRecomputeDecision(
                kind="none",
                modules=[],
                covered_parts=[],
                reason="All profiled activation groups fit within the offload time budget.",
            )

        assumptions = [
            "Activation offload time is estimated as a PCIe round trip based on the profiled activation size.",
            "The offload time budget is compared against the measured TransformerLayer forward+backward time.",
            "Offload decisions are made on recompute-coherent groups: "
            "[core_attn, attn_proj], [attn_norm, mlp_norm], and [expert_fc1, moe_act].",
        ]
        return ActivationStrategyReport(
            profile_path=profile_path,
            profile=profile,
            transfer_model=transfer_model,
            offload=offload,
            recompute=recompute,
            assumptions=assumptions,
        )

    @classmethod
    def _build_groups(
        cls,
        profile: ActivationProfileResult,
        transfer_model: TransferBandwidthModel,
    ) -> list[dict[str, Any]]:
        groups: list[dict[str, Any]] = []
        seen_parts: set[str] = set()
        for group in _OFFLOAD_GROUPS:
            present_parts = [
                part
                for part in group.parts
                if profile.activation_bytes_by_part.get(part, 0) > 0
            ]
            if not present_parts:
                continue
            seen_parts.update(present_parts)
            recompute_module = group.recompute_module
            if group.name == "moe" and present_parts == ["moe_act"]:
                recompute_module = "moe_act"
            groups.append(
                {
                    "name": group.name,
                    "present_parts": present_parts,
                    "total_bytes": sum(
                        profile.activation_bytes_by_part[part] for part in present_parts
                    ),
                    "total_time_s": sum(
                        transfer_model.estimate_seconds(
                            profile.activation_bytes_by_part[part]
                        )
                        for part in present_parts
                    ),
                    "recompute_module": recompute_module,
                }
            )

        unsupported_parts = [
            part
            for part in PROFILED_PARTS
            if profile.activation_bytes_by_part.get(part, 0) > 0 and part not in seen_parts
        ]
        if unsupported_parts:
            groups.append(
                {
                    "name": "fallback_full_recompute",
                    "present_parts": unsupported_parts,
                    "total_bytes": sum(
                        profile.activation_bytes_by_part[part] for part in unsupported_parts
                    ),
                    "total_time_s": 0.0,
                    "recompute_module": "full",
                }
            )
        return groups

    @staticmethod
    def _dedupe_preserve_order(values: Iterable[str]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            ordered.append(value)
        return ordered

    def _save_report(self, report: ActivationStrategyReport) -> None:
        with open(self.config.decision_path, "w") as handle:
            json.dump(report.to_dict(), handle, indent=2, sort_keys=True)
            handle.write("\n")

        summary_path = os.path.join(self.config.output_dir, "summary.txt")
        lines = [
            "ACTIVATION STRATEGY REPORT",
            f"Model: {report.profile.model_name}",
            (
                "TP/CP/Seqlen/max_token_len: "
                f"{report.profile.tp_size}/{report.profile.cp_size}/"
                f"{report.profile.seqlen}/{report.profile.max_token_len}"
            ),
            f"TransformerLayer compute time: {report.profile.total_time_s:.6f}s",
            "",
            "OFFLOAD",
            (
                "Modules: "
                + (", ".join(report.offload.modules) if report.offload.modules else "none")
            ),
            f"Estimated total offload time: {report.offload.total_time_s:.6f}s",
            "",
            "RECOMPUTE",
            f"Kind: {report.recompute.kind}",
            (
                "Modules: "
                + (
                    ", ".join(report.recompute.modules)
                    if report.recompute.modules
                    else "none"
                )
            ),
            f"Reason: {report.recompute.reason}",
        ]
        with open(summary_path, "w") as handle:
            handle.write("\n".join(lines))
            handle.write("\n")
