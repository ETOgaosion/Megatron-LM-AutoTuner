"""CP overlap profiling runner."""

from __future__ import annotations

import glob
import json
import os
import subprocess
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

from .cp_overlap_trace_analyzer import analyze_trace


@dataclass(frozen=True)
class CPOverlapInputCase:
    seqlen: int
    max_token_len: int
    batch_size: int = 128
    micro_batch_size: int = 2
    shape: str = "thd"
    system: str = "megatron"

    def get_case_id(self) -> str:
        return (
            f"seq{self.seqlen}_tok{self.max_token_len}_"
            f"bs{self.batch_size}_mbs{self.micro_batch_size}_{self.shape}"
        )

    def to_test_case_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class CPOverlapRunnerConfig:
    model_name: str
    cases: List[CPOverlapInputCase]
    output_dir: str
    cp_size: int = 2
    profile_script: str = "scripts/cp_overlap_tuner/profile_single_case.sh"
    seed: int = 0
    range_index: Optional[int] = None
    forward_pattern: str = (
        "TransformerEngine-Enhanced/transformer_engine/pytorch/attention/"
        "dot_product_attention/context_parallel_nvshmem.py(479): forward"
    )


@dataclass
class ProfilingResult:
    case: CPOverlapInputCase
    trace_path: Optional[str] = None
    success: bool = False
    error_message: str = ""
    analysis: Optional[Dict[str, object]] = None


class CPOverlapRunner:
    """Run CP overlap profiling and trace analysis."""

    def __init__(self, config: CPOverlapRunnerConfig):
        self.config = config
        os.makedirs(self.config.output_dir, exist_ok=True)

    def _write_test_cases_files(self) -> Dict[str, str]:
        test_cases_dir = os.path.join(self.config.output_dir, "test_cases")
        os.makedirs(test_cases_dir, exist_ok=True)

        manifest = {
            "model": self.config.model_name,
            "cp_size": self.config.cp_size,
            "cases": [case.to_test_case_dict() for case in self.config.cases],
        }
        manifest_path = os.path.join(test_cases_dir, "cases_manifest.json")
        with open(manifest_path, "w") as handle:
            json.dump(manifest, handle, indent=2)

        case_files: Dict[str, str] = {}
        for case in self.config.cases:
            case_path = os.path.join(test_cases_dir, f"{case.get_case_id()}.json")
            with open(case_path, "w") as handle:
                json.dump(
                    {"model": self.config.model_name, "cases": [case.to_test_case_dict()]},
                    handle,
                    indent=2,
                )
            case_files[case.get_case_id()] = case_path
        return case_files

    def run(self, skip_profiling: bool = False) -> List[ProfilingResult]:
        case_files = self._write_test_cases_files()
        if skip_profiling:
            results = self._load_existing_traces()
        else:
            results = self._run_profiling(case_files)
        self._analyze_results(results)
        self._save_report(results)
        return results

    def _run_profiling(self, case_files: Dict[str, str]) -> List[ProfilingResult]:
        results: List[ProfilingResult] = []
        for case in self.config.cases:
            results.append(self._profile_single_case(case, case_files[case.get_case_id()]))
        return results

    def _profile_single_case(
        self, case: CPOverlapInputCase, case_file: str
    ) -> ProfilingResult:
        result = ProfilingResult(case=case)
        case_output_dir = os.path.join(
            self.config.output_dir, "traces", case.get_case_id()
        )
        os.makedirs(case_output_dir, exist_ok=True)

        env = os.environ.copy()
        env.update(
            {
                "MODEL_NAME": self.config.model_name,
                "CP_SIZE": str(self.config.cp_size),
                "OUTPUT_DIR": case_output_dir,
                "TEST_CASES_DIR": os.path.dirname(case_file),
                "TEST_CASES_FILE": os.path.basename(case_file),
            }
        )

        try:
            if os.path.exists(self.config.profile_script):
                subprocess.run(
                    ["bash", self.config.profile_script],
                    env=env,
                    check=True,
                    capture_output=True,
                    text=True,
                )
            else:
                self._run_python_profiling(case_file, case_output_dir)

            trace_files = glob.glob(
                os.path.join(case_output_dir, "**/*.pt.trace.json"), recursive=True
            )
            if trace_files:
                result.trace_path = trace_files[0]
                result.success = True
            else:
                result.error_message = "No trace file generated."
        except subprocess.CalledProcessError as exc:
            result.error_message = exc.stderr or exc.stdout or str(exc)
        except Exception as exc:  # pragma: no cover - defensive error plumbing
            result.error_message = str(exc)
        return result

    def _run_python_profiling(self, case_file: str, case_output_dir: str) -> None:
        cmd = [
            "torchrun",
            f"--nproc_per_node={self.config.cp_size}",
            "-m",
            "AutoTuner.testbench.profile.main",
            "--model-name",
            self.config.model_name,
            "--test-cases-dir",
            os.path.dirname(case_file),
            "--test-cases-file",
            os.path.basename(case_file),
            "--profile-mode",
            "2",
            "--test-ops-list",
            "TEAttenWithCPEnhanced",
            "--run-one-data",
            "--tensor-model-parallel-size",
            "1",
            "--context-parallel-size",
            str(self.config.cp_size),
            "--output-dir",
            case_output_dir,
        ]

        env = os.environ.copy()
        env["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        env["NVTE_FLASH_ATTN"] = "1"
        env["NVTE_FUSED_ATTN"] = "0"
        env["NVTE_NVTX_ENABLED"] = "1"
        subprocess.run(cmd, env=env, check=True, capture_output=True, text=True)

    def _load_existing_traces(self) -> List[ProfilingResult]:
        results: List[ProfilingResult] = []
        for case in self.config.cases:
            result = ProfilingResult(case=case)
            case_output_dir = os.path.join(
                self.config.output_dir, "traces", case.get_case_id()
            )
            trace_files = glob.glob(
                os.path.join(case_output_dir, "**/*.pt.trace.json"), recursive=True
            )
            if trace_files:
                result.trace_path = trace_files[0]
                result.success = True
            else:
                result.error_message = "No existing trace file found."
            results.append(result)
        return results

    def _analyze_results(self, results: List[ProfilingResult]) -> None:
        for result in results:
            if not result.success or not result.trace_path:
                continue
            analysis = analyze_trace(
                trace_path=result.trace_path,
                seed=self.config.seed,
                range_index=self.config.range_index,
                forward_pattern=self.config.forward_pattern,
            )
            analysis["case_id"] = result.case.get_case_id()
            analysis["case"] = result.case.to_test_case_dict()
            result.analysis = analysis

            analysis_path = os.path.join(
                self.config.output_dir,
                "traces",
                result.case.get_case_id(),
                "cp_overlap_analysis.json",
            )
            with open(analysis_path, "w") as handle:
                json.dump(analysis, handle, indent=2, sort_keys=True)
                handle.write("\n")

    def _save_report(self, results: List[ProfilingResult]) -> None:
        report = {
            "model_name": self.config.model_name,
            "cp_size": self.config.cp_size,
            "range_index": self.config.range_index,
            "seed": self.config.seed,
            "forward_pattern": self.config.forward_pattern,
            "results": [
                {
                    "case_id": result.case.get_case_id(),
                    "case": result.case.to_test_case_dict(),
                    "success": result.success,
                    "trace_path": result.trace_path,
                    "error_message": result.error_message,
                    "analysis": result.analysis,
                }
                for result in results
            ],
        }
        report_path = os.path.join(self.config.output_dir, "cp_overlap_report.json")
        with open(report_path, "w") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")

        summary_lines = [
            "CP OVERLAP SUMMARY",
            f"Model: {self.config.model_name}",
            f"CP Size: {self.config.cp_size}",
            "",
        ]
        for result in results:
            prefix = f"{result.case.get_case_id()}:"
            if not result.success:
                summary_lines.append(f"{prefix} ERROR: {result.error_message}")
                continue
            if not result.analysis:
                summary_lines.append(f"{prefix} ERROR: analysis missing")
                continue
            summary_lines.append(
                (
                    f"{prefix} overlap_ratio={result.analysis['overlap_ratio']:.4f}, "
                    f"compute_time_us={result.analysis['compute_time_us']:.2f}, "
                    f"comm_time_us={result.analysis['comm_time_us']:.2f}, "
                    f"overlapped_comm_time_us={result.analysis['overlapped_comm_time_us']:.2f}, "
                    f"compute_streams={result.analysis['compute_stream_tids']}, "
                    f"comm_streams={result.analysis['comm_stream_tids']}"
                )
            )

        summary_path = os.path.join(self.config.output_dir, "summary.txt")
        with open(summary_path, "w") as handle:
            handle.write("\n".join(summary_lines))
            handle.write("\n")
