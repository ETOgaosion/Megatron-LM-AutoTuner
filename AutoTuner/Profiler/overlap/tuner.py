"""
TP Overlap Tuner - Main Orchestrator.

This module provides the main TPOverlapTuner class that orchestrates
the entire tuning workflow:
1. Generate test configurations
2. Run profiling for each configuration
3. Analyze traces to detect overlap
4. Generate reports with optimal configurations
"""

import glob
import json
import os
import subprocess
from dataclasses import dataclass
from typing import List, Optional

from .config_generator import (
    TPOverlapConfigGenerator,
    TPOverlapTestConfig,
    TPOverlapTunerConfig,
    generate_single_test_yaml,
)
from .overlap_detector import OverlapAnalysis, OverlapDetector
from .report_generator import ReportGenerator, TuningReport


@dataclass
class ProfilingResult:
    """Result of a single profiling run."""

    config: TPOverlapTestConfig
    trace_path: Optional[str] = None
    success: bool = False
    error_message: str = ""


class TPOverlapTuner:
    """Main orchestrator for TP overlap tuning."""

    def __init__(
        self,
        tuner_config: TPOverlapTunerConfig,
        profile_script: str = "scripts/tp_overlap_tuner/profile_single_config.sh",
        overlap_threshold: float = 0.5,
    ):
        """Initialize the TP Overlap Tuner.

        Args:
            tuner_config: Configuration for the tuner.
            profile_script: Path to the profiling script.
            overlap_threshold: Minimum overlap ratio to consider effective.
        """
        self.config = tuner_config
        self.profile_script = profile_script
        self.overlap_threshold = overlap_threshold

        # Initialize components
        self.config_generator = TPOverlapConfigGenerator(tuner_config)
        self.overlap_detector = OverlapDetector()
        self.report_generator = ReportGenerator(overlap_threshold=overlap_threshold)

        # Create output directory
        os.makedirs(tuner_config.output_dir, exist_ok=True)

        # Track profiling results
        self.profiling_results: List[ProfilingResult] = []

    def _generate_test_cases_file(self) -> str:
        """Generate test cases JSON file for profiling.

        Returns:
            Path to the generated test cases JSON file.
        """
        # Create test cases directory
        test_cases_dir = os.path.join(self.config.output_dir, "test_cases")
        os.makedirs(test_cases_dir, exist_ok=True)

        # Generate test case JSON
        test_cases_data = {
            "model": self.config.model_name,
            "cases": [
                {
                    "batch_size": 128,
                    "micro_batch_size": 2,
                    "seqlen": 1024,
                    "max_token_len": self.config.max_token_len,
                    "shape": "bshd",
                    "system": "megatron",
                }
            ],
        }

        # Save to file
        test_cases_file = os.path.join(test_cases_dir, "test_cases.json")
        with open(test_cases_file, "w") as f:
            json.dump(test_cases_data, f, indent=2)

        return test_cases_file

    def run(self, skip_profiling: bool = False) -> TuningReport:
        """Run the complete tuning workflow.

        Args:
            skip_profiling: If True, skip profiling and only analyze existing traces.

        Returns:
            TuningReport with analysis results and recommendations.
        """
        print("=" * 60)
        print("TP OVERLAP TUNER")
        print("=" * 60)
        print(f"Model: {self.config.model_name}")
        print(f"Output Directory: {self.config.output_dir}")
        print(f"Operators: {', '.join(self.config.operators)}")
        print(f"Max TP Size: {self.config.max_tp_size}")
        print("")

        # Validate required config files exist
        config_dir = "AutoTuner/testbench/profile/configs/local"
        required_files = ["override_model_config.json", "override_tf_config.json"]

        missing = [
            f for f in required_files if not os.path.exists(os.path.join(config_dir, f))
        ]
        if missing:
            raise FileNotFoundError(
                f"Required config files missing in {config_dir}: {missing}\n"
                f"Copy sample files from AutoTuner/testbench/profile/configs/\n"
                f"See README at {config_dir}/README.md for details"
            )

        # Step 0: Generate test cases file
        print("Step 0: Generating test cases file...")
        self.test_cases_file = self._generate_test_cases_file()
        print(f"  Generated test cases file: {self.test_cases_file}")

        # Step 1: Generate test configurations
        print("\nStep 1: Generating test configurations...")
        test_configs = self.config_generator.generate_all_configs()
        print(f"  Generated {len(test_configs)} test configurations")

        # Step 2: Run profiling for each configuration
        if not skip_profiling:
            print("\nStep 2: Running profiling...")
            self.profiling_results = self._run_profiling(test_configs)
            successful = sum(1 for r in self.profiling_results if r.success)
            print(
                f"  Profiling completed: {successful}/{len(self.profiling_results)} successful"
            )
        else:
            print("\nStep 2: Skipping profiling (using existing traces)")
            self.profiling_results = self._load_existing_traces(test_configs)

        # Step 3: Analyze traces
        print("\nStep 3: Analyzing traces...")
        trace_configs = [
            (r.trace_path, r.config)
            for r in self.profiling_results
            if r.success and r.trace_path
        ]
        analyses = self.overlap_detector.analyze_multiple_traces(trace_configs)
        print(f"  Analyzed {len(analyses)} traces")

        # Step 4: Generate report
        print("\nStep 4: Generating report...")
        report = self.report_generator.generate(analyses, self.config)
        self.report_generator.save_report(report, self.config.output_dir)
        print(f"  Report saved to {self.config.output_dir}")

        # Print summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        for rec in report.recommendations[:10]:  # Show first 10 recommendations
            print(f"  * {rec}")
        if len(report.recommendations) > 10:
            print(f"  ... and {len(report.recommendations) - 10} more recommendations")

        print(f"\nFull report saved to: {self.config.output_dir}")
        return report

    def run_single_config(
        self, config: TPOverlapTestConfig
    ) -> Optional[OverlapAnalysis]:
        """Run profiling and analysis for a single configuration.

        Args:
            config: The test configuration to run.

        Returns:
            OverlapAnalysis if successful, None otherwise.
        """
        result = self._profile_single_config(config)
        if not result.success or not result.trace_path:
            return None

        return self.overlap_detector.analyze_overlap(result.trace_path, config)

    def _run_profiling(
        self, configs: List[TPOverlapTestConfig]
    ) -> List[ProfilingResult]:
        """Run profiling for all configurations.

        Args:
            configs: List of test configurations.

        Returns:
            List of ProfilingResult objects.
        """
        results = []
        total = len(configs)

        for i, config in enumerate(configs):
            print(f"  [{i + 1}/{total}] Profiling {config.get_test_id()}...")
            result = self._profile_single_config(config)
            results.append(result)

            if result.success:
                print(f"    Success: {result.trace_path}")
            else:
                print(f"    ERROR: {result.error_message}")
                print(f"\n{'=' * 60}")
                print("STOPPING: Test case failed!")
                print(f"{'=' * 60}")
                print(f"Failed test: {config.get_test_id()}")
                print(f"Error: {result.error_message}")
                raise RuntimeError(
                    f"Profiling failed for {config.get_test_id()}: {result.error_message}"
                )

        return results

    def _profile_single_config(self, config: TPOverlapTestConfig) -> ProfilingResult:
        """Run profiling for a single configuration.

        Args:
            config: The test configuration.

        Returns:
            ProfilingResult with trace path or error message.
        """
        result = ProfilingResult(config=config)

        # Create output directory for this config
        config_output_dir = os.path.join(
            self.config.output_dir, "traces", config.get_test_id()
        )
        os.makedirs(config_output_dir, exist_ok=True)

        # Generate YAML config file
        yaml_path = os.path.join(config_output_dir, "tp_comm_overlap_cfg.yaml")
        generate_single_test_yaml(config, yaml_path)

        # Determine test class based on operator
        test_class = self._get_test_class_for_operator(config.operator)

        try:
            # Run profiling script
            env = os.environ.copy()
            env.update(
                {
                    "TP_SIZE": str(config.tp_size),
                    "OPERATOR": config.operator,
                    "TEST_CLASS": test_class,
                    "OUTPUT_DIR": config_output_dir,
                    "YAML_CONFIG": yaml_path,
                    "MODEL_NAME": self.config.model_name,
                    "TEST_CASES_DIR": os.path.dirname(self.test_cases_file),
                    "TEST_CASES_FILE": os.path.basename(self.test_cases_file),
                }
            )

            # Check if profile script exists
            if os.path.exists(self.profile_script):
                subprocess.run(
                    ["bash", self.profile_script],
                    env=env,
                    check=True,
                    capture_output=True,
                    text=True,
                )
            else:
                # Use Python-based profiling
                result = self._run_python_profiling(
                    config, config_output_dir, yaml_path
                )
                return result

            # Find the generated trace file
            trace_files = glob.glob(
                os.path.join(config_output_dir, "**/*.pt.trace.json"), recursive=True
            )
            if trace_files:
                result.trace_path = trace_files[0]
                result.success = True
            else:
                result.error_message = "No trace file generated"

        except subprocess.CalledProcessError as e:
            result.error_message = f"Profiling script failed: {e.stderr}"
        except Exception as e:
            result.error_message = str(e)

        return result

    def _run_python_profiling(
        self, config: TPOverlapTestConfig, output_dir: str, yaml_path: str
    ) -> ProfilingResult:
        """Run profiling using Python subprocess.

        This method launches the profiling directly using torchrun.
        """
        result = ProfilingResult(config=config)

        test_class = self._get_test_class_for_operator(config.operator)

        cmd = [
            "torchrun",
            f"--nproc_per_node={config.tp_size}",
            "-m",
            "AutoTuner.testbench.profile.main",
            "--model-name",
            self.config.model_name,
            "--test-cases-dir",
            os.path.dirname(self.test_cases_file),
            "--test-cases-file",
            os.path.basename(self.test_cases_file),
            "--profile-mode",
            "2",  # torch profiler mode
            "--test-ops-list",
            test_class,
            "--tp-comm-buffer-name",
            config.operator,
            "--run-one-data",
            "--tensor-model-parallel-size",
            str(config.tp_size),
            "--output-dir",
            output_dir,
        ]

        env = os.environ.copy()
        env["UB_SKIPMC"] = "1"
        env["NVTE_FLASH_ATTN"] = "1"
        env["NVTE_FUSED_ATTN"] = "0"

        # Copy YAML config to expected location
        config_dir = "AutoTuner/testbench/profile/configs/local"
        os.makedirs(config_dir, exist_ok=True)
        target_yaml = os.path.join(config_dir, "tp_comm_overlap_cfg.yaml")

        import shutil

        shutil.copy(yaml_path, target_yaml)

        try:
            subprocess.run(cmd, env=env, check=True, capture_output=True, text=True)

            # Find trace file
            trace_files = glob.glob(
                os.path.join(output_dir, "**/*.pt.trace.json"), recursive=True
            )
            if trace_files:
                result.trace_path = trace_files[0]
                result.success = True
            else:
                result.error_message = "No trace file generated"

        except subprocess.CalledProcessError as e:
            result.error_message = f"torchrun failed: {e.stderr}"
        except Exception as e:
            result.error_message = str(e)

        return result

    def _load_existing_traces(
        self, configs: List[TPOverlapTestConfig]
    ) -> List[ProfilingResult]:
        """Load existing trace files for configurations.

        Looks for trace files that match the configuration IDs.
        """
        results = []

        traces_dir = os.path.join(self.config.output_dir, "traces")
        if not os.path.exists(traces_dir):
            return results

        for config in configs:
            result = ProfilingResult(config=config)
            config_dir = os.path.join(traces_dir, config.get_test_id())

            if os.path.exists(config_dir):
                trace_files = glob.glob(
                    os.path.join(config_dir, "**/*.pt.trace.json"), recursive=True
                )
                if trace_files:
                    result.trace_path = trace_files[0]
                    result.success = True

            results.append(result)

        return results

    def _get_test_class_for_operator(self, operator: str) -> str:
        """Get the test class name for an operator."""
        mapping = {
            "fc1": "TEColumnParallelLinear",
            "fc2": "TERowParallelLinear",
            "qkv": "TEColumnParallelLinear",
            "proj": "TERowParallelLinear",
        }
        return mapping.get(operator, "TEColumnParallelLinear")
