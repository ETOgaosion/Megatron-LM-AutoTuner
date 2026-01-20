"""
TP Overlap Config Generator.

This module generates test configurations for TP communication/computation
overlap tuning using binary search strategy for num_sm parameter.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import yaml


class OverlapMethod(Enum):
    """Overlap methods for TP communication.

    RING_EXCHANGE: Uses ring-based all-gather/reduce-scatter. Recommended for
                   forward pass (fprop) and some backward passes (dgrad).
    BULK: Uses bulk collective operations with configurable SM count.
          Recommended for backward passes (dgrad, wgrad) when tuning SM usage.
    PIPELINE: Defined for completeness but NOT currently used. Per requirements,
              "reduce-scatter has no meaning for Pipeline method" in TP overlap.
              Kept for potential future use or alternative overlap strategies.
    """

    RING_EXCHANGE = "ring_exchange"
    BULK = "bulk"
    PIPELINE = "pipeline"  # Not used - see docstring


class LinearType(Enum):
    """Types of parallel linear layers."""

    COLUMN = "column"  # ColumnParallelLinear (fc1, qkv)
    ROW = "row"  # RowParallelLinear (fc2, proj)


class Phase(Enum):
    """Forward and backward phases."""

    FPROP = "fprop"
    DGRAD = "dgrad"
    WGRAD = "wgrad"


# Mapping of operators to their linear types
OPERATOR_LINEAR_TYPE = {
    "fc1": LinearType.COLUMN,
    "fc2": LinearType.ROW,
    "qkv": LinearType.COLUMN,
    "proj": LinearType.ROW,
}

# Default overlap configurations for each operator and phase
# Based on TransformerEngine's recommended configurations
DEFAULT_CONFIGS = {
    "qkv_fprop": {"method": OverlapMethod.RING_EXCHANGE, "aggregate": 1},
    "proj_fprop": {"method": OverlapMethod.RING_EXCHANGE, "aggregate": 1},
    "fc1_fprop": {"method": OverlapMethod.RING_EXCHANGE, "aggregate": 0},
    "fc2_fprop": {"method": OverlapMethod.RING_EXCHANGE, "aggregate": 0},
    "fc2_dgrad": {"method": OverlapMethod.RING_EXCHANGE, "aggregate": 0},
    "fc1_dgrad": {"method": OverlapMethod.BULK, "num_sm": 2, "set_sm_margin": 0},
    "fc1_wgrad": {"method": OverlapMethod.BULK, "num_sm": 2, "set_sm_margin": 0},
    "proj_dgrad": {"method": OverlapMethod.RING_EXCHANGE, "aggregate": 1},
    "qkv_dgrad": {"method": OverlapMethod.BULK, "num_sm": 2, "set_sm_margin": 0},
    "qkv_wgrad": {"method": OverlapMethod.BULK, "num_sm": 2, "set_sm_margin": 0},
}


@dataclass
class TPOverlapTestConfig:
    """Configuration for a single TP overlap test."""

    tp_size: int
    operator: str  # "fc1", "fc2", "qkv", "proj"
    phase: str  # "fprop", "dgrad", "wgrad"
    overlap_method: OverlapMethod
    aggregate: int = 0  # for ring_exchange (0 or 1)
    num_sm: int = 2  # for bulk method
    set_sm_margin: int = 0  # for bulk method
    num_splits: int = 4  # for pipeline method
    is_baseline: bool = False  # baseline without overlap

    @property
    def linear_type(self) -> LinearType:
        """Get the linear type for this operator."""
        return OPERATOR_LINEAR_TYPE.get(self.operator, LinearType.COLUMN)

    @property
    def config_key(self) -> str:
        """Get the config key for YAML (e.g., 'fc1_fprop')."""
        return f"{self.operator}_{self.phase}"

    def to_yaml_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for YAML output."""
        result = {"method": self.overlap_method.value}

        if self.overlap_method == OverlapMethod.RING_EXCHANGE:
            result["aggregate"] = self.aggregate
        elif self.overlap_method == OverlapMethod.BULK:
            result["num_sm"] = self.num_sm
            result["set_sm_margin"] = self.set_sm_margin
        elif self.overlap_method == OverlapMethod.PIPELINE:
            result["num_sm"] = self.num_sm
            result["num_splits"] = self.num_splits
            result["set_sm_margin"] = self.set_sm_margin

        return result

    def get_test_id(self) -> str:
        """Get a unique identifier for this test configuration."""
        if self.is_baseline:
            return f"tp{self.tp_size}_{self.operator}_{self.phase}_baseline"

        if self.overlap_method == OverlapMethod.RING_EXCHANGE:
            return f"tp{self.tp_size}_{self.operator}_{self.phase}_ring_agg{self.aggregate}"
        elif self.overlap_method == OverlapMethod.BULK:
            return f"tp{self.tp_size}_{self.operator}_{self.phase}_bulk_sm{self.num_sm}"
        elif self.overlap_method == OverlapMethod.PIPELINE:
            return f"tp{self.tp_size}_{self.operator}_{self.phase}_pipe_sm{self.num_sm}_split{self.num_splits}"
        return f"tp{self.tp_size}_{self.operator}_{self.phase}_unknown"


@dataclass
class TPOverlapTunerConfig:
    """Configuration for the TP Overlap Tuner.

    Model parameters (hidden_size, ffn_hidden_size, num_attention_heads, num_kv_heads)
    are automatically fetched from the model name using HuggingFace config.
    """

    model_name: str
    max_tp_size: int = 8
    max_token_len: int = 8192
    operators: List[str] = field(default_factory=lambda: ["fc1", "fc2", "qkv", "proj"])
    output_dir: str = "outputs/tp_overlap_tuner"
    # Binary search parameters
    min_num_sm: int = 1
    max_num_sm: int = 16
    # Ring exchange aggregate values to test
    aggregate_values: List[int] = field(default_factory=lambda: [0, 1])
    # Model parameters (auto-populated from model_name)
    hidden_size: int = field(default=0, init=False)
    ffn_hidden_size: int = field(default=0, init=False)
    num_attention_heads: int = field(default=0, init=False)
    num_kv_heads: int = field(default=0, init=False)

    def __post_init__(self):
        """Fetch model parameters from HuggingFace config."""
        from AutoTuner.utils.config import get_hf_model_config

        hf_config = get_hf_model_config(self.model_name)
        self.hidden_size = hf_config.hidden_size
        self.ffn_hidden_size = getattr(
            hf_config, "intermediate_size", hf_config.hidden_size * 4
        )
        self.num_attention_heads = hf_config.num_attention_heads
        self.num_kv_heads = getattr(
            hf_config, "num_key_value_heads", hf_config.num_attention_heads
        )


class TPOverlapConfigGenerator:
    """Generates test configurations for TP overlap tuning."""

    def __init__(self, tuner_config: TPOverlapTunerConfig):
        """Initialize the config generator."""
        self.config = tuner_config
        # TP sizes to test: 1 (no TP, baseline), 2, 4, 8
        # TP=1 is used as baseline to compare if TP is beneficial at all
        self.tp_sizes = [1, 2, 4, 8]
        # Filter to max_tp_size
        self.tp_sizes = [s for s in self.tp_sizes if s <= tuner_config.max_tp_size]

    def generate_baseline_configs(
        self, tp_size: int, operator: str
    ) -> List[TPOverlapTestConfig]:
        """Generate baseline configurations without overlap."""
        configs = []
        for phase in ["fprop", "dgrad", "wgrad"]:
            # Skip wgrad for fc2 and proj (RowParallelLinear forward doesn't have wgrad overlap)
            config = TPOverlapTestConfig(
                tp_size=tp_size,
                operator=operator,
                phase=phase,
                overlap_method=OverlapMethod.RING_EXCHANGE,
                aggregate=0,
                is_baseline=True,
            )
            configs.append(config)
        return configs

    def generate_ring_exchange_configs(
        self, tp_size: int, operator: str
    ) -> List[TPOverlapTestConfig]:
        """Generate ring_exchange configurations."""
        configs = []
        phases = ["fprop", "dgrad"]  # ring_exchange typically used for fprop and dgrad

        for phase in phases:
            for aggregate in self.config.aggregate_values:
                config = TPOverlapTestConfig(
                    tp_size=tp_size,
                    operator=operator,
                    phase=phase,
                    overlap_method=OverlapMethod.RING_EXCHANGE,
                    aggregate=aggregate,
                )
                configs.append(config)
        return configs

    def generate_bulk_configs(
        self, tp_size: int, operator: str
    ) -> List[TPOverlapTestConfig]:
        """Generate bulk configurations with different num_sm values."""
        configs = []
        phases = ["dgrad", "wgrad"]  # bulk typically used for backward

        # Binary search values for num_sm: 1, 2, 4, 8, 16
        num_sm_values = []
        current = self.config.min_num_sm
        while current <= self.config.max_num_sm:
            num_sm_values.append(current)
            current *= 2

        for phase in phases:
            for num_sm in num_sm_values:
                config = TPOverlapTestConfig(
                    tp_size=tp_size,
                    operator=operator,
                    phase=phase,
                    overlap_method=OverlapMethod.BULK,
                    num_sm=num_sm,
                    set_sm_margin=0,
                )
                configs.append(config)
        return configs

    def generate_all_configs(self) -> List[TPOverlapTestConfig]:
        """Generate all test configurations."""
        all_configs = []

        for tp_size in self.tp_sizes:
            for operator in self.config.operators:
                if tp_size == 1:
                    # TP=1: No tensor parallelism, only baseline (no communication)
                    # This is used as the reference for comparing TP efficiency
                    all_configs.extend(self.generate_baseline_configs(tp_size, operator))
                else:
                    # TP>=2: Test baseline and various overlap configurations
                    # Generate baseline configs
                    all_configs.extend(self.generate_baseline_configs(tp_size, operator))
                    # Generate ring_exchange configs
                    all_configs.extend(
                        self.generate_ring_exchange_configs(tp_size, operator)
                    )
                    # Generate bulk configs
                    all_configs.extend(self.generate_bulk_configs(tp_size, operator))

        return all_configs

    def generate_configs_for_operator(
        self, tp_size: int, operator: str
    ) -> List[TPOverlapTestConfig]:
        """Generate all test configurations for a specific operator and TP size."""
        configs = []
        configs.extend(self.generate_baseline_configs(tp_size, operator))
        configs.extend(self.generate_ring_exchange_configs(tp_size, operator))
        configs.extend(self.generate_bulk_configs(tp_size, operator))
        return configs

    def generate_default_yaml_config(self) -> Dict[str, Dict[str, Any]]:
        """Generate a default YAML configuration based on best practices."""
        return {key: self._default_config_to_yaml(val) for key, val in DEFAULT_CONFIGS.items()}

    def _default_config_to_yaml(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Convert default config to YAML format."""
        result = {"method": config["method"].value}
        if config["method"] == OverlapMethod.RING_EXCHANGE:
            result["aggregate"] = config.get("aggregate", 0)
        elif config["method"] == OverlapMethod.BULK:
            result["num_sm"] = config.get("num_sm", 2)
            result["set_sm_margin"] = config.get("set_sm_margin", 0)
        return result


def generate_yaml_config_file(
    configs: List[TPOverlapTestConfig], output_path: str
) -> None:
    """Generate a YAML configuration file from test configs."""
    yaml_config = {}

    for config in configs:
        key = config.config_key
        yaml_config[key] = config.to_yaml_dict()

    with open(output_path, "w") as f:
        yaml.dump(yaml_config, f, default_flow_style=False, sort_keys=False)


def generate_single_test_yaml(config: TPOverlapTestConfig, output_path: str) -> None:
    """Generate a YAML file for a single test configuration.

    This creates a complete tp_comm_overlap_cfg.yaml with the given config
    for its specific operator/phase, and defaults for other operators.
    """
    # Start with default configs
    # Use a real small model to fetch config for generating defaults
    generator = TPOverlapConfigGenerator(
        TPOverlapTunerConfig(model_name="Qwen/Qwen2.5-0.5B")
    )
    yaml_config = generator.generate_default_yaml_config()

    # Override with the specific test config
    key = config.config_key
    if not config.is_baseline:
        yaml_config[key] = config.to_yaml_dict()

    with open(output_path, "w") as f:
        yaml.dump(yaml_config, f, default_flow_style=False, sort_keys=False)


def load_yaml_config(config_path: str) -> Dict[str, Dict[str, Any]]:
    """Load a YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
