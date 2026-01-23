"""
Unit tests for AutoTuner.Profiler.overlap.config_generator module.
"""

import os
import tempfile
import unittest

from AutoTuner.Profiler.overlap.config_generator import (
    DEFAULT_CONFIGS,
    OPERATOR_LINEAR_TYPE,
    LinearType,
    OverlapMethod,
    Phase,
    TPOverlapConfigGenerator,
    TPOverlapTestConfig,
    TPOverlapTunerConfig,
    generate_single_test_yaml,
    generate_yaml_config_file,
    load_yaml_config,
)


class TestTPOverlapTestConfig(unittest.TestCase):
    """Tests for TPOverlapTestConfig dataclass."""

    def test_linear_type_column(self):
        """Test linear type detection for ColumnParallelLinear operators."""
        for operator in ["fc1", "qkv"]:
            config = TPOverlapTestConfig(
                tp_size=2,
                operator=operator,
                phase="fprop",
                overlap_method=OverlapMethod.RING_EXCHANGE,
            )
            self.assertEqual(config.linear_type, LinearType.COLUMN)

    def test_linear_type_row(self):
        """Test linear type detection for RowParallelLinear operators."""
        for operator in ["fc2", "proj"]:
            config = TPOverlapTestConfig(
                tp_size=2,
                operator=operator,
                phase="fprop",
                overlap_method=OverlapMethod.RING_EXCHANGE,
            )
            self.assertEqual(config.linear_type, LinearType.ROW)

    def test_config_key(self):
        """Test config key generation."""
        config = TPOverlapTestConfig(
            tp_size=2,
            operator="fc1",
            phase="fprop",
            overlap_method=OverlapMethod.RING_EXCHANGE,
        )
        self.assertEqual(config.config_key, "fc1_fprop")

    def test_test_id_baseline(self):
        """Test test ID generation for baseline config."""
        config = TPOverlapTestConfig(
            tp_size=2,
            operator="fc1",
            phase="fprop",
            overlap_method=OverlapMethod.RING_EXCHANGE,
            is_baseline=True,
        )
        self.assertEqual(config.get_test_id(), "tp2_fc1_fprop_baseline")

    def test_test_id_ring_exchange(self):
        """Test test ID generation for ring_exchange config."""
        config = TPOverlapTestConfig(
            tp_size=4,
            operator="qkv",
            phase="dgrad",
            overlap_method=OverlapMethod.RING_EXCHANGE,
            aggregate=1,
        )
        self.assertEqual(config.get_test_id(), "tp4_qkv_dgrad_ring_agg1")

    def test_test_id_bulk(self):
        """Test test ID generation for bulk config."""
        config = TPOverlapTestConfig(
            tp_size=8,
            operator="fc1",
            phase="wgrad",
            overlap_method=OverlapMethod.BULK,
            num_sm=4,
        )
        self.assertEqual(config.get_test_id(), "tp8_fc1_wgrad_bulk_sm4")

    def test_to_yaml_dict_ring_exchange(self):
        """Test YAML dict generation for ring_exchange method."""
        config = TPOverlapTestConfig(
            tp_size=2,
            operator="fc1",
            phase="fprop",
            overlap_method=OverlapMethod.RING_EXCHANGE,
            aggregate=1,
        )
        yaml_dict = config.to_yaml_dict()
        self.assertEqual(yaml_dict["method"], "ring_exchange")
        self.assertEqual(yaml_dict["aggregate"], 1)
        self.assertNotIn("num_sm", yaml_dict)

    def test_to_yaml_dict_bulk(self):
        """Test YAML dict generation for bulk method."""
        config = TPOverlapTestConfig(
            tp_size=2,
            operator="fc1",
            phase="dgrad",
            overlap_method=OverlapMethod.BULK,
            num_sm=8,
            set_sm_margin=0,
        )
        yaml_dict = config.to_yaml_dict()
        self.assertEqual(yaml_dict["method"], "bulk")
        self.assertEqual(yaml_dict["num_sm"], 8)
        self.assertEqual(yaml_dict["set_sm_margin"], 0)
        self.assertNotIn("aggregate", yaml_dict)


class TestTPOverlapTunerConfig(unittest.TestCase):
    """Tests for TPOverlapTunerConfig dataclass."""

    def test_auto_fetch_model_params(self):
        """Test that model parameters are auto-fetched from HuggingFace."""
        tuner_config = TPOverlapTunerConfig(
            model_name="Qwen/Qwen2.5-0.5B",
            max_tp_size=4,
        )

        # Verify that model parameters were fetched and set
        self.assertIsNotNone(tuner_config.hidden_size)
        self.assertIsNotNone(tuner_config.ffn_hidden_size)
        self.assertIsNotNone(tuner_config.num_attention_heads)
        self.assertIsNotNone(tuner_config.num_kv_heads)
        self.assertGreater(tuner_config.hidden_size, 0)
        self.assertGreater(tuner_config.ffn_hidden_size, 0)
        self.assertGreater(tuner_config.num_attention_heads, 0)
        self.assertGreater(tuner_config.num_kv_heads, 0)

    def test_default_values(self):
        """Test default values for tuner config."""
        tuner_config = TPOverlapTunerConfig(model_name="Qwen/Qwen2.5-0.5B")

        self.assertEqual(tuner_config.max_tp_size, 8)
        self.assertEqual(tuner_config.max_token_len, 8192)
        self.assertEqual(tuner_config.operators, ["fc1", "fc2", "qkv", "proj"])
        self.assertEqual(tuner_config.min_num_sm, 1)
        self.assertEqual(tuner_config.max_num_sm, 16)
        self.assertEqual(tuner_config.aggregate_values, [0, 1])


class TestTPOverlapConfigGenerator(unittest.TestCase):
    """Tests for TPOverlapConfigGenerator class."""

    def test_tp_sizes_includes_tp1(self):
        """Test that TP sizes include TP=1 as baseline."""
        tuner_config = TPOverlapTunerConfig(
            model_name="Qwen/Qwen2.5-0.5B",
            max_tp_size=8,
        )
        generator = TPOverlapConfigGenerator(tuner_config)

        self.assertIn(1, generator.tp_sizes)
        self.assertEqual(generator.tp_sizes, [1, 2, 4, 8])

    def test_tp_sizes_filtered_by_max(self):
        """Test that TP sizes are filtered by max_tp_size."""
        tuner_config = TPOverlapTunerConfig(
            model_name="Qwen/Qwen2.5-0.5B",
            max_tp_size=4,
        )
        generator = TPOverlapConfigGenerator(tuner_config)

        self.assertEqual(generator.tp_sizes, [1, 2, 4])
        self.assertNotIn(8, generator.tp_sizes)

    def test_generate_baseline_configs(self):
        """Test baseline config generation."""
        tuner_config = TPOverlapTunerConfig(model_name="Qwen/Qwen2.5-0.5B")
        generator = TPOverlapConfigGenerator(tuner_config)

        configs = generator.generate_baseline_configs(tp_size=2, operator="fc1")

        self.assertEqual(len(configs), 3)  # fprop, dgrad, wgrad
        for config in configs:
            self.assertTrue(config.is_baseline)
            self.assertEqual(config.tp_size, 2)
            self.assertEqual(config.operator, "fc1")

    def test_generate_ring_exchange_configs(self):
        """Test ring_exchange config generation."""
        tuner_config = TPOverlapTunerConfig(model_name="Qwen/Qwen2.5-0.5B")
        generator = TPOverlapConfigGenerator(tuner_config)

        configs = generator.generate_ring_exchange_configs(tp_size=2, operator="fc1")

        # 2 phases (fprop, dgrad) x 2 aggregate values (0, 1) = 4 configs
        self.assertEqual(len(configs), 4)
        for config in configs:
            self.assertEqual(config.overlap_method, OverlapMethod.RING_EXCHANGE)
            self.assertIn(config.aggregate, [0, 1])
            self.assertIn(config.phase, ["fprop", "dgrad"])

    def test_generate_bulk_configs(self):
        """Test bulk config generation with binary search num_sm values."""
        tuner_config = TPOverlapTunerConfig(
            model_name="Qwen/Qwen2.5-0.5B",
            min_num_sm=1,
            max_num_sm=16,
        )
        generator = TPOverlapConfigGenerator(tuner_config)

        configs = generator.generate_bulk_configs(tp_size=2, operator="fc1")

        # 2 phases (dgrad, wgrad) x 5 num_sm values (1,2,4,8,16) = 10 configs
        self.assertEqual(len(configs), 10)

        num_sm_values = set(c.num_sm for c in configs)
        self.assertEqual(num_sm_values, {1, 2, 4, 8, 16})

        for config in configs:
            self.assertEqual(config.overlap_method, OverlapMethod.BULK)
            self.assertIn(config.phase, ["dgrad", "wgrad"])

    def test_generate_all_configs_tp1_only_baseline(self):
        """Test that TP=1 only generates baseline configs (no overlap)."""
        tuner_config = TPOverlapTunerConfig(
            model_name="Qwen/Qwen2.5-0.5B",
            max_tp_size=2,
            operators=["fc1"],
        )
        generator = TPOverlapConfigGenerator(tuner_config)

        all_configs = generator.generate_all_configs()

        tp1_configs = [c for c in all_configs if c.tp_size == 1]
        tp2_configs = [c for c in all_configs if c.tp_size == 2]

        # TP=1: only baseline (3 phases)
        self.assertEqual(len(tp1_configs), 3)
        for config in tp1_configs:
            self.assertTrue(config.is_baseline)

        # TP=2: baseline + ring_exchange + bulk
        self.assertGreater(len(tp2_configs), 3)

    def test_generate_all_configs_count(self):
        """Test total config count for all operators."""
        tuner_config = TPOverlapTunerConfig(
            model_name="Qwen/Qwen2.5-0.5B",
            max_tp_size=8,
            operators=["fc1", "fc2", "qkv", "proj"],
        )
        generator = TPOverlapConfigGenerator(tuner_config)

        all_configs = generator.generate_all_configs()

        # Should have configs for all TP sizes and operators
        self.assertGreater(len(all_configs), 0)

        # Check that all operators are represented
        operators = set(c.operator for c in all_configs)
        self.assertEqual(operators, {"fc1", "fc2", "qkv", "proj"})

        # Check that all TP sizes are represented
        tp_sizes = set(c.tp_size for c in all_configs)
        self.assertEqual(tp_sizes, {1, 2, 4, 8})


class TestYAMLFunctions(unittest.TestCase):
    """Tests for YAML utility functions."""

    def test_generate_yaml_config_file(self):
        """Test YAML config file generation."""
        configs = [
            TPOverlapTestConfig(
                tp_size=2,
                operator="fc1",
                phase="fprop",
                overlap_method=OverlapMethod.RING_EXCHANGE,
                aggregate=0,
            ),
            TPOverlapTestConfig(
                tp_size=2,
                operator="fc1",
                phase="dgrad",
                overlap_method=OverlapMethod.BULK,
                num_sm=4,
            ),
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml_path = f.name

        try:
            generate_yaml_config_file(configs, yaml_path)
            loaded = load_yaml_config(yaml_path)

            self.assertIn("fc1_fprop", loaded)
            self.assertIn("fc1_dgrad", loaded)
            self.assertEqual(loaded["fc1_fprop"]["method"], "ring_exchange")
            self.assertEqual(loaded["fc1_dgrad"]["method"], "bulk")
            self.assertEqual(loaded["fc1_dgrad"]["num_sm"], 4)
        finally:
            os.unlink(yaml_path)

    def test_generate_single_test_yaml(self):
        """Test single test YAML generation with defaults."""
        config = TPOverlapTestConfig(
            tp_size=2,
            operator="fc1",
            phase="dgrad",
            overlap_method=OverlapMethod.BULK,
            num_sm=8,
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml_path = f.name

        try:
            generate_single_test_yaml(config, yaml_path)
            loaded = load_yaml_config(yaml_path)

            # Should have the test config
            self.assertIn("fc1_dgrad", loaded)
            self.assertEqual(loaded["fc1_dgrad"]["num_sm"], 8)

            # Should also have default configs for other operators
            self.assertIn("qkv_fprop", loaded)
            self.assertIn("proj_fprop", loaded)
        finally:
            os.unlink(yaml_path)


class TestConstants(unittest.TestCase):
    """Tests for module constants."""

    def test_operator_linear_type_mapping(self):
        """Test operator to linear type mapping."""
        self.assertEqual(OPERATOR_LINEAR_TYPE["fc1"], LinearType.COLUMN)
        self.assertEqual(OPERATOR_LINEAR_TYPE["fc2"], LinearType.ROW)
        self.assertEqual(OPERATOR_LINEAR_TYPE["qkv"], LinearType.COLUMN)
        self.assertEqual(OPERATOR_LINEAR_TYPE["proj"], LinearType.ROW)

    def test_default_configs_keys(self):
        """Test default config keys exist."""
        expected_keys = [
            "qkv_fprop",
            "proj_fprop",
            "fc1_fprop",
            "fc2_fprop",
            "fc2_dgrad",
            "fc1_dgrad",
            "fc1_wgrad",
            "proj_dgrad",
            "qkv_dgrad",
            "qkv_wgrad",
        ]
        for key in expected_keys:
            self.assertIn(key, DEFAULT_CONFIGS)

    def test_overlap_method_enum(self):
        """Test OverlapMethod enum values."""
        self.assertEqual(OverlapMethod.RING_EXCHANGE.value, "ring_exchange")
        self.assertEqual(OverlapMethod.BULK.value, "bulk")
        self.assertEqual(OverlapMethod.PIPELINE.value, "pipeline")

    def test_phase_enum(self):
        """Test Phase enum values."""
        self.assertEqual(Phase.FPROP.value, "fprop")
        self.assertEqual(Phase.DGRAD.value, "dgrad")
        self.assertEqual(Phase.WGRAD.value, "wgrad")


if __name__ == "__main__":
    unittest.main()
