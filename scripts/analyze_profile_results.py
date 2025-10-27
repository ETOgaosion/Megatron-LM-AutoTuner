import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

# --- Constants ---
HARDWARE_PEAK_TFLOPS = 35.58  # Peak TFLOPS of the hardware for MFU calculation.
THEORETICAL_FILE = "theoretical_performance.json"
TIMING_FILE = "timing.json"
WEIGHTS_FILE = "memory_weights.json"
ACTIVATION_FILE = "memory_activation.json"

# --- Utility Functions ---

def bytes_to_megabytes(b: float) -> float:
    """Converts bytes to megabytes."""
    return b / 1024 / 1024

def parse_memory_str(mem_str: str) -> int:
    """Parses a memory string (e.g., '1024 KB') into bytes."""
    if not isinstance(mem_str, str):
        return 0
    try:
        val_str, unit = mem_str.strip().split()
        val = float(val_str)
        unit = unit.lower()
        if unit == 'kb': return int(val * 1024)
        if unit == 'mb': return int(val * 1024 * 1024)
        if unit == 'gb': return int(val * 1024 * 1024 * 1024)
        if unit == 'b': return int(val)
    except (ValueError, IndexError):
        return 0
    return 0

def parse_time_str(time_str: str) -> float:
    """Parses a time string (e.g., '5.23 ms') into seconds."""
    if not isinstance(time_str, str):
        return 0.0
    return float(time_str.strip().split()[0])

def find_nested_value(data: Dict, path: list) -> Any:
    """Safely retrieves a value from a nested dictionary using a list of keys."""
    try:
        temp = data
        for key in path:
            temp = temp[key]
        return temp
    except (KeyError, TypeError, AttributeError):
        return None

def calculate_percentage_diff(theoretical: float, practical: float) -> float:
    """Calculates the percentage difference between two values."""
    if theoretical == 0 and practical == 0: return 0.0
    if theoretical == 0: return float('inf')
    return ((practical - theoretical) / theoretical) * 100

# --- Core Analysis Logic ---

def analyze_results(result_dir: Path):
    """
    Loads, analyzes, and prints performance data from a specified rank directory.
    """
    print(f"\n{' Analyzing Results in ':=^80}")
    print(f"Directory: {result_dir}")
    print(f"{'=':=^80}\n")

    # Load all required JSON data files.
    required_files = [THEORETICAL_FILE, TIMING_FILE, WEIGHTS_FILE, ACTIVATION_FILE]
    json_data = {}
    try:
        for file_name in required_files:
            with open(result_dir / file_name, 'r') as f:
                json_data[file_name] = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: Could not find required file: {e.filename}")
        print("Please ensure the directory contains all necessary JSON files.")
        return
    
    theoretical_data = json_data[THEORETICAL_FILE]
    timing_data = json_data[TIMING_FILE]
    weights_data = json_data[WEIGHTS_FILE]
    activation_data = json_data[ACTIVATION_FILE]

    # Iterate through each operator and its test cases.
    for op_name, op_data in theoretical_data.items():
        print(f"--- Operator: {op_name} ---")
        
        def traverse_and_analyze(data_node: Dict, path: list):
            # A leaf node contains direct performance values, not nested dictionaries.
            is_leaf_node = any(not isinstance(v, dict) for v in data_node.values())

            if is_leaf_node:
                test_case_str = ' -> '.join(
                    f"{k.split('=')[0]}={v}" for k, v in zip(path[::2], path[1::2])
                )
                
                print(f"\n  Test Case: {test_case_str}")
                
                # Find theoretical and practical values from loaded data.
                theo_mem = find_nested_value(theoretical_data, [op_name, "memory"] + path)
                theo_flops = find_nested_value(theoretical_data, [op_name, "flops"] + path)
                
                prac_weights_str = weights_data.get(op_name, {}).get("mem_diff")
                prac_activation_str = find_nested_value(activation_data, path + [op_name])
                prac_time_fwd_str = find_nested_value(timing_data, [op_name, "forward"] + path)
                prac_time_bwd_str = find_nested_value(timing_data, [op_name, "backward"] + path)

                # Parse theoretical values.
                theo_weights = theo_mem.get('weights', 0) if theo_mem else 0
                theo_activations = theo_mem.get('activations', 0) if theo_mem else 0
                
                # Parse practical (measured) values.
                prac_weights = parse_memory_str(prac_weights_str)
                prac_activations = parse_memory_str(prac_activation_str)
                prac_time_fwd = parse_time_str(prac_time_fwd_str)
                prac_time_bwd = parse_time_str(prac_time_bwd_str)

                # --- Memory Analysis ---
                print("  - Memory Analysis:")
                diff_w = calculate_percentage_diff(theo_weights, prac_weights)
                print(f"    - Weights:     Theoretical={bytes_to_megabytes(theo_weights):>7.2f} MB,  Practical={bytes_to_megabytes(prac_weights):>7.2f} MB,  Diff={diff_w:>6.1f}%")
                diff_a = calculate_percentage_diff(theo_activations, prac_activations)
                print(f"    - Activations: Theoretical={bytes_to_megabytes(theo_activations):>7.2f} MB,  Practical={bytes_to_megabytes(prac_activations):>7.2f} MB,  Diff={diff_a:>6.1f}%")

                # --- Performance Analysis ---
                print("  - Performance Analysis:")
                if prac_time_fwd > 0:
                    theo_fwd_flops = theo_flops.get('forward', 0) if theo_flops else 0
                    
                    # Calculate performance metrics.
                    achieved_fwd_tflops = (theo_fwd_flops / prac_time_fwd) / 1e12
                    mfu_fwd = (achieved_fwd_tflops / HARDWARE_PEAK_TFLOPS) * 100
                    theo_time_fwd_ms = ((theo_fwd_flops / (HARDWARE_PEAK_TFLOPS * 1e12)) * 1000) if HARDWARE_PEAK_TFLOPS > 0 else 0
                    
                    print(f"    - Forward:     Theoretical Time={theo_time_fwd_ms:>7.3f} ms | Practical Time={prac_time_fwd*1000:>7.3f} ms | Achieved={achieved_fwd_tflops:>7.3f} TFLOPS | MFU={mfu_fwd:>5.2f}%")

                if prac_time_bwd > 0:
                    theo_bwd_flops = theo_flops.get('backward', 0) if theo_flops else 0
                    
                    achieved_bwd_tflops = (theo_bwd_flops / prac_time_bwd) / 1e12
                    mfu_bwd = (achieved_bwd_tflops / HARDWARE_PEAK_TFLOPS) * 100
                    theo_time_bwd_ms = ((theo_bwd_flops / (HARDWARE_PEAK_TFLOPS * 1e12)) * 1000) if HARDWARE_PEAK_TFLOPS > 0 else 0
                    
                    print(f"    - Backward:    Theoretical Time={theo_time_bwd_ms:>7.3f} ms | Practical Time={prac_time_bwd*1000:>7.3f} ms | Achieved={achieved_bwd_tflops:>7.3f} TFLOPS | MFU={mfu_bwd:>5.2f}%")
                
            else:
                # Recursively traverse deeper into the nested dictionary.
                for key, next_node in data_node.items():
                    traverse_and_analyze(next_node, path + [key])

        if op_data.get("memory"):
            traverse_and_analyze(op_data["memory"], [])
        print(f"--- End Operator: {op_name} ---\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze AutoTuner profiling results by comparing theoretical and practical values."
    )
    parser.add_argument(
        "model_dir",
        type=str,
        help="Path to the model-specific directory containing the 'collect_data' folder (e.g., 'outputs/2025-10-27_10-00-00/Qwen/Qwen3-0.6B')."
    )

    args = parser.parse_args()
    
    model_path = Path(args.model_dir)
    collect_data_path = model_path / "collect_data"

    if not collect_data_path.is_dir():
        print(f"Error: Directory '{collect_data_path}' not found.")
        print("Please provide a path that contains a 'collect_data' subdirectory.")
    else:
        # Find and sort all 'rank_X' directories for analysis.
        rank_dirs = sorted([d for d in collect_data_path.iterdir() if d.is_dir() and d.name.startswith('rank_')])
        
        if not rank_dirs:
            print(f"No 'rank_X' directories found in '{collect_data_path}'.")
        else:
            print(f"Found {len(rank_dirs)} rank directories to analyze.")
            for rank_dir in rank_dirs:
                analyze_results(rank_dir)
