import os
import GPUtil
from functools import lru_cache

GPU_SPECS_DATABASE = {
    # Ampere Architecture (30 Series & A100)
    "NVIDIA A100-SXM4-80GB": 312.0,
    "NVIDIA A100-PCIE-40GB": 312.0,
    "NVIDIA GeForce RTX 3090": 71.16,
    "NVIDIA GeForce RTX 3080 Ti": 68.5, # 修正为更精确的~68.5
    "NVIDIA GeForce RTX 3080": 59.6,  # 修正为更精确的~59.6

    # Hopper Architecture (H-Series)
    "NVIDIA H100 PCIe": 756.0,
    "NVIDIA H100 SXM5": 989.0,

    # Ada Lovelace Architecture (40 Series & L40)
    "NVIDIA GeForce RTX 4090": 165.16,
    "NVIDIA GeForce RTX 4080": 97.48,
    "NVIDIA L40": 181.0,

    # Blackwell Architecture (50 Series)
    "NVIDIA GeForce RTX 5090": 209.5,
    
    # Default/Fallback value
    "DEFAULT": 71.16 # RTX 3090
}

@lru_cache(maxsize=1)
def get_gpu_peak_flops() -> float:
    """
    Automatically detects the model of the first NVIDIA GPU in the system
    and queries its theoretical peak FP32 FLOPS from a database.

    Uses lru_cache to ensure this function is executed only once per run,
    avoiding repeated detections.

    Returns:
        float: The theoretical peak FLOPS of the GPU (in units of e12, e.g., 35.58e12).
            Returns a default value if no GPU is found or the model is not in the database.
    """


    env_flops = os.environ.get("GPU_PEAK_FLOPS")
    if env_flops:
        try:
            return float(env_flops)
        except ValueError:
            print(f"Warning: Invalid GPU_PEAK_FLOPS environment variable '{env_flops}'. Ignoring.")

    try:
        gpus = GPUtil.getGPUs()
        if not gpus:
            print("Warning: No NVIDIA GPU detected. Using default PEAK_FLOPS.")
            return GPU_SPECS_DATABASE["DEFAULT"] * 1e12

        gpu = gpus[0]
        gpu_name = gpu.name
        
        tflops = GPU_SPECS_DATABASE.get(gpu_name)
        
        if tflops:
            print(f"Detected GPU: {gpu_name}. Using {tflops} TFLOPS (BF16).")
            return tflops * 1e12
        else:
            print(f"Warning: GPU '{gpu_name}' not found in specs database. Using default PEAK_FLOPS.")
            return GPU_SPECS_DATABASE["DEFAULT"] * 1e12

    except Exception as e:
        print(f"Error detecting GPU: {e}. Using default PEAK_FLOPS.")
        return GPU_SPECS_DATABASE["DEFAULT"] * 1e12

GPU_PEAK_FLOPS = get_gpu_peak_flops()

