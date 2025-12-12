import os
from typing import Dict

import torch
from megatron.core.transformer.enums import AttnBackend
from transformers import AutoConfig

from verl.models.mcore.registry import hf_to_mcore_config

BASE_DATA_DIR = os.environ.get("BASE_DATA_DIR", "/data/common/")
HUGGINGFACE_MODEL_DIR = os.environ.get(
    "HUGGINGFACE_MODEL_DIR", os.path.join(BASE_DATA_DIR, "models")
)
HUGGINGFACE_MODEL_DIR = ""


def get_hf_model_config(model_name: str, **kwargs) -> AutoConfig:
    config = AutoConfig.from_pretrained(
        os.path.join(HUGGINGFACE_MODEL_DIR, model_name), **kwargs
    )
    if model_name == "deepseek-ai/DeepSeek-V3-Base":
        config.num_nextn_predict_layers = 0
        if hasattr(config, "quantization_config"):
            delattr(config, "quantization_config")
    return config


def get_mcore_model_config(
    model_name: str,
    dtype: torch.dtype = torch.bfloat16,
    hf_kwargs: Dict = {},
    **override_tf_config_kwargs,
) -> Dict:
    hf_config = get_hf_model_config(model_name, **hf_kwargs)
    mcore_config = hf_to_mcore_config(hf_config, dtype, **override_tf_config_kwargs)
    return mcore_config


def get_mcore_model_config_from_hf_config(
    hf_config: str, dtype: torch.dtype = torch.bfloat16, **override_tf_config_kwargs
) -> Dict:
    override_tf_config_kwargs["attention_backend"] = AttnBackend.flash
    mcore_config = hf_to_mcore_config(hf_config, dtype, **override_tf_config_kwargs)
    return mcore_config
