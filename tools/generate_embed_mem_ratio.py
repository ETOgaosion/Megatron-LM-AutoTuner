import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

MODELS = [
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-7B",
    "Qwen/Qwen2.5-72B",
    "Qwen/Qwen3-0.6B-Base",
    "Qwen/Qwen3-1.7B-Base",
    "Qwen/Qwen3-4B-Base",
    "Qwen/Qwen3-8B-Base",
    "Qwen/Qwen3-14B-Base",
    "Qwen/Qwen3-30B-A3B-Base",
    "Qwen/Qwen3-235B-A22B",
]


def get_args():
    parser = argparse.ArgumentParser(
        description="Generate embedding memory ratio configuration file."
    )
    parser.add_argument(
        "--config-base-dir",
        type=str,
        help="Directory containing model configuration files.",
        default="/data/common/models",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Path to the output configuration file.",
        default="outputs/tools/embed",
    )
    return parser.parse_args()


def validate_args(args):

    if not os.path.exists(args.config_base_dir):
        raise ValueError(
            f"Config base directory {args.config_base_dir} does not exist."
        )
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    return args


def get_model_embed_info(model_name: str, config_base_dir: str):
    config_path = os.path.join(config_base_dir, model_name, "config.json")
    if not os.path.exists(config_path):
        raise ValueError(f"Config file {config_path} does not exist.")

    with open(config_path, "r") as f:
        config = json.load(f)

    vocab_size = config.get("vocab_size", None)
    hidden_size = config.get("hidden_size", None)
    num_attention_heads = config.get("num_attention_heads", 1)
    num_query_groups = config.get("num_key_value_heads", num_attention_heads)
    kv_channels = config.get("head_dim", hidden_size // num_attention_heads)
    query_proj_size = hidden_size
    kv_proj_size = kv_channels * num_query_groups
    ffn_hidden_size = config.get("intermediate_size", 4 * hidden_size)
    num_experts = config.get("num_experts", 1)

    seqlen = 2048  # Assuming a default sequence length

    embedding_weights = (vocab_size * hidden_size) * 2 / (2**30)
    transformer_layer_weights = (
        (
            hidden_size * (query_proj_size + 2 * kv_proj_size)
            + hidden_size * query_proj_size
            + 2 * hidden_size * ffn_hidden_size * num_experts
        )
        * 2
        / (2**30)
    )
    return (embedding_weights, transformer_layer_weights)


def plot_and_export_embed_info(config_base_dir, output_dir):
    total_embedding_weights = []
    total_transformer_layer_weights = []
    for model in MODELS:
        print(f"Handling {model}")
        embedding_weights, transformer_layer_weights = get_model_embed_info(
            model, config_base_dir
        )

        total_embedding_weights.append(embedding_weights)
        total_transformer_layer_weights.append(transformer_layer_weights)

    # Create figure and axis
    fig, axes = plt.subplots(2, 1, figsize=(12, 12))

    # Extract model names (short version)
    model_names = [model.split("/")[-1] for model in MODELS]

    # Set up bar positions
    x = np.arange(len(model_names))
    width = 0.6

    # Create stacked bar chart
    bars1 = axes[0].bar(x, total_embedding_weights, width, label="Embedding Weights")
    bars2 = axes[0].bar(
        x,
        total_transformer_layer_weights,
        width,
        bottom=total_embedding_weights,
        label="Transformer Layer Weights",
    )

    # Customize the plot
    axes[0].set_xlabel("Model")
    axes[0].set_ylabel("Weights (GB)")
    axes[0].set_title("Embedding Weights vs Transformer Layer Weights by Model")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(model_names, rotation=45, ha="right")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    # Create stacked bar chart
    ratio = [
        x / (x + y)
        for x, y in zip(total_embedding_weights, total_transformer_layer_weights)
    ]
    bars1 = axes[1].bar(x, ratio, width, label="Embedding Weights Ratio")
    axes[1].hlines(1, -1, len(x) + 1, colors="black", linestyles="dashed")

    # Customize the plot
    axes[1].set_xlabel("Model")
    axes[1].set_ylabel("Ratio")
    axes[1].set_title("Embedding Weights vs Transformer Layer Weights Ratio")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(model_names, rotation=45, ha="right")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].set_xlim(-1, len(x))

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save the figure
    output_path = os.path.join(output_dir, "embed_mem_ratio.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Figure saved to {output_path}")

    # Also save the data as JSON
    data = {
        "models": MODELS,
        "embedding_weights (GB)": total_embedding_weights,
        "transformer_layer_weights (GB)": total_transformer_layer_weights,
    }
    json_path = os.path.join(output_dir, "embed_mem_ratio.json")
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Data saved to {json_path}")


args = get_args()
args = validate_args(args)
plot_and_export_embed_info(args.config_base_dir, args.output_dir)
