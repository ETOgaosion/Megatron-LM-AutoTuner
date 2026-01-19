# TP Overlap Tuner - Test Environment Configuration
# Copy this file to test_env.sh and modify for your setup

# Model name from HuggingFace
# Model parameters (hidden_size, ffn_hidden_size, num_attention_heads, num_kv_heads)
# are automatically fetched from HuggingFace config
MODEL_NAME="Qwen/Qwen3-0.6B"

# Tuning parameters
MAX_TP_SIZE=8                       # Tests TP=2, 4, 8 up to this value
MAX_TOKEN_LEN=8192                  # Sequence length for profiling
OPERATORS="fc1 fc2 qkv proj"        # Operators to tune (space-separated)

# Analysis parameters
OVERLAP_THRESHOLD=0.5               # Minimum overlap ratio to consider effective

# Binary search parameters for bulk method
MIN_NUM_SM=1
MAX_NUM_SM=16

# Output directory (leave empty for auto-generated timestamp)
OUTPUT_DIR=""
