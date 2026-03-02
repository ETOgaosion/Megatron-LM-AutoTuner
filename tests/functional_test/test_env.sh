MODEL_NAME="Qwen/Qwen3-0.6B"
TEST_CASES_FILE="local/qwen3_0_6b.json"

# Test CPU embedding op only
TEST_OPS_LIST=("TEColumnParallelLinear")
TEST_CASE_IDXES=None

TP_COMM_OVERLAP=True

TP_SIZE=2
CP_SIZE=1
EP_SIZE=1
ETP_SIZE=1

NSYS_BIN=/usr/local/cuda-12.8/NsightSystems-cli-2025.1.1/bin/nsys