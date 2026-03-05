# Requirement: Compare with EPLB

CONTEXT: Now try to add scripts to use EPLB in this directory to run load balance on such data and compare with our system

TASK: Input: EP size
1. Read EPLB submodule and eplb.py in it
2. Use EPLB to produce load balance on the same data
3. draw the token balance result like ours
4. Compare memory occupations, memory include:
    - Experts weights (EPLB copy experts)
    - token count (We use Qwen/Qwen1.5-MoE-A2.7B-Chat , read `/data/common/models/Qwen/Qwen1.5-MoE-A2.7B/config.json` for activation calculation)
5. draw memory occupation comparison of different systems