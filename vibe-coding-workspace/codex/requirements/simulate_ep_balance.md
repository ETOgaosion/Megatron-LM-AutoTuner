# Requirement: Simulate EP balance results

CONTEXT: In `simulator/simulate_ep_balance`, there are original data of number of tokens hold by every expert. You should write an algorithm to reorder experts and let every GPU holds more balanced workloads.

TASK:
1. write an algorithm (can use karmarkar-karp algorithm) to reorder experts to let every GPU holding experts with same total token count
    - Input: EP size
    - Output: New maps and GPU holds token number (old and new), and also the varience of data
2. paint image to show GPU holding token difference, show differnt EP size results, output to `outputs` directory, make a special subfolder.
    - Before and after reorder

CONSTRAINTS:
1. All GPU holding same tokens