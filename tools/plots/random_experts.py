import numpy as np
import os
import matplotlib.pyplot as plt

# Generate random data
num_experts = 128
expert_indices = np.arange(num_experts)
token_amounts = np.random.randint(200, 700, num_experts)

# Create bar chart
plt.figure(figsize=(10, 6))
plt.bar(expert_indices, token_amounts, color='steelblue')
plt.xlabel('Expert Index')
plt.ylabel('Token Amounts')
plt.title('Token Distribution Across Experts')
plt.xticks([i for i in range(0, num_experts, 8)])
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
os.makedirs('outputs/plots/no_meaning', exist_ok=True)
plt.savefig('outputs/plots/no_meaning/random_experts.png')
plt.show()