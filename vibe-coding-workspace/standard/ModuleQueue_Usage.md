# ModuleQueue Usage Guide

## Overview

`ModuleQueue` is a memory-efficient variant of `GPTModel` designed for the last pipeline stage in RLHF training scenarios. It reduces GPU memory usage by:

1. Keeping post-process (output layer) weights on CPU initially
2. During forward pass: offloading computed transformer layers to CPU while loading chunks of post-process weights to GPU
3. During backward pass: offloading post-process weights after computation and loading transformer layers back from CPU

This approach enables better load balancing and reduces peak GPU memory usage by overlapping computation with CPU-GPU data transfers.

## Configuration

ModuleQueue is controlled by two configuration parameters in `TransformerConfig`:

- `enable_module_queue` (bool, default: False): Enable ModuleQueue for last pipeline stage
- `module_queue_num_chunks` (int, default: 4): Number of chunks to split the output layer weights into

## Usage Example

### Basic Usage

```python
from megatron.core.models.gpt import ModuleQueue
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.transformer.transformer_config import TransformerConfig

# Configure transformer with ModuleQueue enabled
config = TransformerConfig(
    num_layers=32,
    hidden_size=4096,
    num_attention_heads=32,
    enable_module_queue=True,  # Enable ModuleQueue
    module_queue_num_chunks=4,  # Split output layer into 4 chunks
)

# Create model - only effective when post_process=True (last pipeline stage)
model = ModuleQueue(
    config=config,
    transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
    vocab_size=50257,
    max_sequence_length=2048,
    pre_process=False,
    post_process=True,  # Must be True for ModuleQueue to activate
)

# Use like regular GPTModel
output = model.forward(
    input_ids=input_ids,
    position_ids=position_ids,
    attention_mask=attention_mask,
)
```

### Replacing GPTModel with ModuleQueue

ModuleQueue is a drop-in replacement for GPTModel on the last pipeline stage:

```python
# Before: Using GPTModel
from megatron.core.models.gpt import GPTModel

model = GPTModel(
    config=config,
    transformer_layer_spec=layer_spec,
    vocab_size=vocab_size,
    max_sequence_length=max_seq_len,
    post_process=True,
)

# After: Using ModuleQueue
from megatron.core.models.gpt import ModuleQueue

config.enable_module_queue = True
config.module_queue_num_chunks = 4

model = ModuleQueue(
    config=config,
    transformer_layer_spec=layer_spec,
    vocab_size=vocab_size,
    max_sequence_length=max_seq_len,
    post_process=True,
)
```

### Conditional Usage Based on Pipeline Stage

```python
from megatron.core.models.gpt import GPTModel, ModuleQueue
from megatron.core import parallel_state

# Determine if this is the last pipeline stage
pp_rank = parallel_state.get_pipeline_model_parallel_rank()
pp_size = parallel_state.get_pipeline_model_parallel_world_size()
is_last_stage = (pp_rank == pp_size - 1)

# Use ModuleQueue only for last stage
ModelClass = ModuleQueue if (is_last_stage and config.enable_module_queue) else GPTModel

model = ModelClass(
    config=config,
    transformer_layer_spec=layer_spec,
    vocab_size=vocab_size,
    max_sequence_length=max_seq_len,
    pre_process=(pp_rank == 0),
    post_process=is_last_stage,
)
```

## Configuration Parameters

### enable_module_queue

- **Type**: bool
- **Default**: False
- **Description**: Enable ModuleQueue mechanism for memory-efficient training. Only takes effect when `post_process=True`.

### module_queue_num_chunks

- **Type**: int
- **Default**: 4
- **Description**: Number of chunks to split the output layer weights into. More chunks allow finer-grained overlap between computation and data transfer, but may increase overhead. Recommended values: 2-8.

## How It Works

### Forward Pass

1. Initially, all transformer layers are on GPU, output layer weights are on CPU
2. As each transformer layer completes:
   - Layer weights are offloaded to CPU asynchronously
   - A chunk of output layer weights is loaded to GPU asynchronously
3. By the time all layers complete, output layer is fully loaded on GPU
4. Output layer computation proceeds normally

### Backward Pass

1. Output layer backward completes first
2. Output layer weights are offloaded to CPU
3. Transformer layers are loaded back to GPU in reverse order (last layer first)
4. Backward pass proceeds through transformer layers

### Memory Savings

For a model with:
- N transformer layers
- Output layer size: V (vocab size) × H (hidden size)

Peak memory reduction: Approximately (N × layer_size) - (V × H / num_chunks)

## Performance Considerations

### Chunk Size Selection

- **Fewer chunks (2-3)**: Lower overhead, less overlap
- **More chunks (6-8)**: Better overlap, higher overhead
- **Recommended**: Start with 4 chunks and tune based on profiling

### When to Use ModuleQueue

**Use ModuleQueue when:**
- You're running the last pipeline stage
- GPU memory is constrained
- Output layer is large (large vocabulary)
- You have sufficient CPU-GPU bandwidth

**Don't use ModuleQueue when:**
- GPU memory is not a constraint
- Model is small
- CPU-GPU bandwidth is limited
- You're not using pipeline parallelism

### Overhead

ModuleQueue introduces:
- CPU-GPU transfer overhead (mitigated by async transfers and overlap)
- Small bookkeeping overhead for queue management

These are typically negligible compared to memory savings in large-scale training.

## Compatibility

### Compatible with:
- Pipeline parallelism (PP)
- Tensor parallelism (TP)
- Data parallelism (DP)
- Mixed precision training (FP16/BF16)
- Gradient accumulation

### Not compatible with:
- Inference mode (automatically disabled)
- Stages without post_process=True (automatically disabled)

## Debugging

### Checking if ModuleQueue is Active

```python
if hasattr(model, '_module_queue_enabled'):
    print(f"ModuleQueue enabled: {model._module_queue_enabled}")
    if model._module_queue_enabled:
        print(f"Number of chunks: {model.num_chunks}")
        print(f"Number of layers: {model.num_layers}")
else:
    print("Model is not a ModuleQueue instance")
```

### Monitoring Memory Usage

```python
import torch

# Before forward pass
torch.cuda.reset_peak_memory_stats()

# Forward and backward
loss = model.forward(...)
loss.backward()

# Check peak memory
peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
print(f"Peak GPU memory: {peak_memory:.2f} GB")
```

## Example: Complete Training Loop

```python
import torch
from megatron.core.models.gpt import ModuleQueue
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec

# Configure
config = TransformerConfig(
    num_layers=32,
    hidden_size=4096,
    num_attention_heads=32,
    enable_module_queue=True,
    module_queue_num_chunks=4,
)

# Create model
model = ModuleQueue(
    config=config,
    transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
    vocab_size=50257,
    max_sequence_length=2048,
    post_process=True,
).cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training loop
for batch in dataloader:
    input_ids = batch['input_ids'].cuda()
    position_ids = batch['position_ids'].cuda()
    attention_mask = batch['attention_mask'].cuda()
    labels = batch['labels'].cuda()

    # Forward pass (ModuleQueue handles weight movement automatically)
    loss = model.forward(
        input_ids=input_ids,
        position_ids=position_ids,
        attention_mask=attention_mask,
        labels=labels,
    )

    # Backward pass (ModuleQueue handles weight movement automatically)
    loss.backward()

    # Optimizer step
    optimizer.step()
    optimizer.zero_grad()
```

## Testing

Run the unit tests to verify ModuleQueue functionality:

```bash
cd Megatron-LM-Enhanced
pytest tests/unit_tests/models/test_module_queue.py -v
```

## References

- Image diagram: `vibe-coding-workspace/requirements/photos/module_queue.png`
- Requirements: `vibe-coding-workspace/requirements/modulequeue.md`
- Implementation: `megatron/core/models/gpt/module_queue_gpt_model.py`
- Configuration: `megatron/core/transformer/transformer_config.py`
