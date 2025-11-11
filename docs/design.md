
## 🚀 Generating `HiddenStatus`

When testing our operators (ops), we frequently require `HiddenStatus` as an input. To address this need, we have implemented a base class called `TestWithHiddenInputs`, which is capable of providing `HiddenStatus` for testing.

---

### 📝 Background

First, a brief overview of the current operator testing framework:

1. To test an operator (Op) **OP**, we implement the **OP class** (which contains the operator's logic) and a corresponding **TestOP class**.
    
2. The **TestOP class** inherits methods for testing operators, including a crucial method named `prepare_input`, which is responsible for generating the operator's inputs.
    
3. Finally, a **`Launcher` class** calls methods within **TestOP** to execute the test. Specifically, it calls the **TestOP**'s `prepare_input` method and uses the returned values as inputs for calling the **OP class**'s `forward` method.
    

---

### ✨ `TestWithHiddenInputs`

`TestWithHiddenInputs` is designed to serve as a base class for your **TestOP** classes. It provides an implementation of the `prepare_input` method that automatically supplies `HiddenStatus`.

The only additional step required to use `TestWithHiddenInputs` is to initialize it within your custom **TestOP** class.

Here is an example demonstrating how to test a Decoder operator:

- **1. Implement the Op:** Create the `DecoderForTest` class (the OP class).
    
- **2. Implement the TestOp:** Create the `TestDecoderWithHiddenInputs` class, which **inherits from `TestWithHiddenInputs`**.
	-  Initialize the base class as follows:
```python
class TestDecoderWithHiddenInputs(TestWithHiddenInputs):

    def __init__(
        self,
        tf_config: TransformerConfig,
        hf_config: PretrainedConfig,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
        profile_mode: int = 0,
        warmup_iters: int = 2,
    ):
        super().__init__(
            hf_config=hf_config,
            tf_config=tf_config,
            profile_mode=profile_mode,
            warmup_iters=warmup_iters,
            tp_group=tp_group,
        )
```

- **4. Execution:** `TestDecoderWithHiddenInputs` inherits the `prepare_input` method from its parent class, which is then called by the `Launcher` to generate the operator's inputs.
    

---

#### Important Notes

1. **Initialization Parameters:** The parameters shown in the example (`tf_config`, `hf_config`, etc.) are the fundamental parameters provided by the `Launcher` when it initializes the **TestOP** class. When initializing `TestWithHiddenInputs`, there are many other parameters (primarily for RoPE/Rotary Positional Embedding) that use **default values** and can typically be left as is.
    
2. **`prepare_input` Return Value:** The `prepare_input` method in `TestWithHiddenInputs` returns a tuple structured as follows:
    > 
    > ```Python
    > # The first element is the main decoder input (HiddenStatus), 
    > # and the others are auxiliary inputs for attention mask and rotary embedding.
    > return (
    >     decoder_input,    # This serves as the HiddenStatus
    >     attention_mask,
    >     rotary_pos_emb,
    >     packed_seq_params,
    >     sequence_len_offset,
    > )
    > ```