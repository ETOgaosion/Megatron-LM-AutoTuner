import torch
import math
from typing import Tuple, Optional, List

class HiddenStateGenerator:
    """
    Utility class for generating random Hidden State tensors for language model intermediate operators.
    
    Tensors you can generate:
    - Input IDs with padding to simulate variable-length sequences.
    - Hidden States in THD format ([T_nnz, H], can support dimension addition or decrease by setting ffn_hidden_size).
    - Hidden States in SBH format ([S, B, H]).
    - Attention Masks combining Causal and Padding constraints.
    """
    
    def __init__(self, 
                 hidden_size: int = 1024,                
                 ffn_hidden_size: Optional[int] = None,     # Intermediate size for MLP
                 dtype: torch.dtype = torch.bfloat16,
                 device: str = 'cuda'):
        
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.dtype = dtype
        self.device = device
        self.pad_token_id = 0      # Padding token ID
        self.vocab_size = 50257    # Placeholder vocabulary size (50000 + 256 + 1)
        
        print(f"hidden_size: {self.hidden_size}, ffn_hidden_size: {self.ffn_hidden_size}, dtype: {self.dtype}")


    def generate_input_ids(self, 
                           micro_batch_size: int, 
                           sequence_length: int, 
                           target_fill_ratio: float = 0.95,
                           min_length_ratio: float = 0.1) -> torch.Tensor:
        """
        Generates input IDs with padding, where each sequence's valid length 
        is randomly and broadly chosen between a minimum and a target maximum length.
        
        Args:
            micro_batch_size (int): Number of sequences in the batch (B).
            sequence_length (int): Maximum length of a sequence (S).
            target_fill_ratio (float): The target maximum ratio of non-padded tokens 
                                       (e.g., 0.95 means lengths are typically <= 0.95 * S).
            min_length_ratio (float): The minimum ratio of S for sequence length (e.g., 0.1).
        
        Returns:
            torch.Tensor: The input IDs tensor of shape [B, S].
        """
        
        # Initialize tensor with all padding IDs
        input_ids = torch.full(
            (micro_batch_size, sequence_length), 
            self.pad_token_id, 
            dtype=torch.long, 
            device=self.device
        )

        min_valid_length = max(1, int(sequence_length * min_length_ratio))
        max_limit_length = max(min_valid_length, int(sequence_length * target_fill_ratio))
        
        sequence_lengths_list = [] 

        for i in range(micro_batch_size):
            current_valid_length = torch.randint(
                min_valid_length, max_limit_length + 1, (1,)
            ).item()
            sequence_lengths_list.append(current_valid_length)
            input_ids[i, :current_valid_length] = torch.randint(
                1, self.vocab_size, (current_valid_length,), device=self.device
            )

        # IDS Logging
        B = micro_batch_size
        S = sequence_length
        total_nnz_tokens = sum(sequence_lengths_list)
        
        print(f" [IDs] Generating Input IDs: Shape [{B}, {S}]")
        print(f" [IDs] Valid length range: [{min(sequence_lengths_list)}, {max(sequence_lengths_list)}] (Target Max: {max_limit_length})")
        print(f" [IDs] Total T_nnz: {total_nnz_tokens}")
        
        return input_ids

    # THD Hidden State (Flattened, T_nnz derived from input_ids)
    def generate_thd_hidden_state(self, 
                                 input_ids: torch.Tensor,
                                 target_ffn_dim: bool = False,
                                 requires_grad: bool = True) -> torch.Tensor:
        """
        Generates a flattened activation tensor (THD format: [total_valid_tokens, feature_dim]).
        T_nnz is strictly calculated from the non-padded tokens in input_ids.
        
        Args:
            input_ids (torch.Tensor): The [B, S] input IDs used to calculate T_nnz.
            target_ffn_dim (bool): If True, output dimension is D_ffn (for dimension expand or decrease).
        
        Returns:
            the generated THD tensor.
        """
        # Calculate T_nnz
        is_valid_token = (input_ids != self.pad_token_id)
        total_valid_tokens = is_valid_token.sum().item()
        
        # Determine the feature dimension (expanded/decrease or standard)
        feature_dim = self.ffn_hidden_size if target_ffn_dim else self.hidden_size
        
        # THD Logging
        print(f" [THD] Generating Hidden State: Shape [{total_valid_tokens}, {feature_dim}] (D={'Expand_or_Decrease' if target_ffn_dim else 'Normal'})")
        print(f" [THD] Total Valid Tokens (T_nnz): {total_valid_tokens}")
        return torch.randn(
            (total_valid_tokens, feature_dim), 
            device=self.device, 
            dtype=self.dtype
        ).requires_grad_(requires_grad)


    # SBH Hidden State (S x B x H)
    def generate_sbh_hidden_state(self, 
                                 micro_batch_size: int, 
                                 sequence_length: int, 
                                 requires_grad: bool = True) -> torch.Tensor:
        """
        Generates a structured activation tensor (SBH format: [S, B, H]).
        Used for operators that require the original sequence structure, like QKV.
        
        Returns:
            the generated SBH tensor.
        """
        tensor_size = (sequence_length, micro_batch_size, self.hidden_size)
        print(f" [SBH] Generating Hidden State: Shape {list(tensor_size)}")

        return torch.randn(
            tensor_size, 
            device=self.device, 
            dtype=self.dtype
        ).requires_grad_(requires_grad)

    # Attention Mask (Combines Causal and Padding)
    def generate_attention_mask(self, 
                                input_ids: torch.Tensor, 
                                is_causal: bool = True,
                                mask_value: float = -10000.0) -> torch.Tensor:
        """
        Generates the 4D floating-point mask ([B, 1, S, S]) by combining Causal and Padding masks.
        
        Args:
            input_ids: used to derive padding locations.
            is_causal: If True, applies a lower-triangular mask.
            mask_value: used to mask logits.
            
        Returns:
            the final [B, 1, S, S] attention mask.
        """
        B, S = input_ids.shape
        
        mask_tensor = torch.tensor(mask_value, dtype=self.dtype, device=self.device)

        # Padding Mask (Identifies invalid K/V tokens)
        padding_mask_bool = (input_ids != self.pad_token_id).view(B, 1, 1, S)
        
        padding_mask = torch.where(
            padding_mask_bool,
            torch.tensor(0.0, dtype=self.dtype, device=self.device),
            mask_tensor
        )
        
        # Causal Mask (Lower-triangular S x S matrix)
        if is_causal:
            causal_mask_matrix = torch.full(
                (S, S), mask_value, dtype=self.dtype, device=self.device
            )
            causal_mask_matrix = causal_mask_matrix.triu(diagonal=1).T
            
            # Combine Causal and Padding constraints via addition/broadcasting
            # padding_mask [B, 1, 1, S] 
            # causal_mask_matrix [S, S] 
            final_mask = padding_mask + causal_mask_matrix.unsqueeze(0).unsqueeze(0)
        else:
            # Only use padding mask
            final_mask = padding_mask.expand(B, 1, S, S)

        print(f" [Mask] Generating {'Causal+Padding' if is_causal else 'Padding Only'} Mask: Shape {list(final_mask.shape)}")

        return final_mask
    

# Usage Example, now commented out to avoid execution during imports

# if __name__ == '__main__':
#     device_to_use = 'cuda' if torch.cuda.is_available() else 'cpu'
    
#     generator = HiddenStateGenerator(
#         hidden_size=1024, 
#         dtype=torch.float16, 
#         device=device_to_use,
#         ffn_hidden_size=4096  # Set intermediate MLP size to 4H
#     )

#     BATCH_SIZE = 4
#     SEQ_LENGTH = 512

#     TARGET_FILL_RATIO = 0.99 
#     MIN_LENGTH_RATIO = 0.2

#     print("\n" + "="*50)
#     print("--- 1. Generating Input IDs ---")
#     print("="*50)
    

#     input_ids_tensor = generator.generate_input_ids(
#         micro_batch_size=BATCH_SIZE,
#         sequence_length=SEQ_LENGTH,
#         target_fill_ratio=TARGET_FILL_RATIO,
#         min_length_ratio=MIN_LENGTH_RATIO
#     )
    
#     print("\n[Verification] Sampled Input ID tails (0 = Padding):")
#     for i in range(BATCH_SIZE):
#         print(f" Seq {i}: {input_ids_tensor[i, -100:-80].tolist()}")
    
    
#     print("\n" + "="*50)
#     print("--- 2. Generating Attention Mask (Causal + Padding) ---")
#     print("="*50)
    
#     # Generate Causal + Padding Mask ([B, 1, S, S])
#     attn_mask_tensor = generator.generate_attention_mask(input_ids_tensor, is_causal=True)
    
#     print(f"[Verification] Mask Shape: {list(attn_mask_tensor.shape)}, Dtype: {attn_mask_tensor.dtype}")
    
    
#     print("\n" + "="*50)
#     print("--- 3. Generating Hidden States (THD - Flattened) ---")
#     print("---    Simulating MLP Input/Output Stages ---")
#     print("="*50)
    
#     # Standard dimension (H)
#     thd_h_model = generator.generate_thd_hidden_state(
#         input_ids=input_ids_tensor, 
#         target_ffn_dim=False
#     )
    
#     # FFN/MLP like D_ffn dimension (4H)
#     thd_h_ffn = generator.generate_thd_hidden_state(
#         input_ids=input_ids_tensor, 
#         target_ffn_dim=True
#     )
    
    
#     print("\n" + "="*50)
#     print("--- 4. Generating Hidden States ---")
#     print("="*50)

#     sbh_h = generator.generate_sbh_hidden_state(
#         micro_batch_size=BATCH_SIZE,
#         sequence_length=SEQ_LENGTH
#     )