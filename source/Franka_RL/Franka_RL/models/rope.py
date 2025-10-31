"""
Rotary Position Embedding (RoPE) for Transformer models.

RoPE encodes position information by rotating query and key vectors,
enabling better relative position modeling for long sequences.

Paper: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
https://arxiv.org/abs/2104.09864
"""

import torch
import torch.nn as nn
import math
from typing import Tuple, Optional


class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) for Transformer models.
    
    RoPE encodes position information by rotating query and key vectors,
    enabling better relative position modeling for long sequences.
    
    Args:
        dim (int): Dimension of the embeddings (should be even)
        max_seq_len (int): Maximum sequence length to pre-compute
        theta (float): Base value for frequency calculation (default: 10000.0)
        device (str): Device to place tensors on
    """
    
    def __init__(
        self,
        dim: int,
        max_seq_len: int = 100,
        theta: float = 10000.0,
        device: str = "cpu"
    ):
        super().__init__()
        
        assert dim % 2 == 0, f"Embedding dimension must be even, got {dim}"
        
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        
        # Pre-compute frequency matrix
        # freq = 1 / (theta ^ (2i/d)) for i in [0, d/2)
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Pre-compute rotation matrices for max_seq_len positions
        self._build_cache(max_seq_len, device)
    
    def _build_cache(self, seq_len: int, device: str = None):
        """Pre-compute cos and sin values for all positions."""
        if device is None:
            device = self.inv_freq.device
            
        # Create position indices: [0, 1, 2, ..., seq_len-1]
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        
        # Compute frequencies: [seq_len, dim/2]
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        
        # Create [seq_len, dim] by duplicating each frequency
        # This matches the structure of query/key vectors
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # Pre-compute cos and sin
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
        self.max_seq_len_cached = seq_len
    
    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """
        Rotate half the dimensions of x.
        
        For vector [x1, x2, x3, x4, ...], return [-x2, x1, -x4, x3, ...]
        This is used in the rotation operation.
        """
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    
    def apply_rotary_pos_emb(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        seq_dim: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary position embedding to query and key tensors.
        
        Args:
            q: Query tensor, shape [..., seq_len, dim]
            k: Key tensor, shape [..., seq_len, dim]
            seq_dim: Dimension index for sequence length
            
        Returns:
            Rotated query and key tensors
        """
        seq_len = q.shape[seq_dim]
        
        # Expand cache if needed
        if seq_len > self.max_seq_len_cached:
            self._build_cache(seq_len, device=q.device)
        
        # Get cos and sin for current sequence length
        cos = self.cos_cached[:seq_len]
        sin = self.sin_cached[:seq_len]
        
        # Reshape for broadcasting
        # Original shape: [seq_len, dim]
        # Need shape compatible with [..., seq_len, dim]
        if seq_dim == 1:  # Most common: [batch, seq, dim]
            cos = cos.unsqueeze(0)  # [1, seq_len, dim]
            sin = sin.unsqueeze(0)
        elif seq_dim == 0:  # For [seq, batch, dim]
            cos = cos.unsqueeze(1)  # [seq_len, 1, dim]
            sin = sin.unsqueeze(1)
        
        # Apply rotation: x * cos + rotate_half(x) * sin
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        
        return q_embed, k_embed
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        seq_dim: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass - apply RoPE to query and key.
        
        Args:
            q: Query tensor
            k: Key tensor
            seq_dim: Which dimension is the sequence length
            
        Returns:
            Position-encoded query and key tensors
        """
        return self.apply_rotary_pos_emb(q, k, seq_dim=seq_dim)


class RoPETransformerEncoderLayer(nn.TransformerEncoderLayer):
    """
    TransformerEncoderLayer with RoPE instead of standard positional encoding.
    
    This layer replaces the attention mechanism to apply RoPE before
    computing attention scores.
    """
    
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        rope_theta: float = 10000.0,
        max_seq_len: int = 2048,
        **kwargs
    ):
        super().__init__(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            **kwargs
        )
        
        # Initialize RoPE for each attention head
        head_dim = d_model // nhead
        self.rope = RotaryPositionEmbedding(
            dim=head_dim,
            max_seq_len=max_seq_len,
            theta=rope_theta,
        )
        self.nhead = nhead
        self.head_dim = head_dim
    
    def _sa_block(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
        is_causal: bool = False
    ) -> torch.Tensor:
        """
        Self-attention block with RoPE.
        
        Overrides the parent class method to apply RoPE before attention.
        """
        # Get Q, K, V projections
        # x shape: [seq_len, batch_size, d_model]
        seq_len, batch_size, d_model = x.shape
        
        # Project to Q, K, V using the self_attn module
        # We need to manually compute Q, K, V to apply RoPE
        q = self.self_attn.in_proj_weight[:d_model] @ x.reshape(-1, d_model).T
        k = self.self_attn.in_proj_weight[d_model:2*d_model] @ x.reshape(-1, d_model).T
        v = self.self_attn.in_proj_weight[2*d_model:] @ x.reshape(-1, d_model).T
        
        if self.self_attn.in_proj_bias is not None:
            q = q + self.self_attn.in_proj_bias[:d_model].unsqueeze(1)
            k = k + self.self_attn.in_proj_bias[d_model:2*d_model].unsqueeze(1)
            v = v + self.self_attn.in_proj_bias[2*d_model:].unsqueeze(1)
        
        # Reshape: [d_model, batch*seq] -> [batch, seq, d_model]
        q = q.T.reshape(batch_size, seq_len, d_model)
        k = k.T.reshape(batch_size, seq_len, d_model)
        v = v.T.reshape(batch_size, seq_len, d_model)
        
        # Reshape for multi-head: [batch, seq, nhead, head_dim]
        q = q.reshape(batch_size, seq_len, self.nhead, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.nhead, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.nhead, self.head_dim)
        
        # Apply RoPE to each head
        q_rot_list = []
        k_rot_list = []
        for head_idx in range(self.nhead):
            q_head = q[:, :, head_idx, :]  # [batch, seq, head_dim]
            k_head = k[:, :, head_idx, :]
            
            q_rot, k_rot = self.rope(q_head, k_head, seq_dim=1)
            
            q_rot_list.append(q_rot)
            k_rot_list.append(k_rot)
        
        # Recombine heads: [batch, seq, nhead, head_dim]
        q = torch.stack(q_rot_list, dim=2)
        k = torch.stack(k_rot_list, dim=2)
        
        # Reshape to [batch*nhead, seq, head_dim]
        q = q.reshape(batch_size * self.nhead, seq_len, self.head_dim)
        k = k.reshape(batch_size * self.nhead, seq_len, self.head_dim)
        v = v.reshape(batch_size * self.nhead, seq_len, self.head_dim)
        
        # Compute scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn_weights = torch.bmm(q, k.transpose(1, 2)) * scale
        
        # Apply masks
        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask
        if key_padding_mask is not None:
            # key_padding_mask: [batch, seq]
            # Need to expand to [batch*nhead, seq, seq]
            expanded_mask = key_padding_mask.unsqueeze(1).repeat_interleave(self.nhead, dim=0)
            attn_weights = attn_weights.masked_fill(
                expanded_mask.unsqueeze(1),
                float('-inf')
            )
        
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout1(attn_weights)
        
        # Apply attention to values
        attn_output = torch.bmm(attn_weights, v)
        
        # Reshape and project output
        attn_output = attn_output.reshape(batch_size, seq_len, d_model)
        attn_output = attn_output.transpose(0, 1)  # [seq, batch, d_model]
        attn_output = self.self_attn.out_proj(attn_output)
        
        return attn_output


def create_rope_transformer_encoder(
    d_model: int,
    nhead: int,
    num_layers: int,
    dim_feedforward: int = 2048,
    dropout: float = 0.1,
    activation: str = "relu",
    rope_theta: float = 10000.0,
    max_seq_len: int = 2048,
) -> nn.TransformerEncoder:
    """
    Factory function to create a TransformerEncoder with RoPE.
    
    Args:
        d_model: Dimension of the model
        nhead: Number of attention heads
        num_layers: Number of encoder layers
        dim_feedforward: Dimension of feedforward network
        dropout: Dropout rate
        activation: Activation function name
        rope_theta: RoPE theta parameter
        max_seq_len: Maximum sequence length
        
    Returns:
        TransformerEncoder with RoPE-enabled layers
    """
    encoder_layer = RoPETransformerEncoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        activation=activation,
        rope_theta=rope_theta,
        max_seq_len=max_seq_len,
    )
    
    return nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
