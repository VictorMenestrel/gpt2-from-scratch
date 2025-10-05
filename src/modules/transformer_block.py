"""Transformer Block Module.

This module implements the Transformer block used in the GPT-2 architecture.
"""

import torch
from torch import nn

from src.modules.feedforward import FeedForward
from src.modules.layer_normalization import LayerNormalization
from src.modules.multi_head_attention import MultiHeadAttention


class TransformerBlock(nn.Module):
    """Single transformer block with multi-head attention and feed-forward layers.

    Implements the transformer block architecture with pre-layer normalization:
    1. LayerNorm -> MultiHeadAttention -> Dropout -> Residual
    2. LayerNorm -> FeedForward -> Dropout -> Residual
    """

    def __init__(
        self,
        num_heads: int,
        embedding_dimension: int,
        hidden_dimension: int,
        max_seq_len: int,
        hidden_dimension_factor: int = 4,
        dropout_rate: float = 0.1,
    ):
        """Initialize transformer block.

        Args:
            num_heads: Number of attention heads
            embedding_dimension: Size of embeddings
            hidden_dimension: Dimension for attention projections
            max_seq_len: Maximum sequence length
            hidden_dimension_factor: Expansion factor for feedforward (default: 4)
            dropout_rate: Dropout probability (default: 0.1)

        """
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.hidden_dimension = hidden_dimension
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len

        self.layernorm1 = LayerNormalization(embedding_dimension)
        self.layernorm2 = LayerNormalization(embedding_dimension)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.dropout2 = nn.Dropout(p=dropout_rate)
        self.feedforward = FeedForward(embedding_dimension, hidden_dimension_factor)
        self.multiheadattention = MultiHeadAttention(
            num_heads,
            embedding_dimension,
            hidden_dimension,
            max_seq_len,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the transformer block.

        Args:
            x: Input tensor of shape [batch_size, seq_len, embedding_dimension]

        Returns:
            Output tensor of same shape as input

        """
        # First sub-layer: Multi-head attention with residual connection
        res_x = x
        x = self.layernorm1(x)
        x = self.multiheadattention(x)
        x = self.dropout1(x)
        x += res_x

        # Second sub-layer: Feed-forward with residual connection
        res_x = x
        x = self.layernorm2(x)
        x = self.feedforward(x)
        x = self.dropout2(x)
        x += res_x

        return x
