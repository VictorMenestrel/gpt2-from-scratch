"""Feed-Forward Network Module.

This module implements the position-wise feed-forward network for transformers.
"""

import torch
from torch import nn

from src.logger import logger


class FeedForward(nn.Module):
    """Position-wise feed-forward network for transformer.

    Implements the FFN with GELU activation:
    FFN(x) = GELU(xW1 + b1)W2 + b2

    Args:
        embedding_dimension: Size of input/output embeddings
        hidden_dimension_factor: Expansion factor for hidden layer (typically 4)

    """

    def __init__(
        self,
        embedding_dimension: int,
        hidden_dimension_factor: int = 4,
    ) -> None:
        """Initialize the feed-forward network."""
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.hidden_dimension_factor = hidden_dimension_factor
        self.linear1 = nn.Linear(
            embedding_dimension,
            embedding_dimension * hidden_dimension_factor,
        )
        self.linear2 = nn.Linear(
            embedding_dimension * hidden_dimension_factor,
            embedding_dimension,
        )
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply feed-forward transformation.

        Args:
            x: Input tensor of shape [batch_size, seq_len, embedding_dimension]

        Returns:
            Output tensor of same shape as input

        """
        logger.debug("Feed-forward input: %s", x.shape)
        x = self.linear1(x)
        logger.debug("Feed-forward after linear1: %s", x.shape)
        x = self.gelu(x)
        x = self.linear2(x)
        logger.debug("Feed-forward output: %s", x.shape)
        return x
