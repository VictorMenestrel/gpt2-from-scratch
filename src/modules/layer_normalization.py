"""Layer Normalization Module.

This module implements layer normalization.
"""

import torch
from torch import nn

from src.logger import logger


class LayerNormalization(nn.Module):
    """Layer normalization."""

    def __init__(self, embedding_dimension: int, epsilon: float = 1e-5) -> None:
        """Initialize the layer normalization module."""
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.epsilon = epsilon
        self.gamma = nn.Parameter(torch.ones(embedding_dimension))
        self.beta = nn.Parameter(torch.zeros(embedding_dimension))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply layer normalization.

        Args:
            x: Input tensor of shape [batch_size, sequence_length, embedding_dimension]

        Returns:
            Normalized tensor of same shape as input

        """
        logger.debug("Layer normalization input: %s", x.shape)

        variance, mean = torch.var_mean(x, dim=-1, keepdim=True, unbiased=False)
        # variance & mean of shape [batch_size, sequence_length, 1]

        x = torch.divide(
            torch.subtract(x, mean),
            torch.sqrt(torch.add(variance, self.epsilon)),
        )

        x = torch.add(torch.multiply(self.gamma, x), self.beta)

        logger.debug("Layer normalization output: %s", x.shape)
        return x
