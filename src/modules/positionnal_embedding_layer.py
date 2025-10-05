"""Positional Embedding Layer.

This module implements sinusoidal positional embeddings for the GPT-2 model.
"""

import torch
from torch import nn

from src.logger import logger


class PositionalEmbedding(nn.Module):
    """Sinusoidal positional embedding layer.

    Adds position information to token embeddings using the sinusoidal encoding
    from "Attention Is All You Need" paper:
    - PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    - PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Args:
        embedding_dimension: The dimension of the embedding vectors
        max_sequence_length: Maximum sequence length to support (default: 1024)

    """

    def __init__(
        self,
        embedding_dimension: int,
        max_sequence_length: int,
    ) -> None:
        """Initialize the positional embedding layer."""
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.max_sequence_length = max_sequence_length

        # Precompute positional embeddings for efficiency
        self.register_buffer("pos_embeddings", self._create_positional_embeddings())

    def _create_positional_embeddings(self) -> torch.Tensor:
        """Create sinusoidal positional embeddings.

        Returns:
            Positional embeddings of shape [max_sequence_length, embedding_dimension]

        """
        pos_embeddings = torch.zeros(self.max_sequence_length, self.embedding_dimension)

        # Create position indices [0, 1, 2, ..., max_sequence_length-1]
        position = torch.arange(
            0,
            self.max_sequence_length,
            dtype=torch.float,
        ).unsqueeze(1)

        # Create dimension indices [0, 2, 4, ..., embedding_dimension-2]
        div_term = torch.exp(
            torch.arange(0, self.embedding_dimension, 2, dtype=torch.float)
            * -(torch.log(torch.tensor(10000.0)) / self.embedding_dimension),
        )

        # Apply sin to even indices
        pos_embeddings[:, 0::2] = torch.sin(position * div_term)

        # Apply cos to odd indices
        pos_embeddings[:, 1::2] = torch.cos(position * div_term)

        return pos_embeddings

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional embeddings to input tensor.

        Args:
            x: Input tensor of shape [batch_size, sequence_length, embedding_dimension]

        Returns:
            Tensor with positional embeddings added, same shape as input

        """
        batch_size, sequence_length, embedding_dim = x.shape

        # Ensure sequence length doesn't exceed maximum
        if sequence_length > self.max_sequence_length:
            msg = (
                f"Sequence length {sequence_length} exceeds maximum supported "
                f"length {self.max_sequence_length}"
            )
            raise ValueError(msg)

        # Add positional embeddings to input
        # pos_embeddings shape: [sequence_length, embedding_dimension]
        # Broadcasting will handle the batch dimension
        logger.debug("Positional embedding input: %s", x.shape)
        x = x + self.pos_embeddings[:sequence_length, :]
        logger.debug("Positional embedding output: %s", x.shape)
        return x
