"""Token Embedding Layer.

This module implements the token embedding layer for the GPT-2 model.
"""

import torch
from torch import nn

from src.logger import logger


class TokenEmbedding(nn.Module):
    """Token embedding layer that converts token indices to dense embeddings.

    Args:
        embedding_dimension: The dimension of the embedding vectors
        vocabulary_size: The size of the vocabulary

    """

    def __init__(self, embedding_dimension: int, vocabulary_size: int) -> None:
        """Initialize the token embedding layer."""
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.vocabulary_size = vocabulary_size
        self.embedding = nn.Embedding(
            num_embeddings=vocabulary_size,
            embedding_dim=embedding_dimension,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the token embedding layer.

        Args:
            x: Token indices of shape [batch_size, sequence_length]

        Returns:
            Token embeddings of shape [batch_size, sequence_length, embedding_dimension]

        """
        logger.debug("Token embedding input: %s", x.shape)
        x = self.embedding(x)
        logger.debug("Token embedding output: %s", x.shape)
        return x
