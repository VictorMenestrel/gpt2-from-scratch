"""Attention Layer Module."""

import torch
from torch import nn


class SelfAttention(nn.Module):
    """Self-attention layer."""

    def __init__(self, embedding_dimension: int, hidden_dimension: int) -> None:
        """Initialize the self-attention layer."""
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.hidden_dimension = hidden_dimension
        self.Wq = nn.Linear(embedding_dimension, hidden_dimension)
        self.Wk = nn.Linear(embedding_dimension, hidden_dimension)
        self.Wv = nn.Linear(embedding_dimension, hidden_dimension)
        self.sqrt_dk = torch.sqrt(torch.tensor(hidden_dimension, dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for self-attention.

        Args:
            x (torch.Tensor): Input tensor of shape [batch, seq, embedding_dimension].

        Returns:
            torch.Tensor: Output tensor of shape [batch, seq, hidden_dimension].

        """
        batch_size, sequence_length, _ = x.shape

        Q: torch.Tensor = self.Wq(x)  # Shape [batch, seq, hidden_dimension]
        K: torch.Tensor = self.Wk(x)  # Shape [batch, seq, hidden_dimension]
        V: torch.Tensor = self.Wv(x)  # Shape [batch, seq, hidden_dimension]

        attention_scores = (
            Q @ K.transpose(-2, -1)
        ) / self.sqrt_dk  # size [batch, seq, seq]

        # Create a mask to prevent attention to future tokens
        # mask shape [seq, seq]
        mask = torch.triu(
            torch.ones(sequence_length, sequence_length),
            diagonal=1,
        ).bool()

        # Shape [batch, seq, seq]
        attention_scores = attention_scores.masked_fill(mask, -torch.inf)

        attention_weights = torch.softmax(
            attention_scores,
            dim=-1,
        )  # Shape [batch, seq, seq]

        return attention_weights @ V  # Shape [batch, seq, hidden_dimension]
