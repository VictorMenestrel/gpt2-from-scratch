"""Multi-Head Attention Module."""

import torch
from torch import nn

from src.logger import logger


class MultiHeadAttention(nn.Module):
    """Multi-head attention layer."""

    def __init__(
        self,
        num_heads: int,
        embedding_dimension: int,
        hidden_dimension: int,
        max_seq_len: int,
    ) -> None:
        """Initialize the multi-head attention layer."""
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.hidden_dimension = hidden_dimension
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.Wq = nn.Linear(embedding_dimension, hidden_dimension * num_heads)
        self.Wk = nn.Linear(embedding_dimension, hidden_dimension * num_heads)
        self.Wv = nn.Linear(embedding_dimension, hidden_dimension * num_heads)
        self.sqrt_dk = torch.sqrt(torch.tensor(hidden_dimension, dtype=torch.float))
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool(),
        )
        self.Wo = nn.Linear(hidden_dimension * num_heads, embedding_dimension)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for multi-head attention.

        Args:
            x: Input tensor of shape [batch_size, sequence_length, embedding_dimension]

        Returns:
            Tensor of shape [batch_size, sequence_length, embedding_dimension]

        """
        batch_size, sequence_length, _ = x.shape

        logger.debug("Multi-head attention input: %s", x.shape)

        Q: torch.Tensor = self.Wq(x)  # Shape [batch, seq, hidden_dimension * num_head]
        K: torch.Tensor = self.Wk(x)  # Shape [batch, seq, hidden_dimension * num_head]
        V: torch.Tensor = self.Wv(x)  # Shape [batch, seq, hidden_dimension * num_head]

        logger.debug("Q shape: %s", Q.shape)
        logger.debug("K shape: %s", K.shape)
        logger.debug("V shape: %s", V.shape)

        # Reshape Q, K, V for multi-head attention
        Q = Q.view(batch_size, sequence_length, self.num_heads, self.hidden_dimension)
        K = K.view(batch_size, sequence_length, self.num_heads, self.hidden_dimension)
        V = V.view(batch_size, sequence_length, self.num_heads, self.hidden_dimension)
        # Shape is [batch, seq, num_head, hidden_dimension]  # noqa: ERA001

        logger.debug("Reshaped Q shape: %s", Q.shape)
        logger.debug("Reshaped K shape: %s", K.shape)
        logger.debug("Reshaped V shape: %s", V.shape)

        # Reshape to get [batch, num_head, seq, hidden_dimension]
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        logger.debug("Transposed Q shape: %s", Q.shape)
        logger.debug("Transposed K shape: %s", K.shape)
        logger.debug("Transposed V shape: %s", V.shape)

        attention_scores = (
            Q @ K.transpose(-2, -1)
        ) / self.sqrt_dk  # Shape [batch, num_heads, seq, seq]

        logger.debug("Attention scores shape: %s", attention_scores.shape)

        # Create a mask to prevent attention to future tokens
        # mask shape [seq, seq]
        mask = self.causal_mask[:sequence_length, :sequence_length]

        # Shape [batch, num_heads, seq, seq]  # noqa: ERA001
        attention_scores = attention_scores.masked_fill(mask, -torch.inf)

        attention_weights = torch.softmax(
            attention_scores,
            dim=-1,
        )  # Shape [batch, num_heads, seq, seq]

        logger.debug("Attention weights shape: %s", attention_weights.shape)

        output = (
            (attention_weights @ V)  # Shape [batch, num_heads, seq, hidden_dimension]
            .transpose(2, 1)  # Shape [batch, seq, num_heads, hidden_dimension]
            .contiguous()  # Make tensor contiguous in memory
            .view(batch_size, sequence_length, self.hidden_dimension * self.num_heads)
        )  # Shape [batch, seq, hidden_dimension * num_heads]

        logger.debug("Multi-head attention output before Wo: %s", output.shape)

        output = self.Wo(output)  # Shape [batch, seq, embedding_dimension]
        logger.debug("Multi-head attention output after Wo: %s", output.shape)

        return output
