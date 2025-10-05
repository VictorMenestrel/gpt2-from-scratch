"""GPT-2 Model Module.

This module implements the complete GPT-2 model architecture.
"""

import torch
from torch import nn

from src.logger import logger
from src.modules.layer_normalization import LayerNormalization
from src.modules.positionnal_embedding_layer import PositionalEmbedding
from src.modules.token_embedding_layer import TokenEmbedding
from src.modules.transformer_block import TransformerBlock


class GPT2Model(nn.Module):
    """Complete GPT-2 model with token embeddings, positional embeddings, and transformer blocks.

    Architecture:
    1. Token Embedding + Positional Embedding
    2. Stack of N Transformer Blocks
    3. Final Layer Normalization
    4. Language Modeling Head (optional)
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dimension: int,
        num_heads: int,
        hidden_dimension: int,
        num_blocks: int = 48,  # GPT-2 Large has 48 blocks
        max_seq_len: int = 1024,
        hidden_dimension_factor: int = 4,
        dropout_rate: float = 0.1,
    ) -> None:
        """Initialize GPT-2 model.

        Args:
            vocab_size: Size of vocabulary
            embedding_dimension: Size of embeddings (768 for GPT-2 base, 1600 for large)
            num_heads: Number of attention heads
            hidden_dimension: Dimension for attention projections
            num_blocks: Number of transformer blocks (default: 48)
            max_seq_len: Maximum sequence length (default: 1024)
            hidden_dimension_factor: Expansion factor for feedforward (default: 4)
            dropout_rate: Dropout probability (default: 0.1)

        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dimension = embedding_dimension
        self.num_heads = num_heads
        self.hidden_dimension = hidden_dimension
        self.num_blocks = num_blocks
        self.max_seq_len = max_seq_len
        self.hidden_dimension_factor = hidden_dimension_factor
        self.dropout_rate = dropout_rate

        self.token_embeddings = TokenEmbedding(embedding_dimension, vocab_size)
        self.positional_embeddings = PositionalEmbedding(
            embedding_dimension,
            max_seq_len,
        )
        self.dropout = nn.Dropout(p=dropout_rate)
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    num_heads,
                    embedding_dimension,
                    hidden_dimension,
                    max_seq_len,
                    hidden_dimension_factor,
                    dropout_rate,
                )
                for _ in range(num_blocks)
            ],
        )
        self.layernorm = LayerNormalization(embedding_dimension)
        self.lm_head = nn.Linear(embedding_dimension, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass through the GPT-2 model.

        Args:
            input_ids: Token indices of shape [batch_size, seq_len]

        Returns:
            Logits of shape [batch_size, seq_len, vocab_size]

        """
        x = self.token_embeddings(input_ids)  # Shape [batch, seq, emb_dim]
        x = self.positional_embeddings(x)  # Add positional embeddings
        x = self.dropout(x)
        for block in self.transformer_blocks:
            x = block(x)  # Shape [batch, seq, emb_dim]
        x = self.layernorm(x)  # Final layer normalization
        logger.debug("GPT-2 output before lm_head: %s", x.shape)
        logits = self.lm_head(x)  # Shape [batch, seq, vocab_size]
        logger.debug("GPT-2 logits output: %s", logits.shape)

        return logits

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Generate text using the model.

        Args:
            input_ids: Starting tokens [batch_size, seq_len]
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)

        Returns:
            Generated token sequence [batch_size, seq_len + max_new_tokens]

        """
        self.eval()

        for _ in range(max_new_tokens):
            # Forward pass
            with torch.no_grad():
                # TODO : this line seems wrong
                logits = self(input_ids)  # [batch, seq, vocab]

                # Get logits for last token and apply temperature
                last_logits = logits[:, -1, :] / temperature  # [batch, vocab]

                # Sample next token
                probs = torch.softmax(last_logits, dim=-1)  # [batch, vocab]
                next_token = torch.multinomial(probs, num_samples=1)  # [batch, 1]

                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=1)

                # Stop if we exceed max sequence length
                if input_ids.size(1) >= self.max_seq_len:
                    break

        return input_ids
