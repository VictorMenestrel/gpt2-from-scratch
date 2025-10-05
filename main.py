"""Entry point for the GPT-2 from scratch project."""

import torch

from src.modules.gpt2_model import GPT2Model


def main():
    print("Hello from gpt-2-from-scratch!")

    # Create GPT-2 model (XL configuration)
    model = GPT2Model(
        vocab_size=100,  # GPT-2 vocabulary size
        embedding_dimension=16,  # Model dimension
        num_heads=2,  # Number of attention heads
        hidden_dimension=4,  # Dimension per attention head
        num_blocks=4,  # Number of transformer blocks
        max_seq_len=1024,  # Maximum sequence length
        hidden_dimension_factor=4,  # FFN expansion factor
        dropout_rate=0.1,  # Dropout rate
    )

    # Forward pass
    input_ids = torch.randint(0, 100, (1, 10))  # [batch_size, seq_len]
    logits = model(input_ids)  # [batch_size, seq_len, vocab_size]
    print("Logits shape:", logits.shape)  # Should be [1, 10, 100]

    # Generate text
    generated_tokens = model.generate(
        input_ids=input_ids,
        max_new_tokens=100,
        temperature=0.8,
    )
    print("Generated tokens shape:", generated_tokens.shape)  # Should be [1, 110]
    print("Generated tokens:", generated_tokens)


if __name__ == "__main__":
    main()
