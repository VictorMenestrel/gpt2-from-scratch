# GPT-2 from Scratch

A complete implementation of GPT-2 (Generative Pre-trained Transformer 2) built from scratch using PyTorch. This project implements all core components of the transformer architecture without relying on pre-built transformer libraries. This implementation helped me deeply understand the inner workings of transformers and the GPT-2 model. This code is intended for educational purposes and is not optimized for production use.

## 🚀 Features

- **Complete GPT-2 Architecture**: Token embeddings, positional embeddings, multi-head attention, feedforward networks, and layer normalization
- **Modular Design**: Each component is implemented as a separate, reusable module
- **Configurable Model Sizes**: Support for GPT-2 Small, Medium, Large, and XL configurations
- **Text Generation**: Built-in text generation with temperature-controlled sampling

## 📋 Architecture Overview

The implementation follows the original GPT-2 paper architecture:

```
Input Tokens
    ↓
Token Embedding + Positional Embedding
    ↓
Dropout
    ↓
Stack of N Transformer Blocks
    ├── Layer Normalization
    ├── Multi-Head Self-Attention
    ├── Dropout + Residual Connection
    ├── Layer Normalization
    ├── Feed-Forward Network
    └── Dropout + Residual Connection
    ↓
Final Layer Normalization
    ↓
Language Modeling Head
    ↓
Output Logits
```

## 🏗️ Project Structure

```
src/
├── modules/
│   ├── attention_layer.py          # Self-attention mechanism
│   ├── multi_head_attention.py     # Multi-head attention
│   ├── feedforward.py              # Position-wise feed-forward network
│   ├── layer_normalization.py     # Layer normalization
│   ├── token_embedding_layer.py    # Token embeddings
│   ├── poitionnal_embedding_layer.py # Positional embeddings
│   ├── transformer_block.py        # Complete transformer block
│   └── gpt2_model.py              # Complete GPT-2 model
└── main.py                         # Entry point
```

## 📦 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/VictorMenestrel/gpt2-from-scratch.git
   cd gpt-2-from-scratch
   ```

2. **Install dependencies**:

    using uv:
   ```bash
   uv sync
   ```

## 🔧 Usage

### Basic Model Creation

```python
from src.modules.gpt2_model import GPT2Model
import torch

# Create GPT-2 model (XL configuration)
model = GPT2Model(
    vocab_size=50257,           # GPT-2 vocabulary size
    embedding_dimension=1600,   # Model dimension
    num_heads=25,              # Number of attention heads
    hidden_dimension=64,       # Dimension per attention head
    num_blocks=48,             # Number of transformer blocks
    max_seq_len=1024,          # Maximum sequence length
    hidden_dimension_factor=4, # FFN expansion factor
    dropout_rate=0.1           # Dropout rate
)

# Forward pass
input_ids = torch.randint(0, 50257, (1, 10))  # [batch_size, seq_len]
logits = model(input_ids)  # [batch_size, seq_len, vocab_size]
```

### Text Generation

```python
# Generate text
generated_tokens = model.generate(
    input_ids=input_ids,
    max_new_tokens=100,
    temperature=0.8
)
```

### Model Configurations

| Model | Parameters | Layers | Dimension | Heads | Head Dim |
|-------|------------|--------|-----------|-------|----------|
| Small | 117M       | 12     | 768       | 12    | 64       |
| Medium| 345M       | 24     | 1024      | 16    | 64       |
| Large | 762M       | 36     | 1280      | 20    | 64       |
| XL    | 1.5B       | 48     | 1600      | 25    | 64       |

## 🧩 Core Components

### Multi-Head Attention
- Implements scaled dot-product attention
- Parallel processing of multiple attention heads
- Causal masking for autoregressive generation

### Feed-Forward Network
- Position-wise fully connected layers
- GELU activation function
- Configurable expansion factor

### Layer Normalization
- Pre-normalization architecture
- Learnable scale and shift parameters
- Numerical stability with epsilon

### Positional Embeddings
- Sinusoidal positional encoding
- Supports sequences up to maximum length
- Efficient pre-computed embeddings

## 🎯 Key Features

- **From Scratch Implementation**: No dependency on pre-built transformer libraries
- **Educational Focus**: for learning purposes
- **Modular Architecture**: Easy to understand and modify individual components

## 📚 References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer paper
- [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) - GPT-2 paper
- [Vizuara Build Deep Seek from scratch](https://www.youtube.com/playlist?list=PLPTV0NXA_ZSiOpKKlHCyOq9lnp-dLvlms) - An amazing YouTube series that helps in understanding current LLMs.

## 🛠️ Development

This project uses modern Python development practices:

- **Type Hints**: Full type annotation coverage
- **Documentation**: Comprehensive docstrings for all modules
- **Code Quality**: Follows PEP 8 style guidelines, using Ruff for linting
- **Modular Design**: Separation of concerns with clear interfaces

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## ⭐ Acknowledgments

This implementation was built as an educational project to understand the transformer architecture from first principles. Special thanks to the original authors of the GPT-2 paper and the broader research community.

---

**Note**: This is an educational implementation. For production use cases, consider using established libraries like Hugging Face Transformers.