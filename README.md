# Tiny Inference C

A lightweight, dependency-free GPT-2 inference engine written in **pure C** - no PyTorch, no CUDA, no external math libraries. This project re-implements the Transformer decoder architecture from scratch to explore the low-level mechanics of Large Language Models.

---

## What It Does

- Runs **autoregressive text generation** using GPT-2 / DialoGPT-Medium weights loaded from a custom binary format.
- Implements the full Transformer decoder stack:
  - Token + Positional Embeddings
  - Multi-Head Self-Attention with causal masking
  - Layer Normalization
  - Feed-Forward Network with GELU activation
  - Residual connections
  - Tied embedding (weight sharing for the output head)
- Includes a naive space-split tokenizer with GPT-2 vocabulary lookup.

---

## Project Structure

```
tiny_inference_c/
├── engine.c            # Core inference engine (pure C)
├── export_weights.py   # Exports HuggingFace model weights to binary format
└── README.md
```


---

## Setup & Usage

### 1. Install Python dependencies

```bash
pip install torch transformers numpy
```

### 2. Export the model weights

This downloads `microsoft/DialoGPT-medium` from HuggingFace and serializes its weights into a binary file:

```bash
python export_weights.py
```

This produces:
- `gpt2_tiny.bin` — raw float32 weights
- `vocab.txt` — token vocabulary, one token per line

### 3. Compile the C engine

```bash
gcc engine.c -o engine -lm -O2
```

### 4. Run

```bash
./engine
```

You'll be prompted to enter text. The model will generate the next 10 tokens autoregressively.


---

## Hyperparameters (DialoGPT-Medium / GPT-2 Medium)

| Parameter | Value |
|-----------|-------|
| Vocabulary size | 50,257 |
| Embedding dim | 1,024 |
| Layers | 24 |
| Attention heads | 16 |
| Max sequence length | 1,024 |

These are hardcoded in `engine.c`. To run a different model size, update the `#define` values at the top of the file and re-run `export_weights.py`.

---

## Purpose

This is an educational project. The goal is to understand what happens *inside* a Transformer at the lowest level possible , no abstractions, no automatic differentiation, just raw C and math.

Inspired by [llm.c](https://github.com/karpathy/llm.c) by Andrej Karpathy.

---

## License

MIT

