"""
diagnostic.py  -  Step-by-step reference values for engine.c debugging.
Run this after export_weights.py and compare its output to engine.exe debug prints.
"""
import sys
import io
import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

MODEL_NAME = "microsoft/DialoGPT-medium"
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
model.eval()

# Same hyperparams as engine.c
SEQ_LEN = 64
DIM = 1024
DFF = 4096
N_HEADS = 16
DK = DIM // N_HEADS

# Tokens for "Hello world"
# Using input_ids directly to match C's naive tokens without special chars
input_ids = torch.tensor([[15496, 995]])
print(f"Tokens: {tokenizer.convert_ids_to_tokens(input_ids[0])}")

with torch.no_grad():
    x = model.transformer.wte(input_ids) + model.transformer.wpe(torch.arange(0, input_ids.size(1)).unsqueeze(0))
    
    print(f"STAGE 1 - After embedding, token 0 [:4]:")
    print(f"  {x[0, 0, :4].numpy()}")

    # Pass through layer 0
    layer = model.transformer.h[0]
    
    x_norm = layer.ln_1(x)
    print(f"STAGE 2 - Layer 0 after LN1, token 0 [:4]:")
    print(f"  {x_norm[0, 0, :4].numpy()}")

    # Attention
    attn_out = layer.attn(x_norm)[0]
    print(f"STAGE 2 - Layer 0 after Attention, token 0 [:4]:")
    print(f"  {attn_out[0, 0, :4].numpy()}")

    x = x + attn_out

    # MLP
    x_norm_2 = layer.ln_2(x)
    mlp_out = layer.mlp(x_norm_2)
    print(f"STAGE 2 - Layer 0 after MLP, token 0 [:4]:")
    print(f"  {mlp_out[0, 0, :4].numpy()}")

    x = x + mlp_out

    print(f"STAGE 2 - Layer 0 Final, token 0 [:4]:")
    print(f"  {x[0, 0, :4].numpy()}")

    # Pass through all layers
    out = model(input_ids)
    final_x = out.logits
    print(f"\nSTAGE 3 - Final logits for last token, max value and token ID:")
    val, idx = torch.max(final_x[0, -1, :], dim=0)
    print(f"  ID: {idx.item()} Max Logit: {val.item():.4f}")