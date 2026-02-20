import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import json

# --- Configuration ---
MODEL_NAME = "microsoft/DialoGPT-medium" 
OUT_FILE   = "yourpath/tiny_inference_c/gpt2_tiny.bin"
VOCAB_FILE = "vocab.txt"
TEXT       = "Hello world"

# --- Load Model ---
print(f"Loading {MODEL_NAME}...")
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
model.eval()

# --- Helper Function ---
def export_tensor(tensor, f):
    arr = tensor.detach().cpu().numpy().astype(np.float32)
    arr.tofile(f)

# --- Step 1: Export Binary Weights ---
print(f"Exporting weights to {OUT_FILE}...")

with open(OUT_FILE, "wb") as f:
    
    # 1. Embeddings
    export_tensor(model.transformer.wte.weight, f)
    export_tensor(model.transformer.wpe.weight, f)

    # 2. Layers
    for i, layer in enumerate(model.transformer.h):
        print(f"Processing Layer {i}...")
        
        # --- Attention Weights ---
        c_attn_w = layer.attn.c_attn.weight.detach().cpu().numpy()
        c_attn_b = layer.attn.c_attn.bias.detach().cpu().numpy()
        
        q_w, k_w, v_w = np.split(c_attn_w, 3, axis=1)
        q_b, k_b, v_b = np.split(c_attn_b, 3, axis=0)
        
        q_w.astype(np.float32).tofile(f); q_b.astype(np.float32).tofile(f)
        k_w.astype(np.float32).tofile(f); k_b.astype(np.float32).tofile(f)
        v_w.astype(np.float32).tofile(f); v_b.astype(np.float32).tofile(f)
        
        # Output Projection 
        export_tensor(layer.attn.c_proj.weight, f)
        export_tensor(layer.attn.c_proj.bias, f)
        
        # --- Layer Norm 1 ---
        export_tensor(layer.ln_1.weight, f)
        export_tensor(layer.ln_1.bias, f)
        
        # --- MLP Weights ---
        export_tensor(layer.mlp.c_fc.weight, f)   # w_ff1
        export_tensor(layer.mlp.c_fc.bias, f)     # b_ff1
        export_tensor(layer.mlp.c_proj.weight, f) # w_ff2
        export_tensor(layer.mlp.c_proj.bias, f)   # b_ff2
        
        # --- Layer Norm 2 (Before MLP) ---
        export_tensor(layer.ln_2.weight, f)
        export_tensor(layer.ln_2.bias, f)

    # 3. Final Layer Norm
    print("Processing Final Layer Norm...")
    export_tensor(model.transformer.ln_f.weight, f)
    export_tensor(model.transformer.ln_f.bias, f)

print("Weights exported successfully.")

# --- Step 2: Export Vocabulary Text ---
print(f"Exporting vocabulary to {VOCAB_FILE}...")

vocab_map = tokenizer.get_vocab()

sorted_vocab = [None] * len(vocab_map)
for token, idx in vocab_map.items():
    sorted_vocab[idx] = token

with open(VOCAB_FILE, "w", encoding="utf-8") as f:
    for token in sorted_vocab:
        if token is None:
            f.write("<unk>\n")
            continue
        s = token.replace('Ġ', ' ')
        s = s.replace('\n', '\\n')
        f.write(s + "\n")

print(f"Vocabulary exported ({len(sorted_vocab)} tokens).")

tokenizer.pad_token = tokenizer.eos_token
inputs = tokenizer(TEXT, return_tensors="pt")
ids = inputs["input_ids"][0].tolist()

print("\n" + "="*40)
print("SETUP INSTRUCTIONS:")
print("1. Compile your C code.")
print("2. Ensure 'gpt2_tiny.bin' and 'vocab.txt' are in the same folder as the executable.")
print("3. Copy these IDs into the input_ids array in main() for a quick test:")
print("{" + ", ".join(map(str, ids)) + "}")
print("="*40)

# Verification Logic
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits[0, -1, :] # Last token logits
    val, idx = torch.max(logits, dim=0)
    
    print(f"\nVerification Check (Last Token):")
    print(f"PyTorch Predicted ID: {idx.item()}")
    print(f"PyTorch Predicted Word: '{tokenizer.decode([idx.item()])}'")
    print(f"PyTorch Max Logit:    {val.item():.4f}")
