import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformer_blocks import TransformerBlock

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pe[:seq_len, :].unsqueeze(0)
        return x

class TinyGPT(nn.Module):
    def __init__(self, vocab_size, d_model=128, num_heads=4, num_layers=2, max_seq_len=256, dropout=0.1):
        super().__init__()
        self.max_seq_len = max_seq_len
        
        # --- [KOMENTAR EDUKATIF: EMBEDDING] ---
        # Token Embedding: Mengubah ID angka dari tokenizer menjadi vektor numerik padat (dense vector).
        # Ini memberikan "makna/identitas" pada setiap token (kata/karakter).
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional Encoding: Berbeda dengan RNN, Transformer memproses semua kata sekaligus.
        # Oleh karena itu, kita harus "menyuntikkan" informasi urutan/posisi kata ke dalam model.
        self.positional_encoding = PositionalEncoding(d_model, max_len=max_seq_len)
        
        # Tumpukan (Stack) dari Transformer Blocks
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, dropout) for _ in range(num_layers)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        
        # Language Modeling Head: Layer linear terakhir untuk memprediksi probabilitas kata berikutnya.
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Weight tying: Menggunakan bobot yang sama untuk Embedding dan LM Head untuk menghemat memori.
        self.lm_head.weight = self.token_embedding.weight
        
    def generate_square_subsequent_mask(self, sz):
        # Masking segitiga bawah (lower triangular)
        # Angka 1 pada matriks berarti token boleh dilihat, 0 berarti token masa depan (harus disembunyikan)
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        return mask == 0

    def forward(self, idx):
        batch_size, seq_len = idx.size()
        
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} is greater than maximum sequence length {self.max_seq_len}")
            
        # Menggabungkan Token Embedding dan Positional Encoding
        x = self.token_embedding(idx)
        x = self.positional_encoding(x)
        
        # Causal Mask (look-ahead mask)
        mask = self.generate_square_subsequent_mask(seq_len).to(x.device)
        
        # Forward pass through Transformer layers
        for layer in self.layers:
            x = layer(x, mask)
            
        x = self.ln_f(x)
        
        # Produce logits for the vocabulary
        logits = self.lm_head(x)
        
        return logits

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.max_seq_len:]
            logits = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, 1)
            idx = torch.cat((idx, next_idx), dim=1)
        return idx
