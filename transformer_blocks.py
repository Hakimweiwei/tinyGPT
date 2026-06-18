import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear layers for Query, Key, Value
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()
        
        # --- [KOMENTAR EDUKATIF: ATTENTION MECHANISM] ---
        # 1. Transformasi input 'x' menjadi Query (Q), Key (K), dan Value (V).
        # Q: "Apa yang saya cari?" | K: "Apa yang saya miliki?" | V: "Isi informasi saya"
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        
        # Transpose agar dimensi num_heads sejajar untuk perkalian matriks yang efisien
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # 2. Menghitung Attention Scores (Seberapa relevan satu kata dengan kata lainnya?)
        # Rumus: (Q dikali K_Transpose) dibagi dengan akar kuadrat dari d_k (Scaled Dot-Product)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        
        if mask is not None:
            # Masking: Menyembunyikan kata-kata masa depan (autoregressive) agar model tidak "mencontek"
            # Bagian matriks yang bernilai 0 pada mask akan diubah menjadi minus tak terhingga (-inf).
            mask = mask.unsqueeze(0).unsqueeze(1) # shape: (1, 1, seq_len, seq_len)
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        # Softmax: Mengubah skor menjadi probabilitas (bobot yang jumlahnya 1.0)
        attention_weights = F.softmax(scores, dim=-1)
        
        # 3. Mengalikan bobot (attention_weights) dengan Value (V) untuk mendapatkan representasi akhir
        out = torch.matmul(attention_weights, V)
        
        # 4. Menggabungkan (concatenate) semua 'heads' kembali menjadi satu dimensi d_model
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.W_o(out)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=None, dropout=0.1):
        super().__init__()
        if d_ff is None:
            d_ff = d_model * 4
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        # Apply Linear -> GELU -> Dropout -> Linear
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Pre-LN architecture (LayerNorm applied before the block)
        # 1. Self Attention with residual connection
        attn_out = self.attention(self.norm1(x), mask)
        x = x + self.dropout1(attn_out)
        
        # 2. Feed Forward with residual connection
        ff_out = self.feed_forward(self.norm2(x))
        x = x + self.dropout2(ff_out)
        
        return x
