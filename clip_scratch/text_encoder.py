import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super().__init__()
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_seq_len, d_model)
        pe[:, ::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("positional_embedding", pe)

    def forward(self, x):
        return x + self.positional_embedding[: x.size(1)]


class MHA(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.size()

        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scores = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(self.head_dim)

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_output = attn_weights @ v

        attn_output = attn_output.permute(0, 2, 1, 3).reshape(
            batch_size, seq_len, d_model
        )

        return self.proj(attn_output)


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, n_heads, mlp_ratio=4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        self.ln1 = nn.LayerNorm(d_model)
        self.mha = MHA(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_ratio),
            nn.GELU(),
            nn.Linear(d_model * mlp_ratio, d_model),
        )

    def forward(self, x, mask=None):
        residual = x
        x = self.ln1(x)
        x = self.mha(x, mask)
        x = x + residual

        return x + self.fc(self.ln2(x))


class TextEncoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, max_seq_len=77, n_layers=6, n_heads=8, emb_dim=512):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.embed = nn.Embedding(vocab_size, d_model)

        self.pos_embed = PositionalEmbedding(d_model, max_seq_len)

        self.transformer = nn.ModuleList(
            [TransformerEncoder(d_model, n_heads) for _ in range(n_layers)]
        )

        self.projection = nn.Linear(d_model, emb_dim)

    def forward(self, input_ids, mask=None):
        x = self.embed(input_ids)
        x = self.pos_embed(x)

        for encoder_layer in self.transformer:
            x = encoder_layer(x, mask=mask)

        if mask is not None:
            x = x[torch.arange(input_ids.shape[0]), torch.sum(mask, dim=1) - 1]
        else:
            x = x[:, -1]

        if self.projection is not None:
            x = self.projection(x)

        x = x / torch.norm(x, dim=-1, keepdim=True)

        return x
