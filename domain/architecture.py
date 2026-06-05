"""
Módulo de implementación de la arquitectura BERT con RoPE.
"""
import math
from typing import Tuple, Optional
from domain.base import BaseAttention, BaseEncoder

import torch
import torch.nn as nn


def precompute_freqs_cis(
        dim: int,
        seq_len: int,
        theta: float = 10000.0
        ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precalcula las frecuencias sinusoidales para RoPE."""
    freqs = 1.0 / (theta ** (torch.arange(
        0, dim, 2
        )[: (dim // 2)].float() / dim))
    t = torch.arange(seq_len, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.cat((freqs, freqs), dim=-1)
    _fres_cis_cos = freqs_cis.cos().unsqueeze(0).unsqueeze(0)
    _fres_cis_sin = freqs_cis.sin().unsqueeze(0).unsqueeze(0)
    return _fres_cis_cos, _fres_cis_sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rota la mitad de las dimensiones del tensor en el plano complejo."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Aplica las frecuencias RoPE."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MultiHeadAttentionRoPE(BaseAttention):
    """Mecanismo de atención multicabezal que utiliza RoPE."""
    def __init__(
            self,
            hidden_size: int,
            num_heads: int,
            max_len: int = 512,
            dropout: float = 0.1
            ):
        super().__init__()
        self.d_k = hidden_size // num_heads
        self.num_heads = num_heads

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

        cos, sin = precompute_freqs_cis(self.d_k, max_len)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            mask: Optional[torch.Tensor] = None
            ) -> torch.Tensor:
        batch_size = q.size(0)
        seq_len = q.size(1)

        Q = self.query(q).view(
            batch_size, -1, self.num_heads, self.d_k
            ).transpose(1, 2)
        K = self.key(k).view(
            batch_size, -1, self.num_heads, self.d_k
            ).transpose(1, 2)
        V = self.value(v).view(
            batch_size, -1, self.num_heads, self.d_k
            ).transpose(1, 2)

        cos_seq = self.cos[:, :, :seq_len, :]
        sin_seq = self.sin[:, :, :seq_len, :]

        Q, K = apply_rotary_pos_emb(Q, K, cos_seq, sin_seq)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention = torch.softmax(scores, dim=-1)
        x = torch.matmul(
            self.dropout(attention), V
            ).transpose(1, 2).contiguous()
        return self.out(x.view(batch_size, -1, self.num_heads * self.d_k))


class BERTEmbedding(nn.Module):
    """Módulo que combina los embeddings de tokens y segmentos."""
    def __init__(
            self, vocab_size: int, hidden_size: int,
            dropout: float = 0.1, pad_idx: int = 0
            ):
        super().__init__()
        self.token_embedding = nn.Embedding(
            vocab_size, hidden_size, padding_idx=pad_idx
            )
        self.segment_embedding = nn.Embedding(
            3,
            hidden_size,
            padding_idx=pad_idx
            )
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
            self, x: torch.Tensor,
            segment_info: torch.Tensor
            ) -> torch.Tensor:
        """Forward Method."""
        embedding = self.token_embedding(x) + \
            self.segment_embedding(segment_info)
        return self.dropout(self.norm(embedding))


class PositionwiseFeedForward(nn.Module):
    """Red prealimentada (FFN)."""
    def __init__(self, hidden_size: int, ff_hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, ff_hidden_size)
        self.fc2 = nn.Linear(ff_hidden_size, hidden_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward Method."""
        return self.fc2(self.dropout(self.activation(self.fc1(x))))


class EncoderLayer(nn.Module):
    """Un bloque completo del Encoder de un Transformer."""
    def __init__(
            self,
            hidden_size: int,
            num_heads: int,
            ff_hidden_size: int,
            max_len: int,
            dropout: float = 0.1
            ):
        """Init."""
        super().__init__()
        self.attention = MultiHeadAttentionRoPE(
            hidden_size, num_heads, max_len, dropout
            )
        self.ffn = PositionwiseFeedForward(
            hidden_size, ff_hidden_size, dropout
            )

        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.dropout(self.attention(x, x, x, mask)))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class BERT(BaseEncoder):
    """Ensambla el modelo de lenguaje principal."""
    def __init__(
            self,
            vocab_size: int,
            hidden_size: int = 768,
            num_layers: int = 6,
            num_heads: int = 8,
            max_len: int = 512,
            dropout: float = 0.1,
            pad_idx: int = 0):
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding = BERTEmbedding(
            vocab_size, hidden_size, dropout, pad_idx
            )
        self.layers = nn.ModuleList([
            EncoderLayer(
                hidden_size, num_heads, hidden_size * 4, max_len, dropout
                )
            for _ in range(num_layers)
        ])
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def forward(
            self, x: torch.Tensor,
            segment_info: torch.Tensor
            ) -> torch.Tensor:
        """Forward Method."""
        mask = (x != self.pad_idx).unsqueeze(1).unsqueeze(2) 
        out = self.embedding(x, segment_info)
        for layer in self.layers:
            out = layer(out, mask)
        return out
