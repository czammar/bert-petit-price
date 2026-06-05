"""
Módulo de configuración del proyecto.
"""
from dataclasses import dataclass


@dataclass
class BERTConfig:
    """Configuración central para la arquitectura BERT y el entrenamiento."""
    vocab_size: int = 30522
    hidden_size: int = 256
    num_layers: int = 2
    num_heads: int = 4
    max_len: int = 128
    dropout: float = 0.1
    learning_rate: float = 1e-3
    batch_size: int = 16
    warmup_steps: int = 200

    pad_idx: int = 0
    unk_idx: int = 1
    cls_idx: int = 2
    sep_idx: int = 3
    mask_idx: int = 4
