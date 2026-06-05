"""
Módulo de interfaces base para la arquitectura del modelo.
"""
from abc import ABC, abstractmethod
from typing import Optional
import torch


class BaseAttention(ABC, torch.nn.Module):
    """Interfaz abstracta para los mecanismos de atención."""
    @abstractmethod
    def forward(
        self, q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward method."""


class BaseEncoder(ABC, torch.nn.Module):
    """Interfaz abstracta para el bloque codificador completo (Encoder)."""
    @abstractmethod
    def forward(
        self, x: torch.Tensor,
        segment_info: torch.Tensor
    ) -> torch.Tensor:
        """Forward method."""
