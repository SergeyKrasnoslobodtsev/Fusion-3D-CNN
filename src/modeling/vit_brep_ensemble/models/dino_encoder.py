import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn import functional as F

class DINOEncoder(nn.Module):
    """Энкодер для попарного сравнения 8 проекций DINO."""
    def __init__(self, input_dim: int = 364, embed_dim: int = 256):
        super().__init__()
        # Проекция DINO-признаков в рабочее пространство
        self.projection = nn.Linear(input_dim, embed_dim)
        
        # Self-attention для попарного сравнения проекций
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=8,
            batch_first=True,
            dropout=0.1
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, dino_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            dino_features (torch.Tensor): Признаки DINO, форма (B, 8, 364).
        Returns:
            torch.Tensor: Агрегированный эмбеддинг, форма (B, embed_dim).
        """
        # (B, 8, 364) -> (B, 8, embed_dim)
        projected = self.projection(dino_features)
        
        # Попарное сравнение и агрегация внимания
        attended, _ = self.attention(projected, projected, projected)
        
        # Остаточное соединение и нормализация
        attended = self.norm(projected + attended)
        
        # Агрегация 8 проекций в один вектор (среднее)
        embedding = attended.mean(dim=1)
        return embedding