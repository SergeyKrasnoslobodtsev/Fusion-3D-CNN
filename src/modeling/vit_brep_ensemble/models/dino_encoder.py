import torch.nn as nn
from torch.nn import functional as F

class DINOEncoder(nn.Module):
    def __init__(self, input_dim=364, embed_dim=256):
        super().__init__()
        # Обработка каждой проекции отдельно
        self.view_encoder = nn.Linear(input_dim, embed_dim)
        
        # Self-attention для попарного сравнения
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=8,
            batch_first=True,
            dropout=0.1
        )
        
        # Финальная проекция
        self.final_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, views):
        """
        views: [B, 8, 364]
        return: [B, embed_dim] НО с сохранением информации о всех проекциях
        """
        # Кодируем каждую проекцию
        encoded_views = self.view_encoder(views)  # [B, 8, embed_dim]
        
        # Self-attention между проекциями
        attended_views, attention_weights = self.attention(
            encoded_views, encoded_views, encoded_views
        )
        
        # Взвешенное суммирование вместо простого усреднения
        weights = F.softmax(attention_weights.mean(dim=1), dim=-1)  # [B, 8]
        weighted_sum = (attended_views * weights.unsqueeze(-1)).sum(dim=1)
        
        return self.final_proj(weighted_sum)