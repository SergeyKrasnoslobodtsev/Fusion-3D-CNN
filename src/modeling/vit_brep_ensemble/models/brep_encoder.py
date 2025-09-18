import torch
import torch.nn as nn
from typing import List

class BRepEncoder(nn.Module):
    """
    Энкодер для обработки B-rep признаков (набора граней).
    Использует архитектуру на основе трансформера для получения
    единого эмбеддинга для 3D-модели.
    """
    def __init__(
        self,
        input_dim: int = 7,
        embed_dim: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ):
        """
        Args:
            input_dim (int): Размерность признаков для одной грани.
            embed_dim (int): Размерность внутреннего пространства модели.
            nhead (int): Количество голов во multi-head attention.
            num_layers (int): Количество слоев в энкодере трансформера.
            dim_feedforward (int): Размерность feed-forward слоя в трансформере.
            dropout (float): Значение dropout.
        """
        super().__init__()
        self.embed_dim = embed_dim

        # Линейный слой для проекции входных признаков в embed_dim
        self.input_proj = nn.Linear(input_dim, embed_dim)

        # Специальный [CLS] токен для агрегации информации
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Слой энкодера трансформера
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # Важно для формата (batch, seq, feature)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Нормализация выхода
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, face_matrix_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Прямой проход.

        Args:
            face_matrix_list (List[torch.Tensor]): Список тензоров, где каждый
                тензор имеет форму [num_faces, input_dim].

        Returns:
            torch.Tensor: Батч эмбеддингов B-rep, форма [batch_size, embed_dim].
        """
        batch_embeddings = []
        device = self.cls_token.device

        for face_matrix in face_matrix_list:
            # 1. Проецируем входные признаки
            # [num_faces, input_dim] -> [num_faces, embed_dim]
            x = self.input_proj(face_matrix.to(device))

            # 2. Добавляем [CLS] токен в начало последовательности
            # [1, 1, embed_dim] -> [1, num_faces+1, embed_dim]
            cls_token = self.cls_token.expand(1, -1, -1)
            x = torch.cat((cls_token, x.unsqueeze(0)), dim=1)

            # 3. Прогоняем через трансформер
            # [1, num_faces+1, embed_dim] -> [1, num_faces+1, embed_dim]
            transformer_output = self.transformer_encoder(x)

            # 4. Берем выход, соответствующий [CLS] токену (первый токен)
            # [1, num_faces+1, embed_dim] -> [1, embed_dim]
            cls_embedding = transformer_output[:, 0, :]
            
            # 5. Нормализуем и добавляем в список
            cls_embedding = self.layer_norm(cls_embedding)
            batch_embeddings.append(cls_embedding)

        # Собираем эмбеддинги со всего батча в один тензор
        # List of [1, embed_dim] -> [batch_size, embed_dim]
        return torch.cat(batch_embeddings, dim=0)