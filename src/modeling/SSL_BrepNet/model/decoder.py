import torch
import torch.nn as nn
from typing import List

class ConditionalDecoder(nn.Module):
    """
    Оригинальный декодер из репозитория Self-supervised-BRep-learning-for-CAD
    Используется для самообучения через восстановление 3D поверхности
    """
    def __init__(
        self, 
        latent_size: int, 
        hidden_dims: List[int], 
        uv_input_dim: int = 2, 
        output_dim: int = 4
    ) -> None:
        """
        Args:
            latent_size: Размер латентного вектора от энкодера
            hidden_dims: Список размеров скрытых слоёв  
            uv_input_dim: Размерность UV координат (по умолчанию 2)
            output_dim: Размерность выхода [x, y, z, sdf] (по умолчанию 4)
        """
        super().__init__()
        self.latent_size = latent_size
        self.uv_input_dim = uv_input_dim
        self.output_dim = output_dim

        # Входная размерность = latent_size + uv_input_dim
        input_dim = latent_size + uv_input_dim

        # Строим полносвязную сеть
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # Скрытые слои используют ReLU
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)

    def forward(
        self, 
        uv_coords: torch.Tensor, 
        latent_vector: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            uv_coords: [batch_size, uv_input_dim] - UV координаты  
            latent_vector: [latent_size] - вектор от энкодера
            
        Returns:
            output: [batch_size, output_dim] - восстановленная поверхность [x,y,z,sdf]
        """
        # Расширяем latent_vector для совпадения с размером batch
        latent_vector = latent_vector.unsqueeze(0).repeat(uv_coords.shape[0], 1)
        
        # Конкатенация латентного вектора и UV координат
        x = torch.cat([latent_vector, uv_coords], dim=-1)
        
        # Прохождение через сеть
        output = self.network(x)
        return output
