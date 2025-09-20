import torch
from torch.nn import Linear, Sequential, ModuleList, BatchNorm1d, Dropout, LeakyReLU, ReLU
from typing import List

from ..dataset import BRepData


class CustomBRepEncoder(torch.nn.Module):
    """
    Self-supervised B-Rep энкодер из репозитория Self-supervised-BRep-learning-for-CAD
    Адаптирован для чистого PyTorch без torch_geometric и torch_scatter
    
    Args:
        v_in_width: Размерность входных признаков вершин
        e_in_width: Размерность входных признаков рёбер
        f_in_width: Размерность входных признаков граней
        out_width: Размерность выходных эмбеддингов
        num_layers: Количество дополнительных слоёв refinement
        use_attention: Использовать ли механизм внимания
    """
    
    def __init__(
        self, 
        v_in_width: int, 
        e_in_width: int, 
        f_in_width: int, 
        out_width: int, 
        num_layers: int, 
        use_attention: bool = False
    ) -> None:
        super().__init__()
        self.use_attention = use_attention

        # Начальные слои эмбеддинга для вершин, рёбер и граней
        self.embed_v_in = LinearBlock(v_in_width, out_width)
        self.embed_e_in = LinearBlock(e_in_width, out_width)
        self.embed_f_in = LinearBlock(f_in_width, out_width)

        # Слои передачи сообщений для кодирования иерархической структуры
        self.V2E = BipartiteResMRConv(out_width)  # Вершины -> Рёбра
        self.E2F = BipartiteResMRConv(out_width)  # Рёбра -> Грани

        # Дополнительные слои передачи сообщений для уточнения признаков
        self.message_layers = ModuleList([
            BipartiteResMRConv(out_width) for _ in range(num_layers)
        ])

        # Механизм внимания для обработки различных размеров окрестности
        if self.use_attention:
            self.attention_layers = ModuleList([
                torch.nn.MultiheadAttention(
                    embed_dim=out_width, 
                    num_heads=4, 
                    batch_first=True
                ) for _ in range(num_layers)
            ])

    def forward(self, data: BRepData) -> torch.Tensor:
        """
        Прямой проход через SSL энкодер
        
        Args:
            data: B-Rep данные модели
            
        Returns:
            x_f: [num_faces, out_width] - финальные эмбеддинги граней
        """
        # Эмбеддинг начальных признаков
        x_v: torch.Tensor = self.embed_v_in(data.vertices)  # [num_vertices, out_width]
        x_e: torch.Tensor = self.embed_e_in(data.edges)     # [num_edges, out_width]
        x_f: torch.Tensor = self.embed_f_in(data.faces)     # [num_faces, out_width]

        # Восходящий проход: передача информации от вершин к рёбрам, от рёбер к граням
        x_e = self.V2E(x_v, x_e, data.edge_to_vertex[[1, 0]])
        x_f = self.E2F(x_e, x_f, data.face_to_edge[[1, 0]])

        # Уточнение через дополнительные слои передачи сообщений
        for i, layer in enumerate(self.message_layers):
            if self.use_attention:
                # Self-attention между гранями
                x_f_unsqueezed = x_f.unsqueeze(0)  # [1, num_faces, out_width]
                attended, _ = self.attention_layers[i](
                    x_f_unsqueezed, x_f_unsqueezed, x_f_unsqueezed
                )
                x_f = attended.squeeze(0)  # [num_faces, out_width]
            else:
                # Обычная передача сообщений между гранями
                x_f = layer(x_f, x_f, data.face_to_face[:2, :])

        return x_f  # Возвращаем финальные эмбеддинги граней

class BipartiteResMRConv(torch.nn.Module):
    """
    Bipartite Residual Message-passing Convolution
    Заменяет torch_scatter на чистый PyTorch
    
    Args:
        width: Размерность признаков
    """
    
    def __init__(self, width: int) -> None:
        super().__init__()
        self.mlp = LinearBlock(2 * width, width)

    def forward(
        self, 
        x_src: torch.Tensor, 
        x_dst: torch.Tensor, 
        e: torch.Tensor
    ) -> torch.Tensor:
        """
        Передача сообщений между источником и получателем
        
        Args:
            x_src: [num_src, width] - признаки источников
            x_dst: [num_dst, width] - признаки получателей  
            e: [2, num_edges] - связи [src_indices, dst_indices]
            
        Returns:
            updated_dst: [num_dst, width] - обновлённые признаки получателей
        """
        # Извлекаем признаки по индексам связей
        src_features: torch.Tensor = x_src[e[0]]  # [num_edges, width]
        dst_features: torch.Tensor = x_dst[e[1]]  # [num_edges, width]
        
        # Вычисляем разности признаков
        diffs: torch.Tensor = dst_features - src_features  # [num_edges, width]
        
        # Ручная реализация scatter_max для агрегации по получателям
        maxes: torch.Tensor = torch.zeros_like(x_dst)  # [num_dst, width]
        
        for i in range(x_dst.size(0)):
            # Находим все рёбра, ведущие к i-той вершине/грани
            mask: torch.Tensor = (e[1] == i)
            if mask.any():
                # Берём максимум по всем входящим рёбрам
                maxes[i] = diffs[mask].max(dim=0)[0]
        
        # Остаточное соединение с MLP
        return x_dst + self.mlp(torch.cat([x_dst, maxes], dim=1))

class LinearBlock(torch.nn.Module):
    """
    Универсальный блок линейных слоёв с настройками нормализации и активации
    
    Args:
        *layer_sizes: Размеры слоёв (input_dim, hidden1, hidden2, ..., output_dim)
        batch_norm: Применять ли BatchNorm
        dropout: Вероятность Dropout (0.0 = отключён)
        last_linear: Не применять активацию к последнему слою
        leaky: Использовать LeakyReLU вместо ReLU
    """
    
    def __init__(
        self, 
        *layer_sizes: int, 
        batch_norm: bool = False, 
        dropout: float = 0.0, 
        last_linear: bool = False, 
        leaky: bool = True
    ) -> None:
        super().__init__()
        layers: List[torch.nn.Module] = []
        
        for i in range(len(layer_sizes) - 1):
            c_in: int = layer_sizes[i]
            c_out: int = layer_sizes[i + 1]
            
            # Добавляем линейный слой
            layers.append(Linear(c_in, c_out))
            
            # Если это последний слой и last_linear=True, не добавляем активацию
            if last_linear and i + 1 >= len(layer_sizes) - 1:
                break
                
            # Добавляем нормализацию
            if batch_norm:
                layers.append(BatchNorm1d(c_out))
                
            # Добавляем Dropout
            if dropout > 0:
                layers.append(Dropout(p=dropout))
                
            # Добавляем активацию
            layers.append(LeakyReLU() if leaky else ReLU())
            
        self.f = Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, input_dim] - входные признаки
            
        Returns:
            output: [batch_size, output_dim] - выходные признаки
        """
        return self.f(x)
