import json
import os
import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Optional
from dataclasses import dataclass


@dataclass
class BRepData:
    """
    Структура данных для B-Rep представления CAD модели.
    """
    vertices: torch.Tensor      # [num_vertices, v_in_width]
    edges: torch.Tensor         # [num_edges, e_in_width]
    faces: torch.Tensor         # [num_faces, f_in_width]
    edge_to_vertex: torch.Tensor # [2, num_edges]
    face_to_edge: torch.Tensor   # [2, num_face_edges]
    face_to_face: torch.Tensor   # [2, num_face_connections]
    face_batch_idx: Optional[torch.Tensor] = None

class BrepNetSSLDataset(Dataset):
    def __init__(self, root_dir, stats_file='dataset.json', num_samples=10000):  # num_samples для SDF sampling
        self.root_dir = root_dir
        self.files = [f for f in os.listdir(root_dir) if f.endswith('.npz')]
        with open(stats_file, 'r') as f:
            stats = json.load(f)
            # Из BRepNet: mean/std по каналам (адаптируй ключи, если отличаются)
            self.vertex_mean = torch.tensor(stats['vertex_feature_means'])
            self.vertex_std = torch.tensor(stats['vertex_feature_stds'])
            self.edge_mean = torch.tensor(stats['edge_feature_means'])
            self.edge_std = torch.tensor(stats['edge_feature_stds'])
            self.face_mean = torch.tensor(stats['face_feature_means'])
            self.face_std = torch.tensor(stats['face_feature_stds'])
        self.num_samples = num_samples  # Для sampling points

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.root_dir, self.files[idx])
        data = np.load(file_path)
        
        # Загрузи признаки (формат BRepNet .npz)
        vertices = torch.from_numpy(data['vertex_features']).float()  # [N, 3]
        edges = torch.from_numpy(data['edge_features']).float()      # [M, 6]
        faces = torch.from_numpy(data['face_features']).float()      # [K, 7]
        
        # Примени нормализацию (z-score)
        vertices = (vertices - self.vertex_mean) / self.vertex_std
        edges = (edges - self.edge_mean) / self.edge_std
        faces = (faces - self.face_mean) / self.face_std
        
        # Генерация sampled_points и SDF (адаптируй из sampled_data.py)
        # Здесь упрощенная версия; реализуй полный rasterization для твоих BRep
        sampled_points = torch.rand(self.num_samples, 3) * 2 - 1  # Пример: [-1,1] bounding box
        sdf = self.compute_sdf(sampled_points, vertices, edges, faces)  # Реализуй функцию SDF
        
        return {
            'vertices': vertices,
            'edges': edges,
            'faces': faces,
            'points': sampled_points,
            'sdf': sdf
        }

    def compute_sdf(self, points, vertices, edges, faces):
        # Реализуй дифференцируемый SDF на основе BRep (как в self-supervised репозитории)
        # Пример заглушка: расстояние до ближайшей точки (замени на реальный)
        return torch.norm(points - vertices.mean(dim=0), dim=1).unsqueeze(1)  # [num_samples, 1]