import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

from .normlization import BrepNetStandardizer
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .model.encoder import CustomBRepEncoder

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

class BRepDataset(Dataset):
    """
    Dataset для загрузки и предобработки B-Rep данных.
    """
    def __init__(self, 
                 data_dir: str, 
                 split: str, 
                 encoder: 'CustomBRepEncoder', 
                 standardizer: Optional[BrepNetStandardizer] = None):
        """
        Инициализация датасета.

        Аргументы:
            data_dir (str): Путь к директории с данными.
            split (str): Раздел данных ('train', 'val', 'test').
            encoder (CustomBRepEncoder): Энкодер для B-Rep данных.
            standardizer (Optional[BrepNetStandardizer]): Стандартизатор для нормализации данных.
        """
        self.data_dir = Path(data_dir) / split
        self.encoder = encoder
        self.standardizer = standardizer

        self.file_list = list(self.data_dir.glob('*.npz'))
        if not self.file_list:
            raise ValueError(f"No data files found in {self.data_dir}")

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> BRepData:
        file_path = self.file_list[idx]
        data = np.load(file_path)

        vertices = torch.tensor(data['vertices'], dtype=torch.float32)
        edges = torch.tensor(data['edges'], dtype=torch.float32)
        faces = torch.tensor(data['faces'], dtype=torch.float32)
        edge_to_vertex = torch.tensor(data['edge_to_vertex'], dtype=torch.long)
        face_to_edge = torch.tensor(data['face_to_edge'], dtype=torch.long)
        face_to_face = torch.tensor(data['face_to_face'], dtype=torch.long)

        if self.standardizer:
            vertices = self.standardizer.standardize(vertices)

        brep_data = BRepData(
            vertices=vertices,
            edges=edges,
            faces=faces,
            edge_to_vertex=edge_to_vertex,
            face_to_edge=face_to_edge,
            face_to_face=face_to_face
        )

        return brep_data