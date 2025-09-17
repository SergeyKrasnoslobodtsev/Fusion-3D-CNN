from __future__ import annotations
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import HeteroData, Dataset


def load_npz_brepnet(npz_file: Union[str, Path]) -> Dict[str, np.ndarray]:
    """
    Безопасная и явная загрузка .npz из BRepNetExtractor.
    Возвращает обычный словарь с ключами, согласованными с последующей сборкой графа.
    """
    with np.load(npz_file, allow_pickle=False) as data: 
        npz_data: Dict[str, np.ndarray] = {
            "face_features": data["face_features"],
            "face_point_grids": data["face_point_grids"],
            "edge_features": data["edge_features"],
            "coedge_features": data["coedge_features"],
            "coedge_point_grids": data["coedge_point_grids"],
            "coedge_lcs": data["coedge_lcs"],
            "coedge_scale_factors": data["coedge_scale_factors"],
            "coedge_reverse_flags": data["coedge_reverse_flags"],
            "coedge_to_next": data["next"],
            "coedge_to_mate": data["mate"],
            "coedge_to_face": data["face"],
            "coedge_to_edge": data["edge"],
        }
    return npz_data


def _to_float32(x: np.ndarray) -> Tensor:
    return torch.from_numpy(np.asarray(x, dtype=np.float32))


def _to_int64(x: np.ndarray) -> Tensor:
    return torch.from_numpy(np.asarray(x, dtype=np.int64))


class BRepNPZDataset(Dataset):
    """
    Dataset для .npz из BRepNetExtractor: конструирует гетерограф face/edge/coedge и связи.
    """
    def __init__(
        self,
        root_dir: Union[str, Path],
        transform: Optional[Callable[[HeteroData], HeteroData]] = None,
        pre_transform: Optional[Callable[[HeteroData], HeteroData]] = None,
        file_filter: Optional[Callable[[Path], bool]] = None,
    ) -> None:
        super().__init__(root=str(root_dir), transform=transform, pre_transform=pre_transform)
        self.root_dir = Path(root_dir)
        files = sorted(self.root_dir.glob("*.npz"))
        self.files: List[Path] = [f for f in files if file_filter(f)] if file_filter else list(files)
        if len(self.files) == 0:
            raise FileNotFoundError(f"No .npz found in {self.root_dir}")

    def len(self) -> int:
        return len(self.files)

    def get(self, idx: int) -> HeteroData:
        path = self.files[idx]
        d: Dict[str, np.ndarray] = load_npz_brepnet(path)

        # Узловые признаки
        face_x = _to_float32(d["face_features"])              # [F, 7]
        face_uv = _to_float32(d["face_point_grids"])          # [F, 7, 10, 10]
        edge_x = _to_float32(d["edge_features"])              # [E, 10]
        coedge_x = _to_float32(d["coedge_features"])          # [C, 1]
        coedge_grid = _to_float32(d["coedge_point_grids"])    # [C, 12, 10]
        coedge_lcs = _to_float32(d["coedge_lcs"])             # [C, 4, 4]
        coedge_scale = _to_float32(d["coedge_scale_factors"]) # [C]
        coedge_reverse = torch.from_numpy(
            np.asarray(d["coedge_reverse_flags"], dtype=np.bool_)
        )  # [C]

        # Размерности (строго int)
        num_faces = int(face_x.size(0))
        num_edges = int(edge_x.size(0))
        num_coedges = int(coedge_x.size(0))

        # Индексы связей (строго int64)
        nxt = _to_int64(d["coedge_to_next"])    # [C]
        mate = _to_int64(d["coedge_to_mate"])   # [C]
        c2f = _to_int64(d["coedge_to_face"])    # [C]
        c2e = _to_int64(d["coedge_to_edge"])    # [C]

        src = torch.arange(num_coedges, dtype=torch.int64)
        edge_index_next = torch.stack([src, nxt], dim=0)   # [2, C]
        edge_index_mate = torch.stack([src, mate], dim=0)  # [2, C]
        edge_index_c2f = torch.stack([src, c2f], dim=0)    # [2, C]
        edge_index_c2e = torch.stack([src, c2e], dim=0)    # [2, C]

        data = HeteroData()

        data["face"].x = face_x
        data["face"].uv = face_uv
        data["face"].num_nodes = num_faces

        data["edge"].x = edge_x
        data["edge"].num_nodes = num_edges

        data["coedge"].x = coedge_x
        data["coedge"].grid = coedge_grid
        data["coedge"].lcs = coedge_lcs
        data["coedge"].scale = coedge_scale
        data["coedge"].reverse = coedge_reverse
        data["coedge"].num_nodes = num_coedges

        data[("coedge", "next", "coedge")].edge_index = edge_index_next
        data[("coedge", "mate", "coedge")].edge_index = edge_index_mate
        data[("coedge", "to_face", "face")].edge_index = edge_index_c2f
        data[("coedge", "to_edge", "edge")].edge_index = edge_index_c2e

        if self.pre_transform is not None:
            data = self.pre_transform(data)
        return data