from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
import json
import tqdm

from ..utils.running_stats import RunningStats


class DinoProcessor:
    """L2-нормализация по каждой проекции, без усреднения по видам."""
    def __init__(self, eps: float = 1e-8) -> None:
        self.eps = eps

    def process(self, views: np.ndarray) -> np.ndarray:
        assert views.ndim == 2, "Ожидается массив [num_views, feat_dim]"
        norms = np.linalg.norm(views, ord=2, axis=1, keepdims=True)
        norms = np.maximum(norms, self.eps)
        return views / norms

def load_npz_brepnet(npz_file):
    with np.load(npz_file) as data:
        npz_data = {
            "face_features": data["face_features"], # признаки граней
            "face_point_grids": data["face_point_grids"], # координаты точек на гранях
            "edge_features": data["edge_features"], # признаки рёбер
            # "edge_point_grids": data["edge_point_grids"], # координаты точек на рёбрах
            "coedge_features": data["coedge_features"], # признаки ко-рёбер
            "coedge_point_grids": data["coedge_point_grids"], # координаты точек на ко-рёбрах
            "coedge_lcs": data["coedge_lcs"], # локальные системы координат ко-рёбер
            "coedge_scale_factors": data["coedge_scale_factors"], # масштабные коэффициенты ко-рёбер
            "coedge_reverse_flags": data["coedge_reverse_flags"], # флаги реверса ко-рёбер
            "coedge_to_next": data["next"],  # индексы следующих ко-рёбер
            "coedge_to_mate": data["mate"],  # индексы сопряжённых ко-рёбер
            "coedge_to_face": data["face"],  # индексы граней для ко-рёбер
            "coedge_to_edge": data["edge"]   # индексы рёбер для ко-рёбер
        }
    return npz_data

def load_npz_dino(npz_file):
        with np.load(npz_file) as data:
            npz_data = data["views"]
        return npz_data

# ============ Стандартизация ДО агрегации ============

class BRepStandardizer:
    """
    Хранит и применяет статистики отдельно для face, edge и coedge признаков.
    """
    def __init__(self, eps: float = 1e-7) -> None:
        self.eps = eps
        self.stats: Dict[str, Union[List[RunningStats], List[Dict[str, float]]]] = {}

    @property
    def is_fitted(self) -> bool:
        return bool(self.stats)

    def update(self, brep_data: Dict[str, np.ndarray]):
        """Вычисляет статистики онлайн для каждого типа признаков."""
        for key, features in brep_data.items():
            if not key.endswith('_features'):
                continue
            
            num_entities, num_features = features.shape
            if key not in self.stats:
                self.stats[key] = [RunningStats() for _ in range(num_features)]
            
            current_stats = self.stats[key]
            assert all(isinstance(s, RunningStats) for s in current_stats)
            assert len(current_stats) == num_features

            for i in range(num_entities):
                for j in range(num_features):
                    current_stats[j].push(features[i, j]) # type: ignore

    def transform(self, brep_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Применяет стандартизацию к данным."""
        assert self.is_fitted, "Стандартизатор не обучен"
        
        standardized_data = brep_data.copy()
        
        for key, stats_list in self.stats.items():
            if key not in standardized_data:
                continue

            feature_tensor = standardized_data[key]
            num_features = len(stats_list)
            assert feature_tensor.shape[1] == num_features, f"Несоответствие размерности для {key}"

            means = np.zeros((1, num_features), dtype=np.float32)
            stds = np.ones((1, num_features), dtype=np.float32)

            for j, s in enumerate(stats_list):
                if isinstance(s, RunningStats):
                    means[0, j] = s.mean()
                    stds[0, j] = max(s.standard_deviation(), self.eps)
                else: # Загружено из JSON
                    means[0, j] = s["mean"]
                    stds[0, j] = max(s["standard_deviation"], self.eps)
            
            standardized_data[key] = (feature_tensor - means) / stds
            
        return standardized_data

    def to_json(self) -> Dict[str, List[Dict[str, float]]]:
        """Сохраняет статистики в формате JSON."""
        output = {}
        for key, stats_list in self.stats.items():
            assert all(isinstance(s, RunningStats) for s in stats_list)
            output[key] = [{"mean": s.mean(), "standard_deviation": s.standard_deviation()} for s in stats_list] # type: ignore
        return {"feature_standardization": output} # type: ignore

    @classmethod
    def from_json(cls, data: Dict[str, Dict[str, List[Dict[str, float]]]]) -> "BRepStandardizer":
        """Загружает статистики из словаря (JSON)."""
        obj = cls()
        obj.stats = data["feature_standardization"] # type: ignore
        return obj


@dataclass(frozen=True)
class CADItem:
    item_id: str
    dino_path: Path
    brep_npz_path: Path

class FusionCADDataset(Dataset):
    def __init__(
        self,
        items: Sequence[CADItem],
        standardizer: Optional[BRepStandardizer] = None,
        dino_processor: Optional[DinoProcessor] = None,
        dtype: np.dtype = np.float32, # type: ignore
    ) -> None:
        self.items = list(items)
        self.standardizer = standardizer
        self.dino_proc = dino_processor or DinoProcessor()
        self.dtype = dtype

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        it = self.items[idx]

        views = load_npz_dino(it.dino_path)
        views = self.dino_proc.process(views)

        brep_data = load_npz_brepnet(it.brep_npz_path)
        
        if self.standardizer and self.standardizer.is_fitted:
            brep_data = self.standardizer.transform(brep_data)

        face_matrix = brep_data["face_features"].astype(self.dtype)

        return {
            "views": torch.from_numpy(views),
            "face_matrix": torch.from_numpy(face_matrix),
            "item_id": it.item_id, # type: ignore
        }

def build_brep_standardizer(train_items: Sequence[CADItem], dtype: np.dtype = np.float32) -> BRepStandardizer: # type: ignore
    """Строит стандартизатор на тренировочном наборе."""
    stdz = BRepStandardizer()
    for it in tqdm.tqdm(train_items, desc="Вычисление статистики"):
        brep = load_npz_brepnet(it.brep_npz_path)
        stdz.update(brep)
    return stdz

def save_stats(stdz: BRepStandardizer, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stdz.to_json(), f, ensure_ascii=False, indent=2)

def load_stats(path: Path) -> BRepStandardizer:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return BRepStandardizer.from_json(data)