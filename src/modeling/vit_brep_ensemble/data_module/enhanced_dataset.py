from typing import Dict, Optional, Sequence
import numpy as np
import torch
from torch import Tensor

from .dataset import FusionCADDataset, CADItem, BRepStandardizer, DinoProcessor

class EnhancedFusionCADDataset(FusionCADDataset):
    """Расширенный датасет с поддержкой псевдо-меток"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pseudo_labels: Optional[np.ndarray] = None

    def update_pseudo_labels(self, pseudo_labels: np.ndarray):
        """Обновляет псевдо-метки"""
        assert len(pseudo_labels) == len(self.items), \
            f"Pseudo labels length {len(pseudo_labels)} != dataset length {len(self.items)}"
        self.pseudo_labels = pseudo_labels

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        # Получаем базовые данные
        item_data = super().__getitem__(idx)
        
        # Добавляем псевдо-метку, если есть
        if self.pseudo_labels is not None:
            item_data['pseudo_label'] = int(self.pseudo_labels[idx]) # type: ignore
        
        return item_data