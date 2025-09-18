from pathlib import Path
from typing import Optional, List, Dict, Any
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch
import numpy as np

from .dataset import CADItem, FusionCADDataset, build_brep_standardizer, save_stats, load_stats

def enhanced_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Улучшенная функция коллации с поддержкой псевдо-меток"""
    views_list = [item['views'] for item in batch]
    face_matrix_list = [item['face_matrix'] for item in batch]
    item_id_list = [item['item_id'] for item in batch]
    
    # Собираем псевдо-метки, если они есть
    pseudo_labels = None
    if 'pseudo_label' in batch[0]:
        pseudo_labels = torch.tensor([item['pseudo_label'] for item in batch], dtype=torch.long)
    
    views_batch = torch.stack(views_list, dim=0)
    
    result = {
        'views': views_batch,
        'face_matrix': face_matrix_list,
        'item_id': item_id_list
    }
    
    if pseudo_labels is not None:
        result['pseudo_labels'] = pseudo_labels
    
    return result

class EnhancedFusionDataModule(pl.LightningDataModule):
    """Улучшенный DataModule с поддержкой псевдо-меток"""
    
    def __init__(
        self, 
        brep_features_dir: Path, 
        dino_features_dir: Path, 
        stats_path: Path, 
        batch_size: int = 32, 
        train_split: float = 0.8,
        num_workers: int = 4
    ):
        super().__init__()
        self.brep_dir = brep_features_dir
        self.dino_dir = dino_features_dir
        self.stats_path = stats_path
        self.batch_size = batch_size
        self.train_split = train_split
        self.num_workers = num_workers
        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None):
        # Ваша существующая логика setup...
        brep_files = {self._get_clean_id(p.stem): p for p in self.brep_dir.glob("*.npz")}
        dino_files = {p.stem: p for p in self.dino_dir.glob("*.npz")}
        common_ids = sorted(brep_files.keys() & dino_files.keys())

        all_items = [
            CADItem(
                item_id=item_id,
                brep_npz_path=brep_files[item_id],
                dino_path=dino_files[item_id],
            )
            for item_id in common_ids
        ]

        train_size = int(self.train_split * len(all_items))
        self.train_items = all_items[:train_size]
        self.val_items = all_items[train_size:]

        # Стандартизатор
        if self.stats_path.exists():
            self.standardizer = load_stats(self.stats_path)
        else:
            self.standardizer = build_brep_standardizer(self.train_items)
            save_stats(self.standardizer, self.stats_path)

        # Создаём датасеты
        self.train_dataset = FusionCADDataset(
            items=self.train_items, 
            standardizer=self.standardizer
        )
        self.val_dataset = FusionCADDataset(
            items=self.val_items, 
            standardizer=self.standardizer
        )

    def update_pseudo_labels(self, pseudo_labels: np.ndarray):
        """Обновляет псевдо-метки в train датасете"""
        if hasattr(self.train_dataset, 'update_pseudo_labels'):
            self.train_dataset.update_pseudo_labels(pseudo_labels) # type: ignore

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            collate_fn=enhanced_collate_fn, 
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            collate_fn=enhanced_collate_fn, 
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )

    def _get_clean_id(self, filename: str) -> str:
        name = Path(filename).stem
        if name.endswith('.prt'):
            name = name[:-4]
        return name