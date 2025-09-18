from pathlib import Path
from typing import Optional, List
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .dataset import (
    CADItem,
    FusionCADDataset,
    build_brep_standardizer,
    save_stats,
    load_stats,
)

def get_clean_id(filename: str):
    name = Path(filename).stem
    if name.endswith('.prt'):
        name = name[:-4]
    return name

def custom_collate_fn(batch):
    """Собирает батч, оставляя тензоры переменной длины в виде списка."""
    import torch
    views_list = [item['views'] for item in batch]
    face_matrix_list = [item['face_matrix'] for item in batch]
    item_id_list = [item['item_id'] for item in batch]
    
    views_batch = torch.stack(views_list, dim=0)
    return {
        'views': views_batch,
        'face_matrix': face_matrix_list,
        'item_id': item_id_list
    }

class FusionDataModule(pl.LightningDataModule):
    def __init__(self, brep_features_dir: Path, dino_features_dir: Path, stats_path: Path, batch_size: int = 32, train_split: float = 0.8):
        super().__init__()
        self.brep_dir = brep_features_dir
        self.dino_dir = dino_features_dir
        self.stats_path = stats_path
        self.batch_size = batch_size
        self.train_split = train_split
        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None):
        
        brep_files = {get_clean_id(p.stem): p for p in self.brep_dir.glob("*.npz")}
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

        # Вычисляем или загружаем стандартизатор ТОЛЬКО на train данных
        if self.stats_path.exists():
            print(f"Загрузка статистики из {self.stats_path}")
            self.standardizer = load_stats(self.stats_path)
        else:
            print("Вычисление и сохранение статистики...")
            self.standardizer = build_brep_standardizer(self.train_items)
            save_stats(self.standardizer, self.stats_path)

        self.train_dataset = FusionCADDataset(items=self.train_items, standardizer=self.standardizer)
        self.val_dataset = FusionCADDataset(items=self.val_items, standardizer=self.standardizer)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=custom_collate_fn, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=custom_collate_fn, num_workers=0)

