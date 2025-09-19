from __future__ import annotations
import torch
from typing import Optional, Union
from pathlib import Path

import pytorch_lightning as pl
from torch_geometric.loader import DataLoader

from .brep_dataset import BRepNPZDataset


class BRepDataModule(pl.LightningDataModule):
    """
    LightningDataModule для .npz B-Rep графов.
    Делит список файлов на train/val/test и создаёт PyG DataLoader для HeteroData.
    """
    def __init__(
        self,
        data_dir: Union[str, Path],
        batch_size: int = 32,
        num_workers: int = 4,
        val_ratio: float = 0.2,
        test_ratio: float = 0.15,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

        self.ds_train: Optional[BRepNPZDataset] = None
        self.ds_val: Optional[BRepNPZDataset] = None
        self.ds_test: Optional[BRepNPZDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        files = sorted(self.data_dir.glob("*.npz"))
        if len(files) == 0:
            raise FileNotFoundError(f"No .npz in {self.data_dir}")

        # Разбиение
        n = len(files)
        n_test = int(n * self.test_ratio)
        n_val = int(n * self.val_ratio)
        n_train = n - n_val - n_test

        g = torch.Generator().manual_seed(self.seed)
        perm = torch.randperm(n, generator=g).tolist()

        train_files = {files[i] for i in perm[:n_train]}
        val_files = {files[i] for i in perm[n_train : n_train + n_val]}
        test_files = {files[i] for i in perm[n_train + n_val :]}

        def select(paths: set[Path]):
            return lambda p: p in paths

        self.ds_train = BRepNPZDataset(self.data_dir, file_filter=select(train_files))
        self.ds_val = BRepNPZDataset(self.data_dir, file_filter=select(val_files))
        self.ds_test = BRepNPZDataset(self.data_dir, file_filter=select(test_files))

    def train_dataloader(self) -> DataLoader:
        assert self.ds_train is not None
        return DataLoader(
            self.ds_train, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self) -> DataLoader:
        assert self.ds_val is not None
        return DataLoader(
            self.ds_val, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers
        )

    def test_dataloader(self) -> DataLoader:
        assert self.ds_test is not None
        return DataLoader(
            self.ds_test, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers
        )