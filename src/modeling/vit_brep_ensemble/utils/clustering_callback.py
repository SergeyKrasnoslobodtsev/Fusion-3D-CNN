import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import torch
import numpy as np
from sklearn.cluster import KMeans
import logging

logger = logging.getLogger(__name__)

class PseudoLabelingCallback(Callback):
    """Callback для генерации псевдо-меток через кластеризацию"""
    
    def __init__(
        self, 
        num_clusters: int = 50, 
        update_freq: int = 5,
        start_epoch: int = 1
    ):
        self.num_clusters = num_clusters
        self.update_freq = update_freq
        self.start_epoch = start_epoch

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Обновляем псевдо-метки в начале эпохи"""
        current_epoch = trainer.current_epoch
        
        # Проверяем условия обновления
        if (current_epoch < self.start_epoch or 
            current_epoch % self.update_freq != 0):
            return
        
        logger.info(f"Epoch {current_epoch}: Updating pseudo-labels with {self.num_clusters} clusters...")
        
        # Переводим модель в режим eval
        pl_module.eval()
        
        # Собираем эмбеддинги со всего train датасета
        embeddings = []
        dataloader = trainer.train_dataloader
        
        with torch.no_grad():
            for batch in dataloader:  # type: ignore
                # Переносим на device
                batch = {k: v.to(pl_module.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Получаем эмбеддинги
                outputs = pl_module(batch)
                embeddings.append(outputs['projections'].cpu().numpy())
        
        # Объединяем все эмбеддинги
        all_embeddings = np.concatenate(embeddings, axis=0)
        
        # Выполняем кластеризацию
        kmeans = KMeans(
            n_clusters=self.num_clusters, 
            random_state=42, 
            n_init=10,
            max_iter=300
        )
        pseudo_labels = kmeans.fit_predict(all_embeddings)
        
        # Обновляем псевдо-метки в датасете
        if hasattr(trainer.datamodule, 'update_pseudo_labels'): # type: ignore
            trainer.datamodule.update_pseudo_labels(pseudo_labels) # type: ignore
        elif hasattr(trainer.train_dataloader.dataset, 'update_pseudo_labels'): # type: ignore
            trainer.train_dataloader.dataset.update_pseudo_labels(pseudo_labels) # type: ignore
        else:
            logger.warning("Cannot update pseudo-labels: dataset doesn't support it")
        
        # Возвращаем модель в режим train
        pl_module.train()
        
        logger.info(f"Pseudo-labels updated. Unique labels: {len(np.unique(pseudo_labels))}")

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Обновляем псевдо-метки для validation"""
        current_epoch = trainer.current_epoch
        
        # Проверяем условия обновления
        if (current_epoch < self.start_epoch or 
            current_epoch % self.update_freq != 0):
            return
        
        logger.info(f"Epoch {current_epoch}: Updating pseudo-labels with {self.num_clusters} clusters...")
        
        # Переводим модель в режим eval
        pl_module.eval()
        
        # Собираем эмбеддинги со всего train датасета
        embeddings = []
        dataloader = trainer.train_dataloader
        
        with torch.no_grad():
            for batch in dataloader:  # type: ignore
                # Переносим на device
                batch = {k: v.to(pl_module.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Получаем эмбеддинги
                outputs = pl_module(batch)
                embeddings.append(outputs['projections'].cpu().numpy())
        
        # Объединяем все эмбеддинги
        all_embeddings = np.concatenate(embeddings, axis=0)
        
        # Выполняем кластеризацию
        kmeans = KMeans(
            n_clusters=self.num_clusters, 
            random_state=42, 
            n_init=10,
            max_iter=300
        )
        pseudo_labels = kmeans.fit_predict(all_embeddings)
        
        # Обновляем псевдо-метки в датасете
        if hasattr(trainer.datamodule, 'update_pseudo_labels'): # type: ignore
            trainer.datamodule.update_pseudo_labels(pseudo_labels) # type: ignore
        elif hasattr(trainer.train_dataloader.dataset, 'update_pseudo_labels'): # type: ignore
            trainer.train_dataloader.dataset.update_pseudo_labels(pseudo_labels) # type: ignore
        else:
            logger.warning("Cannot update pseudo-labels: dataset doesn't support it")
        
        # Возвращаем модель в режим train
        pl_module.train()
        
        logger.info(f"Pseudo-labels updated. Unique labels: {len(np.unique(pseudo_labels))}")