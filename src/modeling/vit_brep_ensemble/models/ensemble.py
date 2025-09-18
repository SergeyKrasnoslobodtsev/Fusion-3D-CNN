from typing import Dict
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn import functional as F
from .dino_encoder import DINOEncoder
from .brep_encoder import BRepEncoder


class ContrastiveFusionModel(pl.LightningModule):
    """
    Мультимодальная модель для самообучения, использующая контрастивную потерю
    для сближения эмбеддингов DINO (визуальные) и B-Rep (геометрические).
    """
    def __init__(self, 
                 dino_dim: int = 384,
                 brep_dim: int = 7, 
                 embed_dim: int = 256, 
                 learning_rate: float = 1e-4, 
                 temperature: float = 0.07):
        super().__init__()
        self.learning_rate = learning_rate
        self.temperature = temperature
        self.embed_dim = embed_dim
        self.save_hyperparameters()

        # Энкодер для DINO-признаков (проекций)
        self.dino_encoder = DINOEncoder(input_dim=dino_dim, embed_dim=embed_dim)
        # Энкодер для B-Rep признаков (геометрии)
        self.brep_encoder = BRepEncoder(input_dim=brep_dim, embed_dim=embed_dim)

        
    def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """Прямой проход для получения эмбеддингов от обеих модальностей."""
        
        # Получаем эмбеддинги DINO
        # batch['views'] имеет размер [B, 8, 384]
        dino_embeddings = self.dino_encoder(batch['views'])
        
        # Получаем эмбеддинги B-Rep
        # batch['face_matrix'] - это список тензоров, обрабатываем в цикле
        brep_embeddings = self.brep_encoder(batch['face_matrix'])

        return {
            "dino_embed": dino_embeddings,
            "brep_embed": brep_embeddings
        }
    
    def _compute_contrastive_loss(self, dino_embed: torch.Tensor, brep_embed: torch.Tensor) -> torch.Tensor:
        """Вычисляет симметричную контрастивную потерю (как в CLIP)."""
        # L2-нормализация эмбеддингов
        dino_embed_norm = F.normalize(dino_embed, p=2, dim=1)
        brep_embed_norm = F.normalize(brep_embed, p=2, dim=1)
        
        # Матрица косинусного сходства [B, B]
        logits = torch.matmul(dino_embed_norm, brep_embed_norm.T) / self.temperature

        batch_size = dino_embed.size(0)
        target = torch.arange(batch_size, device=logits.device)
        
        loss_dino_to_brep = F.cross_entropy(logits, target)
        loss_brep_to_dino = F.cross_entropy(logits.T, target)
        
        loss = 0.5 * (loss_dino_to_brep + loss_brep_to_dino)
        return loss

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        # Получаем эмбеддинги
        embeddings = self.forward(batch)
        
        # Считаем loss
        loss = self._compute_contrastive_loss(embeddings["dino_embed"], embeddings["brep_embed"])
        
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=False)
        return loss
    
    def validation_step(self, batch: Dict, batch_idx: int):
        # Получаем эмбеддинги
        embeddings = self.forward(batch)
        
        # Считаем loss
        loss = self._compute_contrastive_loss(embeddings["dino_embed"], embeddings["brep_embed"])
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer