from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.cluster import KMeans
import numpy as np

class DINOViewEncoder(nn.Module):
    """Энкодер для 8 DINO проекций без агрегации"""
    
    def __init__(self, input_dim: int = 384, embed_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Проекция каждого вида
        self.view_projection = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Self-attention для попарного сравнения видов
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Нормализация
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, views: torch.Tensor) -> torch.Tensor:
        """
        Args:
            views: [B, 8, 364] - 8 DINO проекций
        Returns:
            [B, embed_dim] - агрегированный эмбеддинг
        """
        # Проекция каждого вида
        projected_views = self.view_projection(views)  # [B, 8, embed_dim]
        
        # Self-attention между видами
        attended_views, attention_weights = self.attention(
            projected_views, projected_views, projected_views
        )
        
        # Остаточное соединение
        attended_views = self.norm(projected_views + attended_views)
        
        # Взвешенное суммирование на основе attention weights
        # Используем средние веса по головам для финального пулинга
        avg_weights = attention_weights.mean(dim=1)  # [B, 8]
        weights = F.softmax(avg_weights, dim=-1)
        
        # Взвешенная агрегация
        weighted_embedding = torch.sum(attended_views * weights.unsqueeze(-1), dim=1)
        
        return weighted_embedding


class FaceGeometryEncoder(nn.Module):
    """Энкодер для геометрических признаков граней"""
    
    def __init__(self, input_dim: int = 7, embed_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        
        # MLP для обработки признаков одной грани
        self.face_encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, embed_dim) 
        )
        
        # Внимание на уровне граней
        self.face_attention = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, face_matrix_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            face_matrix_list: List[Tensor] - список матриц граней переменной длины
        Returns:
            [B, embed_dim] - эмбеддинги для батча
        """
        batch_embeddings = []
        device = next(self.parameters()).device
        
        for face_matrix in face_matrix_list:
            face_matrix = face_matrix.to(device)
            
            # Кодирование каждой грани
            face_embeddings = self.face_encoder(face_matrix)  # [N_faces, 256]
            
            # Attention веса для каждой грани
            attention_weights = self.face_attention(face_embeddings)  # [N_faces, 1]
            attention_weights = F.softmax(attention_weights, dim=0)
            
            # Взвешенная агрегация граней
            model_embedding = torch.sum(face_embeddings * attention_weights, dim=0) # [256]
            batch_embeddings.append(model_embedding)
        
        return torch.stack(batch_embeddings, dim=0) # [B, 256]


class SelfSupervisedFusionModel(pl.LightningModule):
    """
    Самообучающаяся модель с кластеризацией и псевдо-метками
    """
    
    def __init__(
        self,
        dino_dim: int = 384,
        face_dim: int = 7,
        embed_dim: int = 256,
        learning_rate: float = 1e-4,
        temperature: float = 0.1,
        num_clusters: int = 10,
        cluster_update_freq: int = 5,  
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Энкодеры
        self.dino_encoder = DINOViewEncoder(dino_dim, embed_dim)
        self.face_encoder = FaceGeometryEncoder(face_dim, embed_dim)
        
        fusion_dim = embed_dim * 2
        # Проекционная голова для контрастивного обучения
        self.projection_head = nn.Sequential(
            nn.Linear(fusion_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, 128)
        )
        
        # Для кластеризации
        self.kmeans = None
        self.cluster_centers = None
        
        # Счётчики
        self.cluster_update_counter = 0

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Прямой проход"""
        
        # Получаем эмбеддинги от каждого энкодера
        dino_embed = self.dino_encoder(batch['views'])  # [B, 256]
        face_embed = self.face_encoder(batch['face_matrix'])  # [B, 256]
        
        # Фузия эмбеддингов
        fused_embed = torch.cat([dino_embed, face_embed], dim=1)  # [B, 512]
        
        # Проекция для контрастивного обучения
        projections = self.projection_head(fused_embed)  # [B, 128]
        projections = F.normalize(projections, p=2, dim=1)
        
        return {
            'embeddings': fused_embed,
            'projections': projections,
            'dino_embed': dino_embed,
            'face_embed': face_embed
        }

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        """Шаг обучения"""
        outputs = self.forward(batch)
        
        # Основная потеря: контрастивное обучение с псевдо-метками
        if 'pseudo_labels' in batch:
            contrastive_loss = self._compute_pseudo_contrastive_loss(
                outputs['projections'], 
                batch['pseudo_labels']
            )
        else:
            # Fallback: потеря реконструкции
            contrastive_loss = self._compute_reconstruction_loss(outputs, batch)
        
        # Вспомогательная потеря: согласованность модальностей
        alignment_loss = self._compute_alignment_loss(
            outputs['dino_embed'], 
            outputs['face_embed']
        )
        
        # Общая потеря
        total_loss = contrastive_loss + 0.1 * alignment_loss
        
        # Логирование
        self.log_dict({
            'train_loss': total_loss,
            'contrastive_loss': contrastive_loss,
            'alignment_loss': alignment_loss,
        }, on_step=False, on_epoch=True, prog_bar=True)
        
        return total_loss

    def validation_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        """Шаг валидации"""
        outputs = self.forward(batch)
        
        if 'pseudo_labels' in batch:
            loss = self._compute_pseudo_contrastive_loss(
                outputs['projections'], 
                batch['pseudo_labels']
            )
        else:
            loss = self._compute_reconstruction_loss(outputs, batch)
        
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def _compute_pseudo_contrastive_loss(self, projections: torch.Tensor, pseudo_labels: torch.Tensor) -> torch.Tensor:
        """InfoNCE потеря с псевдо-метками"""
        batch_size = projections.size(0)
        
        # Матрица сходства
        similarity_matrix = torch.matmul(projections, projections.T) / self.hparams.temperature # type: ignore
        
        # Маска для позитивных пар (одинаковые псевдо-метки)
        labels = pseudo_labels.unsqueeze(1)
        pos_mask = (labels == labels.T).float()
        pos_mask.fill_diagonal_(0)  # Исключаем автосходство
        
        # Маска для негативных пар
        neg_mask = (labels != labels.T).float()
        
        # Проверяем, есть ли позитивные пары
        pos_counts = pos_mask.sum(dim=1)
        if pos_counts.sum() == 0:
            # Если нет позитивных пар, используем простую потерю
            return torch.tensor(0.0, device=projections.device, requires_grad=True)
        
        # InfoNCE потеря
        exp_sim = torch.exp(similarity_matrix)
        
        # Позитивные логиты
        pos_sim = (exp_sim * pos_mask).sum(dim=1) / (pos_counts + 1e-8)
        
        # Все логиты (исключая диагональ)
        mask_diag = torch.eye(batch_size, device=projections.device, dtype=torch.bool)
        all_sim = exp_sim.masked_fill(mask_diag, 0).sum(dim=1)
        
        # Потеря
        loss = -torch.log(pos_sim / (all_sim + 1e-8))
        
        # Усредняем только по элементам с позитивными парами
        valid_mask = pos_counts > 0
        if valid_mask.sum() > 0:
            loss = loss[valid_mask].mean()
        else:
            loss = torch.tensor(0.0, device=projections.device, requires_grad=True)
        
        return loss

    def _compute_alignment_loss(self, dino_embed: torch.Tensor, face_embed: torch.Tensor) -> torch.Tensor:
        """Потеря согласованности между модальностями"""
        dino_norm = F.normalize(dino_embed, p=2, dim=1)
        face_norm = F.normalize(face_embed, p=2, dim=1)
        
        # Cosine similarity между соответствующими эмбеддингами
        cos_sim = (dino_norm * face_norm).sum(dim=1)
        
        # Хотим максимизировать сходство
        return (1 - cos_sim).mean()

    def _compute_reconstruction_loss(self, outputs: Dict, batch: Dict) -> torch.Tensor:
        """Fallback потеря реконструкции"""
        # Простая MSE между проекциями как регуляризация
        projections = outputs['projections']
        target = torch.randn_like(projections) * 0.1  # Небольшой шум как цель
        return F.mse_loss(projections, target)

    def configure_optimizers(self):  # type: ignore
        """Конфигурация оптимизатора"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,  # type: ignore
            weight_decay=1e-4
        )
        
        # Косинусный планировщик с warmup
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=10, 
            T_mult=2,
            eta_min=1e-6
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }