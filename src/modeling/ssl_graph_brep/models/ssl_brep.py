from __future__ import annotations
from typing import Dict
import torch
from torch import nn, Tensor
import pytorch_lightning as pl
from torch_geometric.data import HeteroData

from .encoder import FaceEncoder, CoedgeEncoder, UVNetSurfaceEncoder, UVNetCurveEncoder
from .gnn import BRepHeteroGNN
from ..transforms.coedge_lcs import CoedgeLCSNormalize
from ..losses.topo_pointer import BilinearScorer, build_target_from_edge_index, graph_pointer_ce
from ..augment import two_views
from ..utils.batch_utils import num_graphs_in_batch

class SSLBRepModule(pl.LightningModule):
    """
    SSL‑модуль без «магических» чисел: tau/aug_p/lambda_topo берутся из атрибутов.
    """
    def __init__(
        self,
        proj_dim: int = 128,
        lr: float = 1e-3,
        hidden: int = 128,
        tau: float = 0.1,
        lambda_topo_mate: float = 1.5,
        lambda_topo_next: float = 1.5,
        aug_p: float = 0.15,
        topo_tau: float | None = 0.07, 
        weight_decay: float = 1e-4
    ) -> None:
        super().__init__()
        # модели
        self.face_enc = UVNetSurfaceEncoder(output_dims=hidden)
        self.coedge_enc = UVNetCurveEncoder(in_channels=12, output_dims=hidden)
        self.edge_lin = nn.Linear(10, hidden)
        self.gnn = BRepHeteroGNN(hidden=hidden, out_dim=2 * hidden)
        self.projector = nn.Sequential(
            nn.Linear(2 * hidden, 2 * hidden),
            nn.ReLU(inplace=True),
            nn.Linear(2 * hidden, proj_dim),
        )
        self.norm = CoedgeLCSNormalize(apply_reverse=True, renorm_vectors=False)
        self.scorer_next = BilinearScorer(dim=2 * hidden)
        self.scorer_mate = BilinearScorer(dim=2 * hidden)

        # гиперпараметры
        self.lr = lr
        self.weight_decay = weight_decay
        self.tau = float(tau)
        self.topo_tau = float(topo_tau) if topo_tau is not None else float(tau)
        self.lambda_topo_mate = lambda_topo_mate
        self.lambda_topo_next = lambda_topo_next
        self.aug_p = float(aug_p)  

        self.save_hyperparameters() 

    # ===== общие подфункции =====
    def _embed(self, batch: HeteroData) -> Dict[str, Tensor]:
        batch = self.norm(batch)
        x_dict = {
            "face": self.face_enc(batch["face"].uv),
            "coedge": self.coedge_enc(batch["coedge"].grid),
            "edge": self.edge_lin(batch["edge"].x.float()),
        }
        z_dict = self.gnn(x_dict, {
            ("coedge","next","coedge"): batch[("coedge","next","coedge")].edge_index,
            ("coedge","mate","coedge"): batch[("coedge","mate","coedge")].edge_index,
            ("coedge","to_face","face"): batch[("coedge","to_face","face")].edge_index,
            ("coedge","to_edge","edge"): batch[("coedge","to_edge","edge")].edge_index,
        })
        return z_dict

    def _contrastive_loss(self, z1: Tensor, z2: Tensor) -> Tensor:
        # единая реализация NT‑Xent с настраиваемой температурой
        p1 = nn.functional.normalize(self.projector(z1), dim=-1)
        p2 = nn.functional.normalize(self.projector(z2), dim=-1)
        logits = (p1 @ p2.t()) / self.tau
        labels = torch.arange(p1.size(0), device=p1.device)
        return nn.functional.cross_entropy(logits, labels)

    def training_step(self, batch: HeteroData, _: int) -> Tensor:
        
        v1, v2 = two_views(batch, p=self.aug_p)

        
        z1d = self._embed(v1)
        z2d = self._embed(v2)

        # контраст
        loss_con = self._contrastive_loss(z1d["coedge"], z2d["coedge"])
        p1 = torch.nn.functional.normalize(self.projector(z1d["coedge"]), dim=-1)
        p2 = torch.nn.functional.normalize(self.projector(z2d["coedge"]), dim=-1)
        logits = (p1 @ p2.t()) / self.tau
        labels = torch.arange(p1.size(0), device=p1.device)
        info_acc = (logits.argmax(dim=1) == labels).float().mean()

        # топологические предтексты с настраиваемой температурой
        co_batch = batch["coedge"].batch
        next_ei = batch[("coedge","next","coedge")].edge_index
        mate_ei = batch[("coedge","mate","coedge")].edge_index
        tgt_next = build_target_from_edge_index(next_ei, z1d["coedge"].size(0))
        tgt_mate = build_target_from_edge_index(mate_ei, z1d["coedge"].size(0))

        loss_next, acc_next = graph_pointer_ce(
            z1d["coedge"], co_batch, tgt_next, self.scorer_next, temperature=self.topo_tau
        )
        loss_mate, acc_mate = graph_pointer_ce(
            z1d["coedge"], co_batch, tgt_mate, self.scorer_mate, temperature=self.topo_tau
        )


        loss = (
        
            loss_con 
                + self.lambda_topo_next * loss_next 
                + self.lambda_topo_mate * loss_mate
        )

        bsz = num_graphs_in_batch(batch)
        self.log_dict({
            "train_loss": loss,
            "train_loss_con": loss_con,
            "train_loss_next": loss_next,
            "train_loss_mate": loss_mate,
            "train_infoNCE_acc": info_acc,
            "train_topo_next_top1": acc_next,
            "train_topo_mate_top1": acc_mate,
        }, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=bsz) 
        return loss

    @torch.no_grad()
    def validation_step(self, batch: HeteroData, _: int) -> None:
        # Аугментации: такие же, как в train (или p=0.0 для «чистой» оценки)
        v1, v2 = two_views(batch, p=self.aug_p)

        # Эмбеддинги
        z1d = self._embed(v1)
        z2d = self._embed(v2)

        # Контрастная часть (val/loss_con) и accuracy
        val_loss_con = self._contrastive_loss(z1d["coedge"], z2d["coedge"])
        p1 = torch.nn.functional.normalize(self.projector(z1d["coedge"]), dim=-1)
        p2 = torch.nn.functional.normalize(self.projector(z2d["coedge"]), dim=-1)
        logits = (p1 @ p2.t()) / self.tau
        labels = torch.arange(p1.size(0), device=p1.device)
        val_info_acc = (logits.argmax(dim=1) == labels).float().mean()

        # Топологические предтексты (val/loss_next, val/loss_mate) и topo@1
        co_batch = batch["coedge"].batch
        next_ei = batch[("coedge","next","coedge")].edge_index
        mate_ei = batch[("coedge","mate","coedge")].edge_index
        tgt_next = build_target_from_edge_index(next_ei, z1d["coedge"].size(0))
        tgt_mate = build_target_from_edge_index(mate_ei, z1d["coedge"].size(0))

        val_loss_next, val_topo_next = graph_pointer_ce(
            z1d["coedge"], co_batch, tgt_next, self.scorer_next, temperature=self.topo_tau
        )
        val_loss_mate, val_topo_mate = graph_pointer_ce(
            z1d["coedge"], co_batch, tgt_mate, self.scorer_mate, temperature=self.topo_tau
        )


        val_loss = (

            val_loss_con
                + self.lambda_topo_next * val_loss_next
                + self.lambda_topo_mate * val_loss_mate
        )

        # 6) Логирование на уровне эпохи
        bsz = num_graphs_in_batch(batch)
        self.log_dict({
            "val_loss": val_loss,
            "val_loss_con": val_loss_con,
            "val_loss_next": val_loss_next,
            "val_loss_mate": val_loss_mate,
            "val_infoNCE_acc": val_info_acc,
            "val_topo_next_top1": val_topo_next,
            "val_topo_mate_top1": val_topo_mate,
        }, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=bsz)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)