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
            nn.Dropout(p=0.25),
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
        p1 = nn.functional.normalize(self.projector(z1), dim=-1)
        p2 = nn.functional.normalize(self.projector(z2), dim=-1)
        labels = torch.arange(p1.size(0), device=p1.device)

        logits12 = (p1 @ p2.t()) / self.tau
        logits21 = (p2 @ p1.t()) / self.tau

        loss12 = nn.functional.cross_entropy(logits12, labels)
        loss21 = nn.functional.cross_entropy(logits21, labels)
        return 0.5 * (loss12 + loss21), logits12, labels

    def _two_views_stable(self, batch, p: float):
        # глубокая копия без изменения порядка/количества узлов
        import copy
        v1 = copy.deepcopy(batch)
        v2 = copy.deepcopy(batch)
        if p > 0.0 and self.training:
            # легкий джиттер по фичам (xyz) в LCS; порядок и размеры НЕ трогаем
            def _jitter_grid(g, sigma=0.01):
                g[:, 0:3, :] = g[:, 0:3, :] + sigma * torch.randn_like(g[:, 0:3, :])
                return g
            v1["coedge"].grid = _jitter_grid(v1["coedge"].grid)
            v2["coedge"].grid = _jitter_grid(v2["coedge"].grid)
            # можно добавить очень мягкий noise для face.uv
            v1["face"].uv = v1["face"].uv + 0.01 * torch.randn_like(v1["face"].uv)
            v2["face"].uv = v2["face"].uv + 0.01 * torch.randn_like(v2["face"].uv)
        return v1, v2

    def training_step(self, batch: HeteroData, _: int) -> Tensor:
        
        v1, v2 = self._two_views_stable(batch, p=self.aug_p)

        
        z1d = self._embed(v1)
        z2d = self._embed(v2)

        # контраст
        loss_con, logits_con, labels_con = self._contrastive_loss(z1d["coedge"], z2d["coedge"])

        if logits_con.numel() == 0:
            info_acc = torch.tensor(0.0, device=loss_con.device)
        else:
            labels = torch.arange(logits_con.size(0), device=logits_con.device)
            info_acc = (logits_con.argmax(dim=1) == labels).float().mean()

        # топологические предтексты с настраиваемой температурой
        co_batch = v1["coedge"].batch
        next_ei = v1[("coedge","next","coedge")].edge_index
        mate_ei = v1[("coedge","mate","coedge")].edge_index
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

        bsz = num_graphs_in_batch(v1)
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
        v1, v2 = self._two_views_stable(batch, p=0.0)

        v1, v2 = self._two_views_stable(batch, p=self.aug_p)

        
        z1d = self._embed(v1)
        z2d = self._embed(v2)

        # контраст
        loss_con, logits_con, labels_con = self._contrastive_loss(z1d["coedge"], z2d["coedge"])

        if logits_con.numel() == 0:
            info_acc = torch.tensor(0.0, device=loss_con.device)
        else:
            labels = torch.arange(logits_con.size(0), device=logits_con.device)
            info_acc = (logits_con.argmax(dim=1) == labels).float().mean()

        # топологические предтексты с настраиваемой температурой
        co_batch = v1["coedge"].batch
        next_ei = v1[("coedge","next","coedge")].edge_index
        mate_ei = v1[("coedge","mate","coedge")].edge_index
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
            "val_loss": loss,
            "val_loss_con": loss_con,
            "val_loss_next": loss_next,
            "val_loss_mate": loss_mate,
            "val_infoNCE_acc": info_acc,
            "val_topo_next_top1": acc_next,
            "val_topo_mate_top1": acc_mate,
        }, on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=bsz)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)