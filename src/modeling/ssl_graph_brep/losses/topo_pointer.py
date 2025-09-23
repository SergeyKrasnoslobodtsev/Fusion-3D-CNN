from __future__ import annotations
from typing import Tuple
import torch
from torch import Tensor, nn
import torch.nn.functional as F

class MLPScorer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, 1)
        )
        
    def forward(self, z_i: Tensor, z_j: Tensor) -> Tensor:
        # z_i: [N, D], z_j: [M, D] -> [N, M]
        N, M = z_i.size(0), z_j.size(0)
        zi_exp = z_i.unsqueeze(1).expand(N, M, -1)  # [N, M, D]  
        zj_exp = z_j.unsqueeze(0).expand(N, M, -1)  # [N, M, D]
        pairs = torch.cat([zi_exp, zj_exp], dim=-1)  # [N, M, 2D]
        return self.mlp(pairs).squeeze(-1)  # [N, M]

class BilinearScorer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.W = nn.Parameter(torch.randn(dim, dim) * 0.01)  
        self.bias = nn.Parameter(torch.zeros(1))  
        
    def forward(self, z_i: Tensor, z_j: Tensor) -> Tensor:
        return z_i @ self.W @ z_j.t() + self.bias


@torch.no_grad()
def build_target_from_edge_index(edge_index: Tensor, num_nodes: int) -> Tensor:
    """
    Превращает edge_index [2, C] (дуга из i в j для каждого i) в target: target[i]=j для всех источников i.
    Предполагается по одному выходу из каждого i (как у next/mate). 
    """
    src, dst = edge_index[0], edge_index[1]
    target = torch.full((num_nodes,), -1, dtype=torch.long, device=edge_index.device)
    target[src] = dst
    return target

def graph_pointer_ce(
    z: Tensor,
    batch_vec: Tensor,
    target_global: Tensor,
    scorer: BilinearScorer,
    temperature: float = 0.2,
) -> Tuple[Tensor, Tensor]:
    """
    Cross-entropy по указателю внутри каждого графа: для узла i выбираем класс как ближайший узел в своём графе.
    Возвращает (loss, acc_top1).
    """
    z = nn.functional.normalize(z, dim=-1)
    loss_sum: Tensor = z.new_zeros(())
    correct = 0
    total = 0
    for g in batch_vec.unique():
        idx = (batch_vec == g).nonzero(as_tuple=False).flatten()       # узлы текущего графа
        if idx.numel() < 2:
            continue
        zg = z.index_select(0, idx)                                    # [Ng, D]
        tg = target_global.index_select(0, idx)                         # [Ng]
        # игнорируем узлы без целевого (на полюсах сфер и т.п.)
        mask = tg.ge(0)
        if mask.sum() == 0:
            continue
        zg_q = zg[mask]                                                 # источники
        # логиты по всем j внутри графа
        logits = scorer(zg_q, zg) / temperature                         # [Nq, Ng]
        # таргеты в локальных индексах графа
        tgt_idx = tg[mask]
        # сопоставим глобальные индексы целевых с локальными
        map_local = {int(idx[k]): k for k in range(idx.numel())}
        tgt_local = torch.tensor([map_local[int(j)] for j in tgt_idx.tolist()],
                                 device=z.device, dtype=torch.long)
        loss_sum = loss_sum + nn.functional.cross_entropy(
                        logits, tgt_local, reduction='sum'
                    )

        pred = logits.argmax(dim=1)
        correct += (pred == tgt_local).sum().item()
        total += tgt_local.numel()
    avg_loss = loss_sum / max(total, 1)
    acc = z.new_tensor(correct / max(total, 1))
    return avg_loss, acc