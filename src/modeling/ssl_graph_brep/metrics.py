from __future__ import annotations
import torch
from torch import Tensor

@torch.no_grad()
def alignment(z1: Tensor, z2: Tensor) -> Tensor:
    """
    Средняя квадратичная дистанция между эмбеддингами позитивных пар (чем меньше, тем лучше).
    Ожидает z1,z2 нормализованными по L2. [ICML'20 alignment] 
    """
    z1n = torch.nn.functional.normalize(z1, dim=-1)
    z2n = torch.nn.functional.normalize(z2, dim=-1)
    return ((z1n - z2n) ** 2).sum(dim=-1).mean()


@torch.no_grad()
def uniformity(z: Tensor, t: float = 2.0, max_points: int = 2048) -> Tensor:
    """
    Оценка равномерности на сфере: E[exp(-t * ||zi - zj||^2)], меньше — равномернее.
    Используем случайную подвыборку для квадратичной части. [ICML'20 uniformity]
    """
    z = torch.nn.functional.normalize(z, dim=-1)
    n = z.size(0)
    if n > max_points:
        idx = torch.randperm(n, device=z.device)[:max_points]
        z = z[idx]
        n = z.size(0)
    # попарные расстояния без диагонали
    sim = z @ z.t()
    d2 = (2 - 2 * sim).clamp_min(0)  # ||zi - zj||^2 = 2 - 2 cos
    d2 = d2 + torch.eye(n, device=z.device) * 1e6
    val = torch.exp(-t * d2)
    return (val.sum() - n * torch.exp(torch.tensor(-t * 1e6, device=z.device))) / (n * (n - 1))


@torch.no_grad()
def info_nce_top1_accuracy(z1: Tensor, z2: Tensor, tau: float = 0.2) -> Tensor:
    """
    Accuracy: для каждого i, позитивный j=i должен быть argmax среди всех z2. [SimCLR kNN/linear eval практика]
    """
    q = torch.nn.functional.normalize(z1, dim=-1)
    k = torch.nn.functional.normalize(z2, dim=-1)
    logits = (q @ k.t()) / tau
    pred = logits.argmax(dim=1)
    labels = torch.arange(q.size(0), device=q.device)
    return (pred == labels).float().mean()


@torch.no_grad()
def build_target_from_edge_index(edge_index: Tensor, num_nodes: int) -> Tensor:
    """
    Превращает edge_index вида [2, C] (уникальная дуга на коэдж) в вектор таргетов длины C:
    target[i] = j, куда ведёт ребро из i. [используем для mate/next]
    """
    assert edge_index.dim() == 2 and edge_index.size(0) == 2
    src, dst = edge_index[0], edge_index[1]
    target = torch.full((num_nodes,), -1, dtype=torch.long, device=edge_index.device)
    target[src] = dst
    return target


@torch.no_grad()
def top1_retrieval_accuracy_within_graphs(
    z: Tensor,
    edge_index: Tensor,
    batch_vec: Tensor,
) -> Tensor:
    """
    Для каждого узла находим ближайшего соседа по косинусу в пределах своего графа (исключая себя)
    и сравниваем с таргетом из edge_index (mate или next). Возвращаем среднюю точность. 
    """
    z = torch.nn.functional.normalize(z, dim=-1)
    num_nodes = z.size(0)
    target = build_target_from_edge_index(edge_index, num_nodes)
    assert target.ge(0).all(), "edge_index должен покрывать все источники"
    acc_sum, cnt = 0.0, 0
    for g in batch_vec.unique():
        mask = batch_vec == g
        idx = mask.nonzero(as_tuple=False).flatten()
        if idx.numel() < 2:
            continue
        zg = z.index_select(0, idx)
        sim = zg @ zg.t()
        sim.fill_diagonal_(-1.0)  # исключаем себя
        nn_local = sim.argmax(dim=1)                      # локальные индексы
        nn_global = idx[nn_local]                         # обратно в глобальные
        tgt_global = target.index_select(0, idx)          # таргеты для этой группы
        acc_sum += (nn_global == tgt_global).float().sum().item()
        cnt += idx.numel()
    return torch.tensor(acc_sum / max(cnt, 1), device=z.device)