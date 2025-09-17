from __future__ import annotations
from typing import List, Optional, Tuple
from pathlib import Path
import numpy as np
import torch
from torch import Tensor
from torch_geometric.nn import global_mean_pool, global_max_pool

from ..ssl_graph_brep.models.ssl_brep import SSLBRepModule
from ..ssl_graph_brep.data_module.brep_dataset import BRepNPZDataset

# _embed(batch) должен возвращать {'coedge': [NC,Dc], 'face': [NF,Df], 'edge': ...}

@torch.no_grad()
def extract_embeddings(
    model_ckpt: Path,
    data_dir: Path,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Tuple[List[str], np.ndarray]:

    model = SSLBRepModule.load_from_checkpoint(model_ckpt).to(device).eval()
    ds = BRepNPZDataset(data_dir)
    ids: List[str] = []
    embs: List[np.ndarray] = []
    for i in range(len(ds)):
        data = ds.get(i)
        # Для одного графа нет batch-векторов: создаём нулевые
        data = data.to(device)
        z = model._embed(data)  # {'coedge': [NC,Dc], 'face': [NF,Df], ...}
        co_batch = torch.zeros(z["coedge"].size(0), dtype=torch.long, device=device)
        fa_batch = torch.zeros(z["face"].size(0),   dtype=torch.long, device=device)
        g = _pool_model_embedding(z["coedge"], z["face"], co_batch, fa_batch)  # [1,D]
        embs.append(g.squeeze(0).cpu().numpy())
        ids.append(ds.files[i].stem)  # имена файлов как идентификаторы
    E = np.stack(embs).astype("float32")  # [N,D]
    # К косинусу: L2-нормировка (на всякий случай ещё раз)
    E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-8
    return ids, E


def search_by_name(
    ids: List[str],
    E: np.ndarray,
    query_name: str,
    include_self: bool = True,
    max_print: Optional[int] = 50,
) -> List[Tuple[int, float, str]]:
    """
    Находит по имени/подстроке все совпадения и печатает ранжированный список по косинусу для каждого совпадения.
    Возвращает последний сформированный список (если матчей несколько).
    """
    hits = _find_query_indices(ids, query_name)
    if len(hits) == 0:
        print(f"Запрос '{query_name}': совпадений не найдено")
        return []
    last_result: List[Tuple[int, float, str]] = []
    for idx in hits:
        print(f"\nЗапрос: {ids[idx]}")
        ranked = _rank_all_by_query(ids, E, idx, include_self=include_self)
        last_result = ranked
        # формат: номер, %, имя
        for rank, sim, name in (ranked if max_print is None else ranked[:max_print]):
            pct = round(100.0 * sim, 2)
            print(f"{rank:4d}. {pct:6.2f}%  {name}")
    return last_result

def topk_similar(E: np.ndarray, k: int = 10, include_self: bool = True) -> np.ndarray:
    S = _cosine_similarity_matrix(E)  # косинус при L2-нормировке
    if not include_self:
        np.fill_diagonal(S, -1.0)  # исключаем self только при анализе соседей
    topk_idx = np.argpartition(-S, kth=range(k), axis=1)[:, :k] # type: ignore
    row_sort = np.argsort(-S[np.arange(S.shape[0])[:, None], topk_idx], axis=1) # type: ignore
    return topk_idx[np.arange(S.shape[0])[:, None], row_sort]

def _pool_model_embedding(
    z_coedge: Tensor, z_face: Tensor,
    co_batch: Tensor, fa_batch: Tensor
) -> Tensor:
    zc_mean = global_mean_pool(z_coedge, co_batch)  # [B,Dc]
    zc_max  = global_max_pool(z_coedge, co_batch)   # [B,Dc]
    zf_mean = global_mean_pool(z_face,   fa_batch)  # [B,Df]
    z = torch.cat([zc_mean, zc_max, zf_mean], dim=1)  # [B,2*Dc+Df]
    z = torch.nn.functional.normalize(z, dim=-1)
    return z  # [B, D]

def _cosine_similarity_matrix(E: np.ndarray) -> np.ndarray:
    """
    Предполагается, что E уже L2-нормирован по строкам. Возвращает S=E@E^T.
    """
    return E @ E.T

def _find_query_indices(ids: List[str], query: str, case_insensitive: bool = True) -> List[int]:
    """
    Возвращает список индексов моделей, чьи имена содержат query (подстрока).
    """
    q = query.lower() if case_insensitive else query
    hits = []
    for i, name in enumerate(ids):
        s = name.lower() if case_insensitive else name
        if q in s:
            hits.append(i)
    return hits

def _rank_all_by_query(
    ids: List[str],
    E: np.ndarray,
    query_idx: int,
    include_self: bool = True,
) -> List[Tuple[int, float, str]]:
    """
    Ранжирует все объекты датасета по косинусной схожести с указанным query_idx.
    Возвращает список (rank, similarity, name) для всего набора.
    """
    assert 0 <= query_idx < len(ids)
    s = (E[query_idx:query_idx+1] @ E.T).ravel()            # [N]
    if not include_self:
        s[query_idx] = -1.0
    order = np.argsort(-s)                                  # type: ignore
    out: List[Tuple[int, float, str]] = []
    for rank, j in enumerate(order, start=1):
        sim = float(s[j])
        out.append((rank, sim, ids[j]))
    return out


