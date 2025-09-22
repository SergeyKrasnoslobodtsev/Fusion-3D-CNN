import numpy as np
from pathlib import Path
from typing import List, Dict

def l2norm(x: np.ndarray, axis: int = -1, eps: float = 1e-9) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.clip(n, eps, None)

def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a @ b.T  # (Na, D) x (D, Nb) -> (Na, Nb)

def object_id_from_path(p: Path) -> str:
    # gj 
    return p.stem  

def recall_at_k(ranks: List[int], K: int) -> float:
    # ranks — 1-based ранги релевантной цели; тут ровно 1 релевантная цель на запрос
    return np.mean([1.0 if r <= K else 0.0 for r in ranks]) if ranks else 0.0 # type: ignore

def mean_average_precision(ranks: List[int]) -> float:
    # 1 релевантный документ => AP = 1/rank
    ap = [1.0 / r for r in ranks]
    return float(np.mean(ap)) if ap else 0.0

def ndcg_at_k(ranks: List[int], K: int) -> float:
    # 1 релевантный документ: DCG = 1/log2(rank+1) если rank<=K, иначе 0; IDCG=1
    vals = []
    for r in ranks:
        if r <= K:
            vals.append(1.0 / np.log2(r + 1.0))
        else:
            vals.append(0.0)
    return float(np.mean(vals)) if vals else 0.0

def cohens_d(pos: np.ndarray, neg: np.ndarray) -> float:
    # Эффект-размер между позитивами (same object) и негативами (others)
    m1, m2 = np.mean(pos), np.mean(neg)
    s1, s2 = np.var(pos, ddof=1), np.var(neg, ddof=1)
    n1, n2 = len(pos), len(neg)
    if n1 < 2 or n2 < 2:
        return 0.0
    s_p = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    return (m1 - m2) / (s_p + 1e-9)

def eval_object_max(files: List[Path], load_fn, Ks=(1, 5, 10)) -> Dict[str, float]:
    """
    Query: (i, v) — вид v объекта i.
    Gallery: по объектам j != i, score = max по видам cos( query(i,v), feats_j[:] ).
    Ранг релевантного объекта = позиция i среди всех объектов по score (1-базовый).
    """
    N = len(files)
    obj_ids = [object_id_from_path(p) for p in files]
    feats_list = [l2norm(load_fn(p)) for p in files]  # каждый (V, D)
    V = feats_list[0].shape[0]

    ranks = []
    pos_vals, neg_vals = [], []

    # Преднормировать и заранее подготовить матрицы для ускорения
    # Пройдём по всем запросам (i, v)
    for i in range(N):
        Fi = feats_list[i]  # (V, D)
        for v in range(V):
            q = Fi[v:v+1, :]  # (1, D)

            # score по объекту j = max_v' cos(q, F_j[v'])
            scores = []
            for j in range(N):
                Fj = feats_list[j]  # (V, D)
                s = cosine_sim(q, Fj).max()  # скаляр
                scores.append(s)
            scores = np.asarray(scores)  # (N,)

            # отделяем позитив/негатив для анализа распределений
            pos_vals.append(scores[i])
            neg_vals.extend(np.delete(scores, i))

            # ранг релевантного объекта i среди всех объектов
            # (большее — лучше; rank 1 — идеал)
            order = np.argsort(-scores)  # убыв.
            rank = int(np.where(order == i)[0][0]) + 1  # 1-based
            ranks.append(rank)

    metrics = {
        "queries": N * V,
        "recall@1": recall_at_k(ranks, 1),
        "recall@5": recall_at_k(ranks, 5),
        "recall@10": recall_at_k(ranks, 10),
        "mAP": mean_average_precision(ranks),
        "nDCG@5": ndcg_at_k(ranks, 5),
        "nDCG@10": ndcg_at_k(ranks, 10),
        "pos_mean": float(np.mean(pos_vals)),
        "neg_mean": float(np.mean(neg_vals)),
        "margin": float(np.mean(pos_vals) - np.mean(neg_vals)),
        "cohens_d": float(cohens_d(np.asarray(pos_vals), np.asarray(neg_vals))),
    }
    return metrics