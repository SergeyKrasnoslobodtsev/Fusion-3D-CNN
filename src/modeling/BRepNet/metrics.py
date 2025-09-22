import numpy as np
from pathlib import Path
from typing import Dict, List

def _l2norm(x: np.ndarray, axis: int = -1, eps: float = 1e-9) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.clip(n, eps, None)

def _cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return _l2norm(a) @ _l2norm(b).T

def _recall_at_k(ranks: List[int], K: int) -> float:
    return float(np.mean([r <= K for r in ranks])) if ranks else 0.0

def _map_single_rel(ranks: List[int]) -> float:
    # один релевантный объект на запрос → AP = 1/rank
    return float(np.mean([1.0 / r for r in ranks])) if ranks else 0.0

def _ndcg_at_k(ranks: List[int], K: int) -> float:
    vals = []
    for r in ranks:
        vals.append(1.0 / np.log2(r + 1.0) if r <= K else 0.0)
    return float(np.mean(vals)) if vals else 0.0

def _cohens_d(pos: np.ndarray, neg: np.ndarray) -> float:
    if len(pos) < 2 or len(neg) < 2:
        return 0.0
    m1, m2 = pos.mean(), neg.mean()
    v1, v2 = pos.var(ddof=1), neg.var(ddof=1)
    n1, n2 = len(pos), len(neg)
    s_p = np.sqrt(((n1-1)*v1 + (n2-1)*v2) / (n1+n2-2))
    return float((m1 - m2) / (s_p + 1e-9))

def evaluate_brepnet_faces_object_max(
    faces_dir: Path,
    model_dir: Path,
    Ks=(1,5,10)
) -> Dict[str, float]:
    """
    Query: каждая грань (face) объекта i — вектор f ∈ R^D.
    Gallery по объектам j: score_j = max_v' cos( f, face_j[v'] )  (object-max).
    Позитив: свой объект i. Ранг — позиция i среди всех объектов.
    """
    # загрузим модели и список имён
    model_files = sorted(model_dir.glob("*_model.npy"))
    names = [f.stem.replace("_model", "") for f in model_files]
    name2idx = {n: k for k, n in enumerate(names)}

    # проверим, что для каждого name есть faces
    face_files = []
    for n in names:
        ff = faces_dir / f"{n}_faces.npy"
        if not ff.exists():
            raise FileNotFoundError(
                f"Нет локальных эмбеддингов для '{n}': {ff} не найден. "
                f"Экспортируй faces (F,D), чтобы посчитать метрики."
            )
        face_files.append(ff)

    # загрузим все лица по объектам
    faces_list = [np.load(ff) for ff in face_files]       # список [ (F_i, D), ... ]
    faces_list = [ _l2norm(F) for F in faces_list ]
    # предвычислим "object-max" по каждому объекту для ускорения — просто оставим списком

    ranks = []
    pos_vals, neg_vals = [], []

    # для каждой грани каждого объекта считаем ранги по объектам
    for i, Fi in enumerate(faces_list):
        for q in Fi:  # q: (D,)
            q = q[None, :]  # (1,D)
            scores = []
            for Fj in faces_list:
                # max по всем лицам объекта j
                s = _cos(q, Fj).max()
                scores.append(s)
            scores = np.asarray(scores)  # (N,)

            pos_vals.append(scores[i])
            if len(scores) > 1:
                neg_vals.extend(np.delete(scores, i))

            order = np.argsort(-scores)
            rank = int(np.where(order == i)[0][0]) + 1
            ranks.append(rank)

    pos_vals = np.asarray(pos_vals)
    neg_vals = np.asarray(neg_vals)

    metrics = {
        "queries": len(ranks),
        "recall@1": _recall_at_k(ranks, 1),
        "recall@5": _recall_at_k(ranks, 5),
        "recall@10": _recall_at_k(ranks, 10),
        "mAP": _map_single_rel(ranks),
        "nDCG@5": _ndcg_at_k(ranks, 5),
        "nDCG@10": _ndcg_at_k(ranks, 10),
        "pos_mean": float(pos_vals.mean()) if len(pos_vals) else 0.0,
        "neg_mean": float(neg_vals.mean()) if len(neg_vals) else 0.0,
        "margin": float(pos_vals.mean() - neg_vals.mean()) if len(pos_vals) and len(neg_vals) else 0.0,
        "cohens_d": _cohens_d(pos_vals, neg_vals),
    }
    return metrics