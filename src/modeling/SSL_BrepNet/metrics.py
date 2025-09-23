import numpy as np
from pathlib import Path
from typing import List, Dict
from tqdm.auto import tqdm

def l2norm(x: np.ndarray, axis: int = -1, eps: float = 1e-9) -> np.ndarray:
    if x.size == 0:
        return x
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.clip(n, eps, None)

def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.array([])
    return a @ b.T

def _euclidean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # ||a-b||^2 = ||a||^2 + ||b||^2 - 2<a,b>
    a2 = np.sum(a * a, axis=1, keepdims=True)  # (Na,1)
    b2 = np.sum(b * b, axis=1, keepdims=True)  # (Nb,1)
    ab = a @ b.T                              # (Na,Nb)
    d2 = a2 + b2.T - 2 * ab                    # (Na,Nb)
    return np.sqrt(np.clip(d2, 0.0, None))     # (Na,Nb)

def recall_at_k(ranks: List[int], K: int) -> float:
    return np.mean([1.0 if r <= K else 0.0 for r in ranks]) if ranks else 0.0 # type: ignore

def mean_average_precision(ranks: List[int]) -> float:
    ap = [1.0 / r for r in ranks]
    return float(np.mean(ap)) if ap else 0.0

def ndcg_at_k(ranks: List[int], K: int) -> float:
    vals = [1.0 / np.log2(r + 1.0) if r <= K else 0.0 for r in ranks]
    return float(np.mean(vals)) if vals else 0.0

def cohens_d(pos: np.ndarray, neg: np.ndarray) -> float:
    m1, m2 = np.mean(pos), np.mean(neg)
    s1, s2 = np.var(pos, ddof=1), np.var(neg, ddof=1)
    n1, n2 = len(pos), len(neg)
    if n1 < 2 or n2 < 2:
        return 0.0
    s_p = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    return (m1 - m2) / (s_p + 1e-9)

def eval_object_max(files: List[Path], load_fn) -> Dict[str, float]:
    N = len(files)
    if N == 0:
        print("Warning: No embedding files found.")
        return {}

    feats_list = [l2norm(load_fn(p)) for p in tqdm(files, desc="Loading embeddings")]

    ranks = []
    pos_vals, neg_vals = [], []
    total_queries = 0

    for i in tqdm(range(N), desc="Evaluating queries"):
        Fi = feats_list[i]
        num_query_faces = Fi.shape[0]
        
        if num_query_faces == 0:
            continue

        total_queries += num_query_faces

        q = feats_list[i].reshape(1, -1)
        # q = Fi[v:v+1, :]
        scores = []
        for j in range(N):
            Fj = feats_list[j]
            sim_values = cosine_sim(q, Fj)
            s = sim_values.max() if sim_values.size > 0 else -1.0
            scores.append(s)
        scores = np.asarray(scores)

        pos_vals.append(scores[i])
        neg_vals.extend(np.delete(scores, i))

        order = np.argsort(-scores)
        rank = int(np.where(order == i)[0][0]) + 1
        ranks.append(rank)

    # Приводим словарь к требуемому формату
    metrics = {
        "queries": total_queries,
        "recall@1": recall_at_k(ranks, 1),
        "recall@5": recall_at_k(ranks, 5),
        "recall@10": recall_at_k(ranks, 10),
        "mAP": mean_average_precision(ranks),
        "nDCG@5": ndcg_at_k(ranks, 5),
        "nDCG@10": ndcg_at_k(ranks, 10),
        "pos_mean": float(np.mean(pos_vals)) if pos_vals else 0.0,
        "neg_mean": float(np.mean(neg_vals)) if neg_vals else 0.0,
        "margin": (np.mean(pos_vals) - np.mean(neg_vals)) if pos_vals and neg_vals else 0.0,
        "cohens_d": cohens_d(np.asarray(pos_vals), np.asarray(neg_vals)),
    }
    return metrics