import numpy as np
from pathlib import Path
import pandas as pd

def _l2norm(x, axis=-1, eps=1e-9):
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.clip(n, eps, None)

def _cos_sim(A, B):
    # (Na, D) x (Nb, D) -> (Na, Nb)
    return A @ B.T

def _load_views_npz(p: Path) -> np.ndarray:
    data = np.load(p)
    views = data["views"]              # (V, D)
    return _l2norm(views, axis=1)

def _pairwise_aligned_score(q: np.ndarray, F: np.ndarray) -> float:
    m = min(q.shape[0], F.shape[0])
    if m == 0:
        return -1.0  # на всякий случай
    S = _cos_sim(q[:m], F[:m])       # (m, m)
    return float(np.mean(np.diag(S)))

def vit_topk(model_label: str, files, query, k: int = 10):
    # --- подготовка индекса ---
    files = list(files)
    if not files:
        print(f"Пустой список файлов для модели: {model_label}")
        return pd.DataFrame(columns=["model", "score"])

    names = [Path(f).stem for f in files]
    feats = [_load_views_npz(f) for f in files]   # список (Vi, D)

    # --- запрос ---
    if isinstance(query, int):
        q_name = names[query]
        q = feats[query]                          # (Vq, D)
    else:
        q_path = Path(query)
        q_name = q_path.stem
        q = _load_views_npz(q_path)               # (Vq, D)

    # --- поиск: score = max_{v,v'} cos ---
    scores = []
    for name, F in zip(names, feats):
        s = _pairwise_aligned_score(q, F)
        scores.append((name, float(s)))

    scores.sort(key=lambda t: t[1], reverse=True)
    top = scores[:k]


    # --- вывод ---
    print(f"Поисковая модель: {model_label}")
    # df = pd.DataFrame(top, columns=["model", "score"])
    # for i, (n, s) in enumerate(top, 1):
    #     print(f"{i:2d}. {n}   {s:.6f}")


    return pd.DataFrame(top, columns=["model", "score"])