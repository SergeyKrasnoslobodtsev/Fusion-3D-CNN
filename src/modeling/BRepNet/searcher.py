from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd

def _emb_path(dir_: Path, stem: str) -> Path:
    p = dir_ / f"{stem}.embeddings"
    if p.exists():
        return p
    p = dir_ / f"{stem}_faces.npy"
    if p.exists():
        return p
    raise FileNotFoundError(f"Не найдено: {stem}.embeddings или {stem}_faces.npy в {dir_}")

def _load_mat(p: Path) -> np.ndarray:
    E = np.loadtxt(p) if p.suffix == ".embeddings" else np.load(p)
    return E[None, :] if E.ndim == 1 else E  # [F, D]; БЕЗ нормализации

def _summin_euclid(Q: np.ndarray, T: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    # Q: [Fq, D], T: [Ft, D]
    # попарные расстояния по евклиду
    diff = Q[:, None, :] - T[None, :, :]           # [Fq, Ft, D]
    d2   = np.sum(diff * diff, axis=-1)            # [Fq, Ft]
    jmin = np.argmin(d2, axis=1)                   # [Fq] индекс ближайшей t-граня для каждой q
    dmin = np.sqrt(d2[np.arange(Q.shape[0]), jmin])# [Fq] минимальные расстояния
    score = float(dmin.sum())                      # твоя метрика (сумма)
    return score, dmin, jmin

def search_topk(emb_dir: Path, query_stem: str, k: int = 20, exclude_self: bool = True):
    # собрать список файлов каталога
    files = sorted(emb_dir.glob("*.embeddings"))

    stems = [f.stem for f in files]

    # загрузить запрос
    Q = _load_mat(_emb_path(emb_dir, query_stem))

    results = []
    for stem, f in zip(stems, files):
        if exclude_self and stem == query_stem:
            continue
        T = _load_mat(f)
        score, dmin, jmin = _summin_euclid(Q, T)
        # матчи: [q_idx, t_idx, dist] — чтобы подсветить пары граней
        matches = np.stack([np.arange(len(dmin)), jmin, dmin], axis=1)
        results.append((stem, score, dmin, matches))

    # выбрать k наименьших (меньше — лучше)
    if k < len(results):
        scores = np.array([r[1] for r in results])
        idx = np.argpartition(scores, k-1)[:k]   # корректный partial top-k
        results = [results[i] for i in idx]

    results.sort(key=lambda x: x[1])  # финальная сортировка по score ↑

    # общий интервал для heatmap по топ-K
    all_d = np.concatenate([r[2] for r in results]) if results else np.array([0.0])
    interval = (float(all_d.min()), float(all_d.max()))

    top = []
    matches_list = []
    dmin_list = []
    for stem, score, dmin, matches in results:
        top.append({"model": stem, "score": float(score)})
        dmin_list.append(dmin)
        matches_list.append(matches)

    return pd.DataFrame(top, columns=["model", "score"]), matches_list, dmin_list, interval