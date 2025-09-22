import argparse
import torch
from typing import Any
from pathlib import Path
import numpy as np
import json
import torch.nn.functional as F
from typing import Any

from .model.brepnet import BRepNet

def get_argument_parser():
    parser = argparse.ArgumentParser()

    parser = BRepNet.add_model_specific_args(parser)
    parser.add_argument("--model", type=str, help="Model to load use for evaluation")
    return parser

class BRepNetEmbeddingExtractor:
    def __init__(self, 
                 checkpoint_path: Path, 
                 feature_standardization: Path,
                 segment_files: Path, 
                 kernel_file: Path, 
                 features_list: Path):
        self.opts = get_argument_parser().parse_args([
            "--dataset_file", str(feature_standardization),
            "--segment_names", str(segment_files),
            "--dataset_dir", str(checkpoint_path.parent),
            "--kernel", str(kernel_file),
            "--model", str(checkpoint_path),
            "--num_classes", "8",
            "--input_features", str(features_list)
        ])
        self.model: BRepNet = BRepNet.load_from_checkpoint(str(checkpoint_path), opts=self.opts)
        self.model.eval()
        

    def extract_from_npz(self, npz_path: Path, pool: str = "mean") -> tuple[torch.Tensor, torch.Tensor]:
        """Возвращает (face_embeds [Nf, D], model_embed [D])"""
        T = self.tensors_from_npz(npz_path)
        face_embeds = self.extract_face_embeddings( 
            T["Xf"], T["Gf"], T["Xe"], T["Ge"], T["Xc"], T["Gc"], 
            T["Kf"], T["Ke"], T["Kc"], T["Ce"], T["Cf"], T["Csf"]  
        )  # [num_faces, D]
        # Глобальный вектор модели (простой вариант: mean + L2-нормировка)
        if pool == "max":
            z = face_embeds.max(dim=0).values
        else:  # "mean"
            z = face_embeds.mean(dim=0)
        z = F.normalize(z, dim=0)
        return face_embeds, z

    def extract_face_embeddings(
        self,
        Xf: torch.Tensor,
        Gf: torch.Tensor,
        Xe: torch.Tensor,
        Ge: torch.Tensor,
        Xc: torch.Tensor,
        Gc: torch.Tensor,
        Kf: torch.Tensor,
        Ke: torch.Tensor,
        Kc: torch.Tensor,
        Ce: torch.Tensor,
        Cf: torch.Tensor,
        Csf: Any
    ) -> torch.Tensor:
        with torch.no_grad():
            embeddings: torch.Tensor = self.model.create_face_embeddings(
                Xf, Gf, Xe, Ge, Xc, Gc, Kf, Ke, Kc, Ce, Cf, Csf
            )
        return embeddings
    
    def tensors_from_npz(self, npz_path: Path):
        """
        Загрузка массивов из NPZ по ТВОЕЙ схеме ключей + стандартизация табличных
        признаков и построение недостающих топологических тензоров под BRepNet.
        """

        npz = np.load(npz_path, allow_pickle=True)

        device = next(self.model.parameters()).device
        to_f32 = lambda a: torch.from_numpy(a.astype(np.float32)).to(device)
        to_i64 = lambda a: torch.from_numpy(a.astype(np.int64)).to(device)

        # ---- 1) Признаки и гриды из твоего NPZ ----
        Xf = to_f32(npz["face_features"])                # [F, ?]
        Xe = to_f32(npz["edge_features"])                # [E, ?]
        Xc = to_f32(npz["coedge_features"])              # [C, 1]

        Gf = to_f32(npz["face_point_grids"])             # [F, 7, 10, 10]
        Gc = to_f32(npz["coedge_point_grids"])           # [C, 12, 10]

        # Edge grids в твоих NPZ нет — создаём нулевые (не используются, если use_edge_grids=0)
        E = Xe.shape[0]
        Ge = torch.zeros((E, 12, 10), dtype=torch.float32, device=device)

        # ---- 2) Топология из твоего NPZ ----
        coedge_to_face = npz["face"].astype(np.int64)    # [C]
        coedge_to_edge = npz["edge"].astype(np.int64)    # [C]
        next_idx       = npz["next"].astype(np.int64)    # [C]
        mate_idx       = npz["mate"].astype(np.int64)    # [C]

        C = int(Xc.shape[0])
        F = int(Xf.shape[0])
        E_n = int(Xe.shape[0])

        # ---- 3) Ce: по 2 коэджа на ребро (max-pool по ребру) ----
        ce_rows = []
        for e in range(E_n):
            idxs = np.where(coedge_to_edge == e)[0]
            if len(idxs) >= 2:
                ce_rows.append(idxs[:2])
            elif len(idxs) == 1:
                # дублируем один коэдж, чтобы форма была [E,2]
                ce_rows.append([idxs[0], idxs[0]])
            else:
                ce_rows.append([0, 0])
        Ce = to_i64(np.vstack(ce_rows))                  # [E, 2]

        # ---- 4) Cf/Csf: все лица считаем "маленькими", паддим до максимума ----
        face_to_coedges = [[] for _ in range(F)]
        for ci, f in enumerate(coedge_to_face):
            face_to_coedges[int(f)].append(int(ci))
        max_coedges = max((len(lst) for lst in face_to_coedges), default=1)
        Cf_np = np.full((F, max_coedges), fill_value=C, dtype=np.int64)  # паддинг индексом C
        for f, lst in enumerate(face_to_coedges):
            Cf_np[f, :min(len(lst), max_coedges)] = lst[:max_coedges]
        Cf = to_i64(Cf_np)                               # [F, max_coedges]
        Csf = []                                         # big faces не используем

        # ---- 5) Kf/Ke/Kc: подгоняем размеры под kernel и заполняем "self" (+ next/mate для Kc) ----
        with open(self.opts.kernel, "r", encoding="utf-8") as fh:
            kernel_def = json.load(fh)
        
        kf_cols = len(kernel_def["faces"])
        ke_cols = len(kernel_def["edges"])
        kc_cols = len(kernel_def["coedges"])

        Kf_np = np.empty((C, kf_cols), dtype=np.int64)
        Ke_np = np.empty((C, ke_cols), dtype=np.int64)
        Kc_np = np.empty((C, kc_cols), dtype=np.int64)

        # prev по массиву next (обратное соответствие)
        prev_idx = np.full(C, -1, dtype=np.int64)
        for i, n in enumerate(next_idx):
            prev_idx[n] = i

        for ci in range(C):
            # лица/рёбра: просто текущие индексы, продублированные на всю ширину ядра
            Kf_np[ci, :] = coedge_to_face[ci]
            Ke_np[ci, :] = coedge_to_edge[ci]

            # коэджи: стараемся заполнить [self, next, prev, mate, ...], остаток — self
            seq = [ci]
            if kc_cols > 1: seq.append(int(next_idx[ci]))
            if kc_cols > 2: seq.append(int(prev_idx[ci]) if prev_idx[ci] >= 0 else ci)
            if kc_cols > 3: seq.append(int(mate_idx[ci]))
            if kc_cols > 4:
                # добиваем "self" до нужной длины
                seq.extend([ci] * (kc_cols - len(seq)))
            Kc_np[ci, :] = np.array(seq[:kc_cols], dtype=np.int64)

        Kf = to_i64(Kf_np)                               # [C, kf_cols]
        Ke = to_i64(Ke_np)                               # [C, ke_cols]
        Kc = to_i64(Kc_np)                               # [C, kc_cols]


        with open(self.opts.dataset_file, "r", encoding="utf-8") as fh:
            ds = json.load(fh)
        stats = ds.get("feature_standardization") or ds.get("feature_normalization")


        Xf = self._apply_stats(Xf, stats["face_features"])
        Xe = self._apply_stats(Xe, stats["edge_features"])
        Xc = self._apply_stats(Xc, stats["coedge_features"])

        return dict(
            Xf=Xf, Xe=Xe, Xc=Xc,
            Gf=Gf, Ge=Ge, Gc=Gc,
            Kf=Kf, Ke=Ke, Kc=Kc,
            Ce=Ce, Cf=Cf, Csf=torch.tensor(Csf, dtype=torch.int64, device=device) if isinstance(Csf, list) else Csf
        )

    # def _load_standardization(self, path: Path) -> Dict[str, Any]:
    #     """Чтение JSON с 'feature_standardization' (mean/std) или 'feature_normalization' (min/max)."""
    #     with open(path, "r", encoding="utf-8") as f:
    #         data = json.load(f)
    #     if "feature_standardization" in data:
    #         return {"mode": "standardize", **data["feature_standardization"]}
    #     if "feature_normalization" in data:
    #         return {"mode": "normalize", **data["feature_normalization"]}
    #     # Фолбэк — без преобразований
    #     return {"mode": "none"}

    def _apply_stats(self, X: torch.Tensor, stats_block: list) -> torch.Tensor:
        mean = torch.tensor([f["mean"] for f in stats_block], dtype=X.dtype, device=X.device)
        std = torch.tensor([f["standard_deviation"] for f in stats_block], dtype=X.dtype, device=X.device)
        return (X - mean) / (std + 1e-8)