from __future__ import annotations
import torch
from torch import Tensor
from torch_geometric.data import HeteroData


class CoedgeLCSNormalize:
    """
    Нормализация coedge.grid в локальной системе коэджа с учётом масштаба и reverse-флага.
    Ожидаемые поля:
      - data['coedge'].grid:  [C, 12, U]  (xyz | t | nL | nR)
      - data['coedge'].lcs:   [C,  4, 4]  (гомогенная матрица)
      - data['coedge'].scale: [C]
      - data['coedge'].reverse: [C] bool
    Возвращает обновлённый data с grid в LCS.
    """
    def __init__(self, apply_reverse: bool = True, renorm_vectors: bool = False) -> None:
        self.apply_reverse = apply_reverse
        self.renorm_vectors = renorm_vectors

    @torch.no_grad()
    def __call__(self, data: HeteroData) -> HeteroData:
        grid: Tensor = data["coedge"].grid           # [C,12,U]
        lcs: Tensor = data["coedge"].lcs            # [C,4,4]
        scale: Tensor = data["coedge"].scale        # [C]
        rev: Tensor = data["coedge"].reverse        # [C] bool

        assert grid.dim() == 3 and grid.size(1) == 12, f"grid shape {tuple(grid.shape)}"  # [C,12,U]
        C, _, U = grid.shape
        assert lcs.shape == (C, 4, 4), f"lcs shape {tuple(lcs.shape)}"
        assert scale.shape == (C,), f"scale shape {tuple(scale.shape)}"
        assert rev.shape == (C,), f"reverse shape {tuple(rev.shape)}"

        # Разбор каналов: работаем в [C,3,U]
        xyz = grid[:, 0:3, :]    # [C,3,U]
        tng = grid[:, 3:6, :]    # [C,3,U]
        nL  = grid[:, 6:9, :]    # [C,3,U]
        nR  = grid[:, 9:12, :]   # [C,3,U]

        # Из LCS берём R и t, и сразу строим R^{-1}, t^{-1}
        R = lcs[:, :3, :3]                     # [C,3,3]
        t = lcs[:, :3,  3]                     # [C,3]
        R_inv = R.transpose(1, 2).contiguous() # [C,3,3]
        t_inv = -(R_inv @ t.unsqueeze(-1)).squeeze(-1)  # [C,3]

        # Перенос в начало LCS: xyz' = R^{-1} (xyz - t)
        xyz_local = xyz - t.unsqueeze(-1)                      # [C,3,U]
        xyz_local = torch.einsum('cij,cju->ciu', R_inv, xyz_local)  # [C,3,U]

        # Поворот направлений (нормали/касательные не масштабируем по длине)
        tng_local = torch.einsum('cij,cju->ciu', R_inv, tng)   # [C,3,U]
        nL_local  = torch.einsum('cij,cju->ciu', R_inv, nL)    # [C,3,U]
        nR_local  = torch.einsum('cij,cju->ciu', R_inv, nR)    # [C,3,U]

        # Масштаб (делим координаты, векторы оставляем единичными)
        s = scale.view(C, 1, 1).clamp_min(1e-8)                # [C,1,1]
        xyz_local = xyz_local / s

        # Опциональная нормализация направлений
        if self.renorm_vectors:
            def _renorm(v: Tensor) -> Tensor:
                n = torch.linalg.norm(v, dim=1, keepdim=True).clamp_min(1e-8)
                return v / n
            tng_local = _renorm(tng_local)
            nL_local  = _renorm(nL_local)
            nR_local  = _renorm(nR_local)

        # Обработка реверса: разворот вдоль U, смена знака касательной, перестановка nL/nR
        if self.apply_reverse and rev.any():
            idx = torch.nonzero(rev, as_tuple=False).flatten()
            xyz_local[idx] = torch.flip(xyz_local[idx], dims=[-1])
            tng_local[idx] = -torch.flip(tng_local[idx], dims=[-1])
            nL_flip = torch.flip(nR_local[idx], dims=[-1])
            nR_flip = torch.flip(nL_local[idx], dims=[-1])
            nL_local[idx], nR_local[idx] = nL_flip, nR_flip

        data["coedge"].grid = torch.cat([xyz_local, tng_local, nL_local, nR_local], dim=1).contiguous()  # [C,12,U]
        return data