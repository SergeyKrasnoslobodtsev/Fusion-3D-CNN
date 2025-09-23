from __future__ import annotations
from typing import Tuple
import torch
from torch_geometric.data import HeteroData


def get_main_shape(data: HeteroData) -> str:
    face_x = data["face"].x  # [num_faces, 7]
    is_plane = face_x[:, 0]
    is_cylinder = face_x[:, 1]
    is_cone = face_x[:, 2]
    is_sphere = face_x[:, 3]
    is_torus = face_x[:, 4]
    areas = face_x[:, 5]

    total_area = areas.sum()
    plane_area = areas[is_plane > 0.0].sum()
    cylinder_area = areas[is_cylinder > 0.0].sum()
    cone_area = areas[is_cone > 0.0].sum()
    sphere_area = areas[is_sphere > 0.0].sum()
    torus_area = areas[is_torus > 0.0].sum()

    shape_areas = {
        "prismatic": plane_area,
        "cylindrical": cylinder_area,
        "conical": cone_area,
        "spherical": sphere_area,
        "toroidal": torus_area
    }
    main_shape = max(shape_areas, key=lambda k: shape_areas[k])
    if total_area == 0 or shape_areas[main_shape] / total_area < 0.5:
        return "mixed"
    return main_shape


def dropout_attrs(data: HeteroData, p: float = 0.1) -> HeteroData:
    """
    Простейшая атрибутивная аугментация: зануляем случайные элементы в grid/uv.
    Безопасно для топологии (edge_index не трогаем).
    """
    out = data.clone()
    if "uv" in out["face"]:
        uv = out["face"].uv
        mask = torch.rand_like(uv) < p
        out["face"].uv = uv.masked_fill(mask, 0.0)
    grid = out["coedge"].grid
    mask = torch.rand_like(grid) < p
    out["coedge"].grid = grid.masked_fill(mask, 0.0)
    return out

import torch

def random_rotate_grid(grid: torch.Tensor) -> torch.Tensor:
    # grid: [C, 12, 10] или [F, 7, 10, 10]
    theta = torch.rand(1) * 2 * torch.pi
    rot = torch.tensor([
        [torch.cos(theta), -torch.sin(theta), 0],
        [torch.sin(theta), torch.cos(theta), 0],
        [0, 0, 1]
    ], device=grid.device)
    # Применяем к координатам (предположим, первые 3 канала — xyz)
    grid[:, 0:3, :] = torch.matmul(rot, grid[:, 0:3, :])
    return grid

def strong_augment(data: HeteroData, p: float = 0.2) -> HeteroData:
    out = data.clone()
    if "uv" in out["face"]:
        if torch.rand(1) < p:
            out["face"].uv = random_rotate_grid(out["face"].uv)
    if "grid" in out["coedge"]:
        if torch.rand(1) < p:
            out["coedge"].grid = random_rotate_grid(out["coedge"].grid)
    # Dropout по признакам
    if torch.rand(1) < p:
        mask = torch.rand_like(out["face"].x) < 0.1
        out["face"].x = out["face"].x * (~mask)
    if torch.rand(1) < p:
        mask = torch.rand_like(out["edge"].x) < 0.1
        out["edge"].x = out["edge"].x * (~mask)
    if torch.rand(1) < p:
        mask = torch.rand_like(out["coedge"].x) < 0.1
        out["coedge"].x = out["coedge"].x * (~mask)
    return out

def two_views(data: HeteroData, p: float = 0.1) -> Tuple[HeteroData, HeteroData]:
    """
    Создаёт два аугментированных представления графа.
    Стратегия аугментации зависит от основной формы модели.
    """
    main_shape = get_main_shape(data)

    if main_shape == "prismatic":
        # Более агрессивная аугментация для простых форм
        p1 = p * 1.5
        p2 = p * 1.5
    elif main_shape in ("cylindrical", "conical", "spherical", "toroidal"):
        # Стандартная аугментация для типовых форм
        p1 = p
        p2 = p
    else:  # mixed
        # Более слабая аугментация для сложных или смешанных форм
        p1 = p * 0.5
        p2 = p * 0.5

    # Применяем dropout с вычисленными вероятностями
    view1 = strong_augment(data, p=p1)
    view2 = strong_augment(data, p=p2)

    return view1, view2
