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
    view1 = dropout_attrs(data, p=p1)
    view2 = dropout_attrs(data, p=p2)

    return view1, view2




# def get_main_shape(face_features): new data["face"].x
#     is_plane = face_features[:, 0]
#     is_cylinder = face_features[:, 1]
#     is_cone = face_features[:, 2]
#     is_sphere = face_features[:, 3]
#     is_torus = face_features[:, 4]
#     areas = face_features[:, 5]

#     total_area = areas.sum()
#     plane_area = areas[is_plane > 0.0].sum()
#     cylinder_area = areas[is_cylinder > 0.0].sum()
#     cone_area = areas[is_cone > 0.0].sum()  
#     sphere_area = areas[is_sphere > 0.0].sum()
#     torus_area = areas[is_torus > 0.0].sum()

#     # Основная форма — по максимальной доле площади
#     shape_areas = {
#         "prismatic": plane_area, # плоские и призматические
#         "cylindrical": cylinder_area, # цилиндрические
#         "conical": cone_area, # конические
#         "spherical": sphere_area, # сферические
#         "toroidal": torus_area # тороидальные
#     }
#     main_shape = max(shape_areas, key=lambda k: shape_areas[k])
#     if shape_areas[main_shape] / total_area < 0.5:
#         return "mixed"
#     return main_shape

# # Для всего датасета
# shape_labels = [get_main_shape(item['brepnet_features']) for item in dataset]

# # Пример: вывести все цилиндрические детали
# for idx, label in enumerate(shape_labels):
#     if label == "prismatic":
#         print(f"{idx}: {dataset[idx]['model_id']} | {label}")