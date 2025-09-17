"""Утилиты масштабирования модели.

Функции в этом модуле помогают вычислять ограничивающий бокс тела и
масштабировать тело в единичный куб [-1, 1]^3. Используются API из
pythonocc (OCC) и occwl. Некоторые вызовы occwl/pythonocc сложно выразить
строго через typing, поэтому в местах с сомнительной типизацией добавлены
директивы `# type: ignore`.
"""
from typing import Any

from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.gp import gp_Pnt, gp_Vec, gp_Trsf
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform

# occwl - оболочка удобная для операций над телами
from occwl.solid import Solid  # type: ignore
from occwl.compound import Compound  # type: ignore

from .create_occwl_from_occ import create_occwl


def find_box(solid: Any) -> Bnd_Box:
    """Вычисляет оптимальный ограничивающий бокс тела.

    Args:
        solid: объект тела (TopoDS_Shape или occwl.Solid/Compound)

    Returns:
        Bnd_Box: ограничивающий бокс в координатах тела.
    """
    bbox = Bnd_Box()
    # Use triangulation для более точного бокса при необходимости
    use_triangulation = True
    use_shapetolerance = False
    # brepbndlib.AddOptimal создаёт оптимальный бокс
    brepbndlib.AddOptimal(solid, bbox, use_triangulation, use_shapetolerance)
    return bbox


def scale_solid_to_unit_box(solid: Any) -> Any:
    """Масштабирует тело так, чтобы оно поместилось в куб [-1,1]^3.

    Возвращаемый тип зависит от входа: если передан `occwl.Solid`, возвращается
    `occwl.Solid` (скопированный и масштабированный). Если передан TopoDS_Shape
    (OCC), функция конвертирует его в occwl.Solid, масштабирует и возвращает
    обратно TopoDS_Shape.

    Args:
        solid: входное тело (occwl.Solid или TopoDS_Shape)

    Returns:
        Масштабированное тело (тип совпадает с описанным выше).
    """
    # Если это уже occwl.Solid, используем встроенный метод
    if isinstance(solid, Solid):
        # copy=True чтобы не изменять исходный объект
        return solid.scale_to_unit_box(copy=True)

    # Если это TopoDS_Shape (или другой OCC-объект), преобразуем в occwl.Solid
    occwl_solid = create_occwl(solid) 
    scaled = occwl_solid.scale_to_unit_box(copy=True)  # type: ignore
    # Возвращаем обратно TopoDS_Shape
    return scaled.topods_shape() 



