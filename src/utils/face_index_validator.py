"""Модуль проверки соответствия индексов граней STEP-файла данным меша Fusion.

Утилита сравнивает информацию о гранях, закодированную цветом в STEP (Fusion),
с треугольной разбиением (OBJ) и файлом индексов граней (.fidx).

Основная логика проверки:
    - Сравнить количество граней в Fusion (по .fidx) и в STEP.
    - Проверить, совпадают ли индексы граней (кодированные в цвете) с порядком граней в STEP.
    - Если индексы/цвета не совпадают, выполнить перекрёстную проверку по ограничивающему
        боксу треугольников (Bnd_Box) для каждой грани и отвергнуть модель при значительной
        расхождении.

Файл содержит многочисленные точки взаимодействия с библиотекой pythonocc (OCC).
Некоторые места явно помечены `# type: ignore` для подавления ложных предупреждений
анализаторов типов, т.к. типы OCC сложно корректно выразить для статического анализатора.
"""

import igl
import numpy as np
import math

from typing import List, Dict, Optional, Tuple, Any

from pathlib import Path

from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.BRep import BRep_Tool
from OCC.Core.gp import gp_Pnt
from OCC.Extend.TopologyUtils import TopologyExplorer
from OCC.Core.TopLoc import TopLoc_Location

from OCC.Core.STEPCAFControl import STEPCAFControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.TDocStd import TDocStd_Document
from OCC.Core.TCollection import TCollection_ExtendedString
from OCC.Core.TDF import TDF_LabelSequence

from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB
from OCC.Core.XCAFDoc import XCAFDoc_DocumentTool

from OCC.Core.TopAbs import (TopAbs_VERTEX, TopAbs_EDGE, TopAbs_FACE, TopAbs_WIRE,
                             TopAbs_SHELL, TopAbs_SOLID, TopAbs_COMPOUND,
                             TopAbs_COMPSOLID, TopAbs_FORWARD)

# Импортируем конкретные классы TopoDS для аннотаций типов; анализатор типов может
# жаловаться на них, поэтому помечаем как ignore.
from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Face  # type: ignore

class FaceIndexValidator:
    """Класс для валидации соответствия индексов граней между STEP и Fusion (OBJ/.fidx).

    Атрибуты:
        step_file: Путь к STEP-файлу.
        mesh_dir: Директория с OBJ и .fidx файлами (Fusion).
    """
    def __init__(self, step_file: Path, mesh_dir: Path) -> None:
        self.step_file = step_file
        self.mesh_dir = mesh_dir

    def validate(self) -> bool:
        """Выполняет полную проверку модели.

        Возвращает True, если все проверки пройдены. В противном случае — False.
        """
        face_boxes = self.find_face_boxes(self.step_file.stem)
        if face_boxes is None:
            print(f"{self.step_file.stem} missing face")
            return False
     
        parts, face_map = self.load_parts_and_fusion_indices_step_file(self.step_file)
        if len(parts) != 1:
            print(f"{self.step_file} has {len(parts)} parts")
            return False

        # Для сопоставления данных Fusion и STEP необходимо одинаковое количество граней
        if not len(face_boxes) == len(face_map):
            print(f"In fusion {len(face_boxes)} faces.  In step {len(face_map)} faces")
            return False

        for part in parts:
            if not self.check_part(part, face_map, face_boxes):
                return False

        return True


    def check_part(self, part: Any, face_map: Dict[Any, int], face_boxes: List[Bnd_Box]) -> bool:
        """
        Проверяет часть модели на соответствие карте граней и боксам.

        Args:
            part (Any): Часть модели (TopoDS_Shape из OCC).
            face_map (Dict[Any, int]): Словарь, сопоставляющий грани с индексами.
            face_boxes (List[Bnd_Box]): Список боксов для граней из Fusion.

        Returns:
            bool: True, если проверка прошла, иначе False.
        """

        # Нам нужно вычислить треугольники для части, иначе бокс поверхности будет
        # слишком большим (он бы покрывал всю деталь). Здесь мы явно сетим шаг
        # сетки = 0.1 (см. масштабирование в get_face_triangles).
        mesh = BRepMesh_IncrementalMesh(part, 0.1, True) 
        mesh.Perform() 

        top_exp = TopologyExplorer(part)  
        faces = top_exp.faces()
        face_index_ok = True
        for face_idx, face in enumerate(faces):
            # Найти бокс треугольников, масштабированный к тем же единицам,
            # что и данные меша
            # Получаем Bnd_Box для треугольной апроксимации грани.
            bscaled = self.get_box_from_tris(face)

            # В face_map ключами являются объекты граней (TopoDS_Face) — используем
            # проверку наличия по ссылке.
            if face not in face_map:
                print("Face missing from face map")
                return False

            fusion_face_index = face_map[face]
            
            if fusion_face_index != face_idx:
                # Если индекс, восстановленный из цвета, не совпадает с порядком
                # грани в STEP, помечаем флаг и выполним перекрёстную проверку по боксам.
                face_index_ok = False

            if not face_index_ok:
                fusion_face_box = face_boxes[face_idx]
                if fusion_face_box.IsVoid():
                    print("fusion_face_box is void")
                    return False
                
                if bscaled.IsVoid():
                    print("bscaled box is void")
                    return False

                diag = math.sqrt(bscaled.SquareExtent())
                box_check_ok = self.check_box(fusion_face_box, bscaled, diag/10, "Error exceeds 1/10 of face box")
                if not box_check_ok:
                    print(f"Face index and color do not agree and box check fails!")
                    return False
        return True


    def get_obj_pathname(self, basename: str) -> Path:
        """
        Получает путь к OBJ-файлу для меша Fusion.

        Args:
            basename (str): Базовое имя файла без расширения.

        Returns:
            Path: Путь к OBJ-файлу.
        """
        return  (self.mesh_dir / basename).with_suffix(".obj")

        
    def get_fidx_pathname(self, basename: str) -> Path:
        """
        Получает путь к FIDX-файлу, содержащему индексы граней для каждого треугольника в меше.

        Args:
            basename (str): Базовое имя файла без расширения.

        Returns:
            Path: Путь к FIDX-файлу.
        """
        return (self.mesh_dir / basename).with_suffix(".fidx")


    def get_face_triangles(self, face: TopoDS_Shape) -> Tuple[np.ndarray, np.ndarray]:
        """
        Получает треугольники и вершины для данной грани.

        Args:
            face (Any): Грань модели (TopoDS_Face из OCC).

        Returns:
            Tuple[np.ndarray, np.ndarray]: Кортеж из массива вершин и массива треугольников.
        """
        tris: List[List[int]] = []
        verts: List[List[float]] = []

        # Определяем, соответствует ли ориентация грани нормали поверхности.
        # TopAbs_FORWARD означает, что ориентация совпадает с нормалью поверхности.
        face_orientation_wrt_surface_normal = (face.Orientation() == TopAbs_FORWARD)

        # Triangulation возвращается через BRep_Tool.Triangulation; типизация OCC
        # сложна для статического анализатора, поэтому используем явную проверку
        # на None и подмешиваем `# type: ignore` там, где требуется.
        location = TopLoc_Location()
        brep_tool = BRep_Tool()
        mesh = brep_tool.Triangulation(face, location)  # type: ignore
        if mesh is not None:
            # Проходим по треугольникам (индексация в OCC начинается с 1)
            num_tris = mesh.NbTriangles()  
            for i in range(1, num_tris + 1): 
                index1, index2, index3 = mesh.Triangle(i).Get()

                if face_orientation_wrt_surface_normal:
                    # Сохраняем оригинальную намотку треугольника
                    tris.append([index1 - 1, index2 - 1, index3 - 1])
                else:
                    # Разворачиваем порядок вершин для согласования ориентации
                    tris.append([index3 - 1, index2 - 1, index1 - 1])

            # Вершины (нумерация начинается с 1)
            num_vertices = mesh.NbNodes() 
            for i in range(1, num_vertices + 1): 
                vertex_stp = np.array(list(mesh.Node(i).Coord())) 
                # Масштабируем из единиц STEP (мм) в единицы OBJ (см)
                vertex_obj = 0.1 * vertex_stp
                verts.append(vertex_obj.tolist())

        return np.array(verts), np.array(tris)

    
    def get_box_from_tris(self, face: TopoDS_Shape) -> Bnd_Box:
        """
        Получает ограничивающий бокс для грани на основе треугольников.

        Args:
            face (Any): Грань модели (TopoDS_Face из OCC).

        Returns:
            Bnd_Box: Ограничивающий бокс для грани.
        """
        verts, tris = self.get_face_triangles(face)
        box = Bnd_Box()

        # Примечание: verts уже масштабированы в get_face_triangles (мм->см),
        # поэтому дополнительного преобразования здесь не требуется.
        for i in range(verts.shape[0]):
            vert = verts[i]
            pt = gp_Pnt(float(vert[0]), float(vert[1]), float(vert[2]))
            box.Add(pt)
        return box


    def check_box(self, fusion_face_box: Bnd_Box, step_face_box: Bnd_Box, tol: float, msg: str) -> bool:
        """
        Проверяет, находятся ли два ограничивающих бокса в пределах допуска.

        Args:
            fusion_face_box (Bnd_Box): Бокс из данных Fusion.
            step_face_box (Bnd_Box): Бокс из данных STEP.
            tol (float): Значение допуска для сравнения.
            msg (str): Сообщение для вывода при неудаче.

        Returns:
            bool: True, если боксы в пределах допуска, иначе False.
        """
        # Сравниваем расстояние между соответствующими углами боксов.
        # CornerMin/CornerMax возвращают gp_Pnt.
        min_in_tol = fusion_face_box.CornerMin().Distance(step_face_box.CornerMin()) < tol
        max_in_tol = fusion_face_box.CornerMax().Distance(step_face_box.CornerMax()) < tol
        if not (min_in_tol or max_in_tol):
            print(msg)
            return False
        return True


    def find_face_boxes(self, basename: str) -> Optional[List[Bnd_Box]]:
        """
        Находит ограничивающие боксы для граней из OBJ и FIDX файлов.

        Args:
            basename (str): Базовое имя файлов без расширения.

        Returns:
            Optional[List[Bnd_Box]]: Список боксов для каждой грани, или None если файлы отсутствуют или недействительны.
        """
        obj_pathname = self.get_obj_pathname(basename)
        fidx_pathname = self.get_fidx_pathname(basename)
        if not obj_pathname.exists():
            print(f"{obj_pathname} does not exist")
            return None

        if not fidx_pathname.exists():
            print(f"{fidx_pathname} does not exist")
            return None

        v, f = igl.read_triangle_mesh(str(obj_pathname))
        # Приводим к явным numpy-массивам для статической типизации
        v = np.asarray(v, dtype=float)
        f = np.asarray(f, dtype=int)
        tris_to_faces = np.loadtxt(fidx_pathname, dtype=np.uint64)

        boxes: Dict[int, Bnd_Box] = {}
        # Проходим по всем треугольникам в меше
        for tri_index in range(f.shape[0]):
            # Создаём ограничивающий бокс для текущего треугольника
            tri_box = Bnd_Box()
            for ptidx in f[tri_index]:
                # ptidx — индекс вершины в массиве v
                ptidx_i = int(ptidx)
                point = v[ptidx_i]
                pt = gp_Pnt(float(point[0]), float(point[1]), float(point[2]))
                tri_box.Add(pt)

            face_index = int(tris_to_faces[tri_index])
            # Накопление боксов: расширяем существующий бокс или добавляем новый
            if face_index not in boxes:
                boxes[face_index] = tri_box
            else:
                boxes[face_index].Add(boxes[face_index].CornerMin())  # noop-safe
                boxes[face_index].Add(tri_box.CornerMin())
                boxes[face_index].Add(tri_box.CornerMax())

        # Преобразуем в последовательный список: ожидаем, что ключи от 0..N-1
        box_arr: List[Bnd_Box] = []
        for i in range(len(boxes)):
            if i not in boxes:
                return None
            box_arr.append(boxes[i])

        return box_arr

    def load_parts_and_fusion_indices_step_file(self, pathname: Path) -> Tuple[List[TopoDS_Shape], Dict[Any, int]]:
        """
        Загружает список частей из STEP-файла и возвращает карту от хэша формы к индексу грани Fusion.

        Args:
            pathname (Path): Путь к STEP-файлу.

        Returns:
            Tuple[List[Any], Dict[Any, int]]: Кортеж из списка частей и словаря граней с индексами.
        """
        # Код основан на 
        # #https://github.com/tpaviot/pythonocc-core/blob/master/src/Extend/DataExchange.py
        assert pathname.exists()

        # Создаем дескриптор документа
        # Создание документа вызывает конфликт типизации в анализаторах — подавляем
        doc = TDocStd_Document(TCollection_ExtendedString("FaceIndexValidator"))  # type: ignore

        # Получаем корневую сборку и инструменты для форм и цвета
        shape_tool = XCAFDoc_DocumentTool.ShapeTool(doc.Main())
        color_tool = XCAFDoc_DocumentTool.ColorTool(doc.Main())
        step_reader = STEPCAFControl_Reader()

        # Включаем чтение цветов (мы используем цвет для кодирования индекса)
        step_reader.SetColorMode(True)

        status = step_reader.ReadFile(str(pathname))
        shapes: List[TopoDS_Shape] = []
        face_map: Dict[Any, int] = {}

        if status == IFSelect_RetDone:
            try:
                # Передаём данные STEP в документ
                step_reader.Transfer(doc) 

                # Получаем метки верхнего уровня (free shapes)
                labels = TDF_LabelSequence()
                shape_tool.GetFreeShapes(labels)  

                # Проходим по меткам верхнего уровня: ожидаем твердое тело
                for i in range(labels.Length()):  
                    label = labels.Value(i + 1)

                    # Получаем форму (обычно SOLID)
                    shape = shape_tool.GetShape(label)
                    if shape.ShapeType() == TopAbs_SOLID:
                        shapes.append(shape)
                    else:
                        print("Root shape is not a solid")

                    # Обрабатываем подписи уровня ниже — ожидаем грани
                    sub_shapes_labels = TDF_LabelSequence()
                    shape_tool.GetSubShapes(label, sub_shapes_labels)  
                    for j in range(sub_shapes_labels.Length()): 
                        sub_label = sub_shapes_labels.Value(j + 1)

                        sub_shape = shape_tool.GetShape(sub_label)
                        if sub_shape.ShapeType() != TopAbs_FACE:
                            print("Sub shape is not a face")
                            continue

                        # Получаем цвет (r,g,b) — он кодирует индекс грани из Fusion
                        # Создание Quantity_Color типизировано неточно в stubs — игнорируем
                        c = Quantity_Color(0.5, 0.5, 0.5, Quantity_TOC_RGB)  # type: ignore
                        # Получаем компоненты цвета; API возвращает значения в диапазоне [0,1]
                        color_tool.GetColor(sub_label, 0, c)  # type: ignore
                        color_tool.GetColor(sub_label, 1, c)  # type: ignore
                        color_tool.GetColor(sub_label, 2, c)  # type: ignore

                        # Декодируем индекс: порядок байтов совпадает с тем, как кодировалось
                        # в Fusion (r + g<<8 + b<<16)
                        r = int(c.Red() * 256)
                        g = int(c.Green() * 256 * 256)
                        b = int(c.Blue() * 256 * 256 * 256)
                        recovered_index = r + g + b

                        # Сохраняем соответствие от объекта грани к индексу.
                        # Используем Any, т.к. точные типы OCC трудно выразить для хеширования
                        face_map[sub_shape] = int(recovered_index)
            except Exception:
                print("Step transfer problem")
        else:
            print("Step reading problem.")
            raise AssertionError("Error: can't read file.")

        return shapes, face_map