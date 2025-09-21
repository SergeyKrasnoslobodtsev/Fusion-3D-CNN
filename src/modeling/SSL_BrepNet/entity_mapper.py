from typing import List, Dict, Tuple
from OCC.Extend.TopologyUtils import TopologyExplorer, WireExplorer
from OCC.Core.TopoDS import TopoDS_Shape
from OCC.Core.TopAbs import (
    TopAbs_FORWARD,
    TopAbs_REVERSED,
    TopAbs_Orientation,
)
from OCC.Core.TopoDS import (
    TopoDS_Solid, TopoDS_Shell, 
    TopoDS_Face, TopoDS_Wire, TopoDS_Edge, TopoDS_Vertex, TopoDS_Shape
)

def orientation_to_sense(orientation: TopAbs_Orientation) -> bool:
    """Преобразует ориентацию из формата OpenCascade в логическое значение.
    
    OpenCascade: 
        TopAbs_FORWARD = 0; 
        TopAbs_REVERSED = 1.
    Возвращает True, если ориентация FORWARD, иначе False.
    Не поддерживаются INTERNAL/EXTERNAL (значения 2 и 3).
    """
    assert orientation in (TopAbs_FORWARD, TopAbs_REVERSED), \
        "Orientation must be either FORWARD or REVERSED."
    return orientation == TopAbs_FORWARD


class EntityMapper:
    """Служит для взаимного сопоставления топологических сущностей OpenCascade (тела, 
    поверхности, ребра, вершины и др.) и их уникальных индексов для записи в файл топологии.
    Позволяет работать с моделями любой размерности, включая неориентируемые поверхности.
    
    Attributes:
        body_map, solid_map, shell_map, face_map, loop_map, edge_map, 
        vertex_map, halfedge_map — словари для хранения соответствий между 
        хэшами сущностей (body, solid, shell, face, loop, edge, vertex, halfedge) 
        и их индексом в общем списке.
        primary_face_orientations_map — хранит ориентацию "главных" поверхностей,
        для случаев, когда поверхность встречается в нескольких оболочках.
    
    """

    def __init__(self, bodies: List[TopoDS_Shape] | TopoDS_Shape) -> None:
        """Инициализирует mapper для списка тел или одного тела.
        
        Args:
            bodies: Одно тело или список тел, топология которых будет разобрана.
        Example:
                from OCC.Core.STEPControl import STEPControl_Reader\n
                from OCC.Core.IFSelect import IFSelect_RetDone\n
                reader = STEPControl_Reader()\n
                status = reader.ReadFile("model.stp")\n
                if status == IFSelect_RetDone:\n
                    reader.TransferRoots()
                    shape = reader.Shape()
                    mapper = EntityMapper(shape)
                    # Получение индексов для работы с графом
                    # (например, для визуализации, экспорта или машинного обучения)
                    edge_idx = mapper.edge_index(edge_obj)
                    face_idx = mapper.face_index(face_obj)
                    # и т.д.
        """
        self.body_map: Dict[int, int] = dict()
        self.solid_map: Dict[int, int] = dict()
        self.shell_map: Dict[int, int] = dict()
        self.face_map: Dict[int, int] = dict()
        self.loop_map: Dict[int, int] = dict()
        self.edge_map: Dict[int, int] = dict()
        self.halfedge_map: Dict[Tuple[int, TopAbs_Orientation], int] = dict()
        self.vertex_map: Dict[int, int] = dict()
        self.primary_face_orientations_map: Dict[int, bool] = dict()

        # Если передано одно тело, упаковываем его в список
        bodies_list: List[TopoDS_Shape]
        if isinstance(bodies, TopoDS_Shape):
            bodies_list = [bodies]
        else:
            bodies_list = list(bodies)  # Поддержка любого итерируемого

        for body in bodies_list:
            top_exp = TopologyExplorer(body)
            self.append_body(body)
            self.append_solids(top_exp)
            self.append_shells(top_exp)
            self.append_faces(top_exp)
            self.append_loops(top_exp)
            self.append_edges(top_exp)
            self.append_halfedges(body)
            self.append_vertices(top_exp)
            self.build_primary_face_orientations_map(top_exp)

    # === Публичный интерфейс ===

    def get_nr_of_edges(self) -> int:
        """Возвращает количество уникальных ребер в модели."""
        return len(self.edge_map)

    def get_nr_of_surfaces(self) -> int:
        """Возвращает количество уникальных поверхностей в модели."""
        return len(self.face_map)

    def body_index(self, body: TopoDS_Shape) -> int:
        """Возвращает индекс тела в общей нумерации."""
        h = self.get_hash(body)
        return self.body_map[h]

    def solid_index(self, solid: TopoDS_Solid) -> int:
        """Возвращает индекс твердого тела (solid)."""
        h = self.get_hash(solid)
        return self.solid_map[h]

    def shell_index(self, shell: TopoDS_Shell) -> int:
        """Возвращает индекс оболочки (shell)."""
        h = self.get_hash(shell)
        return self.shell_map[h]

    def face_index(self, face: TopoDS_Face) -> int:
        """Возвращает индекс поверхности (face)."""
        h = self.get_hash(face)
        return self.face_map[h]

    def loop_index(self, loop: TopoDS_Wire) -> int:
        """Возвращает индекс петли (wire, loop)."""
        h = self.get_hash(loop)
        return self.loop_map[h]

    def edge_index(self, edge: TopoDS_Edge) -> int:
        """Возвращает индекс ребра (edge)."""
        h = self.get_hash(edge)
        return self.edge_map[h]

    def halfedge_index(self, halfedge: TopoDS_Edge) -> int:
        """Возвращает индекс полуребра (half-edge), с учетом ориентации.
        Полуребро — это ребро в контексте одной из петель."""
        h = self.get_hash(halfedge)
        orientation = halfedge.Orientation()
        tup = (h, orientation)
        return self.halfedge_map[tup]

    def halfedge_exists(self, halfedge: TopoDS_Edge) -> bool:
        """Проверяет, существует ли полуребро в карте."""
        h = self.get_hash(halfedge)
        orientation = halfedge.Orientation()
        tup = (h, orientation)
        return tup in self.halfedge_map

    def vertex_index(self, vertex: TopoDS_Vertex) -> int:
        """Возвращает индекс вершины (vertex)."""
        h = self.get_hash(vertex)
        return self.vertex_map[h]

    def primary_face_orientation(self, face: TopoDS_Face) -> bool:
        """Возвращает ориентацию "главной" поверхности (True — FORWARD, False — REVERSED)."""
        h = self.get_hash(face)
        return self.primary_face_orientations_map[h]

    # методы построения карты

    def get_hash(self, ent: TopoDS_Shape) -> int:
        """Вычисляет хэш через TShape и Location."""
        intmax = 2147483647
        tshape_hash = ent.TShape().__hash__()
        location_hash = ent.Location().HashCode() 
        combined = (tshape_hash, location_hash)
        return abs(hash(combined)) % intmax

    def append_body(self, body: TopoDS_Shape) -> None:
        """Добавляет тело в карту и присваивает ему уникальный индекс."""
        h = self.get_hash(body)
        assert h not in self.body_map, "Duplicate body hash detected."
        self.body_map[h] = len(self.body_map)

    def append_solids(self, top_exp: TopologyExplorer) -> None:
        """Добавляет все твердые тела (solids) обходчика в карту."""
        for solid in top_exp.solids():
            self.append_solid(solid)

    def append_solid(self, solid: TopoDS_Solid) -> None:
        h = self.get_hash(solid)
        assert h not in self.solid_map, "Duplicate solid hash detected."
        self.solid_map[h] = len(self.solid_map)

    def append_shells(self, top_exp: TopologyExplorer) -> None:
        for shell in top_exp.shells():
            self.append_shell(shell)

    def append_shell(self, shell: TopoDS_Shell) -> None:
        h = self.get_hash(shell)
        assert h not in self.shell_map, "Duplicate shell hash detected."
        self.shell_map[h] = len(self.shell_map)

    def append_faces(self, top_exp: TopologyExplorer) -> None:
        for face in top_exp.faces():
            self.append_face(face)

    def append_face(self, face: TopoDS_Face) -> None:
        h = self.get_hash(face)
        assert h not in self.face_map, "Duplicate face hash detected."
        self.face_map[h] = len(self.face_map)

    def append_loops(self, top_exp: TopologyExplorer) -> None:
        for loop in top_exp.wires():
            self.append_loop(loop)

    def append_loop(self, loop: TopoDS_Wire) -> None:
        h = self.get_hash(loop)
        assert h not in self.loop_map, "Duplicate loop hash detected."
        self.loop_map[h] = len(self.loop_map)

    def append_edges(self, top_exp: TopologyExplorer) -> None:
        for edge in top_exp.edges():
            self.append_edge(edge)

    def append_edge(self, edge: TopoDS_Edge) -> None:
        h = self.get_hash(edge)
        assert h not in self.edge_map, "Duplicate edge hash detected."
        self.edge_map[h] = len(self.edge_map)

    def append_halfedges(self, body: TopoDS_Shape) -> None:
        """Добавляет в карту все полуребра (HalfEdge), с учетом ориентации."""
        oriented_top_exp = TopologyExplorer(body, ignore_orientation=False)
        for wire in oriented_top_exp.wires():
            wire_exp = WireExplorer(wire)
            for halfedge in wire_exp.ordered_edges():
                self.append_halfedge(halfedge)

    def append_halfedge(self, halfedge: TopoDS_Edge) -> None:
        h = self.get_hash(halfedge)
        orientation = halfedge.Orientation()
        tup = (h, orientation)
        if tup not in self.halfedge_map:
            self.halfedge_map[tup] = len(self.halfedge_map)

    def append_vertices(self, top_exp: TopologyExplorer) -> None:
        for vertex in top_exp.vertices():
            self.append_vertex(vertex)

    def append_vertex(self, vertex: TopoDS_Vertex) -> None:
        h = self.get_hash(vertex)
        assert h not in self.vertex_map, "Duplicate vertex hash detected."
        self.vertex_map[h] = len(self.vertex_map)

    def build_primary_face_orientations_map(self, top_exp: TopologyExplorer) -> None:
        """Строит карту ориентаций "главных" поверхностей (primary faces)."""
        for face in top_exp.faces():
            h = self.get_hash(face)
            orientation = orientation_to_sense(face.Orientation())
            assert h not in self.primary_face_orientations_map, \
                "Duplicate primary face hash detected."
            self.primary_face_orientations_map[h] = orientation


