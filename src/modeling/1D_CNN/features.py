import numpy as np
from loguru import logger
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.sparse.csgraph as csgraph
import warnings

from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.BRep import BRep_Tool
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.TopoDS import TopoDS_Shape, topods
from OCC.Extend.TopologyUtils import TopologyExplorer
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.GeomAbs import (
    GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone, GeomAbs_Sphere, GeomAbs_Torus,
    GeomAbs_BezierSurface, GeomAbs_BSplineSurface, GeomAbs_SurfaceOfRevolution,
    GeomAbs_SurfaceOfExtrusion, GeomAbs_OtherSurface
)
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop

class BrepExtractor():
    """
    Экстрактор 1D-последовательностей из CAD (STEP/B-Rep + треугольная сетка).
    Формирует 7 каналов длины feature_dim (по умолчанию 256):
      1) RDF(K)                   -> resample(feature_dim)
      2) hist(edge_lengths, bins) -> resample
      3) hist(face_areas, bins)   -> resample
      4) hist(dihedral_angles, bins, [0, pi]) -> resample
      5) hist(vertex_degrees, bins//2) -> resample
      6) B-Rep surface-type area ratios (10) -> resample
      7) LBO spectrum (k)         -> resample
    Затем либо возвращает матрицу (7, feature_dim), либо агрегирует в 1D-вектор.
    """

    def __init__(
        self,
        rdf_k: int = 256,
        lbo_k: int = 16,
        bins: int = 64,
        feature_dim: int = 256,
        output_mode: str = "matrix",        # 'matrix' | 'pooled'     # для 'pooled': mean|max|
        weld_tol: float = 1e-6,
        aggregate_method: str = "mean"
    ):
        self.rdf_k = int(rdf_k)
        self.lbo_k = int(lbo_k)
        self.bins = int(bins)
        self.feature_dim = int(feature_dim)
        self.output_mode = output_mode
        self.weld_tol = float(weld_tol)
        self.aggregate_method = aggregate_method
        logger.info(
            f"Инициализация (K={self.rdf_k}, LBO={self.lbo_k}, bins={self.bins}, "
            f"D={self.feature_dim}, mode={self.output_mode})"
        )

    def extract(self, stp_file: str) -> np.ndarray | None:
        shape = self._load_step_shape(stp_file)
        if shape is None or shape.IsNull():
            logger.warning(f"Не удалось загрузить/пустая геометрия: {stp_file}")
            return None

        try:
            self._mesh_shape(shape, lin_deflection=0.02, ang_deflection=0.785)
            V, F = self._get_vertices_and_faces(shape)
            if V.shape[0] < 4 or F.shape[0] < 1:
                logger.warning(f"Недостаточно геометрии после триангуляции: {stp_file}")
                return None

            V, F = self._weld_vertices(V, F, tol=self.weld_tol)
            V = self._normalize_vertices(V)

            h_edge, h_area, h_dih, h_deg = self._compute_mesh_histograms(V, F)
            rdf = self._compute_rdf_robust(V, F, K=self.rdf_k)
            if rdf.size == 0:
                rdf = np.zeros(self.rdf_k, dtype=np.float32)
            brep_vec = self._brep_surface_type_hist(shape)
            lbo = self._compute_lbo_spectrum_features(V, F, k=self.lbo_k)

            blocks = [
                self._resample_to_dim(rdf, self.feature_dim),
                self._resample_to_dim(h_edge, self.feature_dim),
                self._resample_to_dim(h_area, self.feature_dim),
                self._resample_to_dim(h_dih, self.feature_dim),
                self._resample_to_dim(h_deg, self.feature_dim),
                self._resample_to_dim(brep_vec, self.feature_dim),
                self._resample_to_dim(lbo, self.feature_dim),
            ]
            feature_matrix = np.stack(blocks, axis=0).astype(np.float32)  # (7, D)

            if self.output_mode == "pooled":
                pooled = self._aggregate_features(feature_matrix)  # (D,)
                return pooled.astype(np.float32)
            else:
                return feature_matrix

        except Exception as e:
            logger.error(f"Ошибка извлечения признаков для {stp_file}: {e}")
            return None

    # ---------------------------- Вспомогательные блоки ---------------------------- #
    def _aggregate_features(self, feature_vectors: np.ndarray) -> np.ndarray:
        """Агрегирует массив векторов признаков (для одной модели) в один вектор"""
        if feature_vectors.ndim != 2:
            feature_vectors = feature_vectors.reshape(-1, feature_vectors.shape[-1])
        if self.aggregate_method == 'mean':
            return np.mean(feature_vectors, axis=0)
        elif self.aggregate_method == 'max':
            return np.amax(feature_vectors, axis=0)
        else:
            raise ValueError(f"Неизвестный метод агрегации: {self.aggregate_method}")


    # --- нормализация и сварка --- #
    @staticmethod
    def _normalize_vertices(V: np.ndarray) -> np.ndarray:
        c = V.mean(axis=0, keepdims=True)
        Vc = V - c
        scale = np.linalg.norm(Vc, axis=1).max() + 1e-12
        return (Vc / scale).astype(np.float64)

    @staticmethod
    def _weld_vertices(V: np.ndarray, F: np.ndarray, tol: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
        """Склеить близкие вершины (устранить дубли)."""
        if V.size == 0 or F.size == 0:
            return V, F
        key = np.round(V / tol).astype(np.int64)
        _, inv, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
        Vw = np.zeros((counts.size, 3), dtype=np.float64)
        np.add.at(Vw, inv, V)
        Vw /= counts[:, None]
        Fw = inv[F]
        return Vw, Fw

    # --- гистограммы по сетке --- #
    def _compute_mesh_histograms(self, V: np.ndarray, F: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        E, L = self.mesh_edge_data(V, F)
        A = self.mesh_face_areas(V, F)
        D = self._mesh_dihedral_angles(V, F)
        deg = self.mesh_vertex_degrees(F, V.shape[0])

        h_edge = self._hist_l1(L, bins=self.bins, log=True)
        h_area = self._hist_l1(A, bins=self.bins, log=True)
        h_dih  = self._hist_l1(D, bins=self.bins, rng=(0.0, np.pi))
        vmax = deg.max() if deg.size else 1
        h_deg  = self._hist_l1(deg.astype(np.float64), bins=max(2, self.bins // 2), rng=(0.0, float(vmax)))
        return h_edge, h_area, h_dih, h_deg

    @staticmethod
    def _hist_l1(x: np.ndarray, bins: int, rng: tuple | None = None, log: bool = False) -> np.ndarray:
        """L1-нормированная гистограмма (сумма=1), опция лог-преобразования."""
        if x.size == 0:
            return np.zeros(bins, dtype=np.float32)
        vals = np.log10(np.clip(x, 1e-12, None)) if log else x
        H, _ = np.histogram(vals, bins=bins, range=rng, density=False)
        s = H.sum()
        return (H / s if s > 0 else np.zeros_like(H)).astype(np.float32)

    # --- RDF --- #
    def _compute_rdf(self, V: np.ndarray, F: np.ndarray, K: int) -> np.ndarray:
        c = V.mean(axis=0)
        rmax = np.linalg.norm(V - c, axis=1).max() + 1e-12
        dirs = self.fibonacci_sphere(K)
        dists = np.array([self.ray_triangle_intersections(c, d, V, F) for d in dirs], dtype=np.float64)
        dists[~np.isfinite(dists)] = rmax
        return (dists / rmax).astype(np.float32)

    def _compute_rdf_robust(self, V: np.ndarray, F: np.ndarray, K: int) -> np.ndarray:
        rdf_raw = self._compute_rdf(V, F, K)
        # лог-сжатие + сглаживание (устойчивость к выбросам/«зазубринам»)
        rdf_log = np.log1p(np.maximum(rdf_raw, 0.0).astype(np.float64))
        from scipy.ndimage import gaussian_filter1d
        rdf_smooth = gaussian_filter1d(rdf_log, sigma=1.0)
        return rdf_smooth.astype(np.float32)

    # --- B-Rep поверхности --- #
    @staticmethod
    def face_area(face):
        props = GProp_GProps()
        brepgprop.SurfaceProperties(face, props)
        return props.Mass()

    def _brep_surface_type_hist(self, shape: TopoDS_Shape) -> np.ndarray:
        keys = [
            GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone, GeomAbs_Sphere, GeomAbs_Torus,
            GeomAbs_BezierSurface, GeomAbs_BSplineSurface, GeomAbs_SurfaceOfRevolution,
            GeomAbs_SurfaceOfExtrusion, GeomAbs_OtherSurface
        ]
        type2area: dict[int, float] = {k: 0.0 for k in keys}
        total_area = 0.0
        topo = TopologyExplorer(shape)
        for face in topo.faces():
            A = self.face_area(face)
            if A <= 0:
                continue
            surf = BRepAdaptor_Surface(face, True)
            st = surf.GetType()
            total_area += A
            type2area[st] = type2area.get(st, 0.0) + A

        if total_area <= 1e-12:
            return np.zeros(len(keys), dtype=np.float32)

        return np.array([type2area[k] / total_area for k in keys], dtype=np.float32)

    # --- ЛБО спектр --- #
    def _compute_lbo_spectrum_features(self, V: np.ndarray, F: np.ndarray, k: int) -> np.ndarray:
        padded = np.zeros(k, dtype=np.float32)
        try:
            evals, _ = self._compute_lbo_spectrum(V, F, k=k)
            n = min(len(evals), k)
            if n > 0:
                padded[:n] = evals[:n].astype(np.float32)
        except (RuntimeError, np.linalg.LinAlgError, ValueError) as e:
            logger.warning(f"Спектр не вычислен: {e}. Заполняю нулями.")
        return padded

    @staticmethod
    def _compute_lbo_spectrum(V, F, k=32, scale_invariant=True):
        # площадь всей сетки
        def total_area(V, F):
            v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
            return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1).sum()

        # удалить вырожденные триугольники и пересэмплировать индексы
        def clean_mesh(V, F, area_eps=1e-14):
            if F.size == 0:
                return V, F
            area2 = np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)
            F = F[area2 > area_eps]
            if F.size == 0:
                return V[:0], F
            used = np.unique(F.ravel())
            remap = -np.ones(V.shape[0], dtype=np.int64)
            remap[used] = np.arange(used.size)
            return V[used], remap[F]

        # оставить крупнейшую связную компоненту (исправлено: все 3 ребра, двунаправленно)
        def keep_largest_component(V, F):
            if F.size == 0:
                return V, F
            n = V.shape[0]
            r = np.arange(F.shape[0])
            pairs = np.stack(
                [F[:, [0, 1]], F[:, [1, 0]], F[:, [1, 2]], F[:, [2, 1]], F[:, [2, 0]], F[:, [0, 2]]],
                axis=0
            ).reshape(-1, 2)
            adj = sp.csr_matrix((np.ones(pairs.shape[0]), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
            ncomp, labels = csgraph.connected_components(adj, directed=False)
            if ncomp <= 1:
                return V, F
            largest = np.argmax(np.bincount(labels))
            mask_v = labels == largest
            idx_old = np.where(mask_v)[0]
            remap = -np.ones(n, dtype=np.int64)
            remap[idx_old] = np.arange(idx_old.size)
            Fm = remap[F]
            Fm = Fm[(Fm >= 0).all(axis=1)]
            return V[idx_old], Fm

        # котангенс-лапласиан + барицентрическая масса
        def build_laplacian_cotan(V, F):
            n = V.shape[0]
            i, j, k = F[:, 0], F[:, 1], F[:, 2]
            vi, vj, vk = V[i], V[j], V[k]
            area2 = np.linalg.norm(np.cross(vj - vi, vk - vi), axis=1)
            area2_safe = np.maximum(area2, 1e-15)
            cot_i = ((vj - vi) * (vk - vi)).sum(axis=1) / area2_safe
            cot_j = ((vi - vj) * (vk - vj)).sum(axis=1) / area2_safe
            cot_k = ((vi - vk) * (vj - vk)).sum(axis=1) / area2_safe
            w_ij, w_jk, w_ki = 0.5 * cot_k, 0.5 * cot_i, 0.5 * cot_j
            rows = np.concatenate([i, j, j, k, k, i])
            cols = np.concatenate([j, i, k, j, i, k])
            data = np.concatenate([-w_ij, -w_ij, -w_jk, -w_jk, -w_ki, -w_ki])
            L = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
            L = L - sp.diags(L.sum(axis=1).A1) # type: ignore
            M_diag = np.zeros(n)
            tri_area = 0.5 * area2
            np.add.at(M_diag, i, tri_area / 3.0)
            np.add.at(M_diag, j, tri_area / 3.0)
            np.add.at(M_diag, k, tri_area / 3.0)
            return L, np.maximum(M_diag, 1e-15)

        # pipeline
        Vc, Fc = clean_mesh(*keep_largest_component(*clean_mesh(V, F)))
        if Vc.shape[0] < 3 or Fc.shape[0] < 1:
            raise RuntimeError("Недостаточно данных для спектра после чистки.")

        Vn = Vc
        if scale_invariant:
            A = total_area(Vc, Fc)
            if A > 0:
                Vn = Vc / np.sqrt(A)

        L, M_diag = build_laplacian_cotan(Vn, Fc)
        n = Vn.shape[0]
        k_solve = min(k, max(1, n - 2))
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='The problem size')
            evals, evecs = spla.eigsh(L, k=k_solve, M=sp.diags(M_diag), sigma=1e-8, which='LM', tol=0)
        order = np.argsort(evals)
        evals = evals[order]
        pos = evals > 1e-10
        return evals[pos], evecs[:, order][:, pos].astype(np.float64)

    # утилиты
    @staticmethod
    def _load_step_shape(step_path: str) -> TopoDS_Shape | None:
        reader = STEPControl_Reader()
        if reader.ReadFile(step_path) != IFSelect_RetDone:
            logger.error(f"Не удалось прочитать STEP: {step_path}")
            return None
        reader.TransferRoots()
        return reader.OneShape()

    @staticmethod
    def _mesh_shape(shape, lin_deflection=0.05, ang_deflection=0.5, is_relative=True, parallel=True):
        BRepMesh_IncrementalMesh(shape, lin_deflection, is_relative, ang_deflection, parallel)

    @staticmethod
    def _get_vertices_and_faces(shape: TopoDS_Shape) -> tuple[np.ndarray, np.ndarray]:
        verts_chunks, faces_chunks = [], []
        v_off = 0
        topo = TopologyExplorer(shape)
        for face in topo.faces():
            loc = TopLoc_Location()
            triangulation = BRep_Tool.Triangulation(face, loc)
            if triangulation is None:
                continue
            trsf = loc.Transformation()
            nb_nodes = triangulation.NbNodes()
            cur_verts = np.empty((nb_nodes, 3), dtype=np.float64)
            for i in range(1, nb_nodes + 1):
                p = triangulation.Node(i).Transformed(trsf)
                cur_verts[i - 1] = [p.X(), p.Y(), p.Z()]
            verts_chunks.append(cur_verts)
            nb_tris = triangulation.NbTriangles()
            cur_faces = np.empty((nb_tris, 3), dtype=np.int64)
            for i in range(1, nb_tris + 1):
                t = triangulation.Triangle(i)
                i1, i2, i3 = t.Get()
                cur_faces[i - 1] = [v_off + i1 - 1, v_off + i2 - 1, v_off + i3 - 1]
            faces_chunks.append(cur_faces)
            v_off += nb_nodes
        if not verts_chunks:
            return np.empty((0, 3)), np.empty((0, 3))
        return np.vstack(verts_chunks), np.vstack(faces_chunks)

    @staticmethod
    def mesh_edge_data(V: np.ndarray, F: np.ndarray):
        E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], axis=0)
        E.sort(axis=1)
        E = np.unique(E, axis=0)
        L = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1)
        return E, L

    @staticmethod
    def mesh_face_areas(V: np.ndarray, F: np.ndarray):
        v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
        return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)

    @staticmethod
    def mesh_face_normals(V: np.ndarray, F: np.ndarray):
        v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
        n = np.cross(v1 - v0, v2 - v0)
        n_norm = np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
        return n / n_norm

    def _mesh_dihedral_angles(self, V: np.ndarray, F: np.ndarray):
        from collections import defaultdict
        E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], axis=0)
        E_sorted = np.sort(E, axis=1)
        tri_idx = np.repeat(np.arange(F.shape[0]), 3)
        edge2tris = defaultdict(list)
        for e, t in zip(map(tuple, E_sorted), tri_idx):
            edge2tris[e].append(t)
        N = self.mesh_face_normals(V, F)
        ang = []
        for tris in edge2tris.values():
            if len(tris) == 2:
                i, j = tris
                c = np.clip(np.dot(N[i], N[j]), -1.0, 1.0)
                ang.append(np.arccos(c))
        return np.array(ang, dtype=np.float64)

    @staticmethod
    def mesh_vertex_degrees(F: np.ndarray, Vn: int):
        deg = np.zeros(Vn, dtype=np.int32)
        np.add.at(deg, F.ravel(), 1)
        return deg

    @staticmethod
    def fibonacci_sphere(n: int):
        i = np.arange(n, dtype=np.float64)
        phi = (1 + 5 ** 0.5) / 2
        theta = 2 * np.pi * i / phi
        z = 1 - (2 * i + 1) / n
        r = np.sqrt(np.maximum(0.0, 1 - z * z))
        x, y = r * np.cos(theta), r * np.sin(theta)
        return np.stack([x, y, z], axis=1)

    @staticmethod
    def ray_triangle_intersections(orig: np.ndarray, dirv: np.ndarray, V: np.ndarray, F: np.ndarray):
        v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
        eps = 1e-9
        e1, e2 = v1 - v0, v2 - v0
        pvec = np.cross(dirv, e2)
        det = (e1 * pvec).sum(axis=1)
        mask = np.abs(det) > eps
        inv_det = np.zeros_like(det)
        inv_det[mask] = 1.0 / det[mask]
        tvec = orig - v0
        u = (tvec * pvec).sum(axis=1) * inv_det
        qvec = np.cross(tvec, e1)
        v = (dirv * qvec).sum(axis=1) * inv_det
        t = (e2 * qvec).sum(axis=1) * inv_det
        cond = (mask) & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > eps)
        t_valid = np.where(cond, t, np.inf)
        return t_valid.min()

    # приведение длины
    @staticmethod
    def _resample_to_dim(x: np.ndarray, L: int) -> np.ndarray:
        """Линейная интерполяция/обрезка/паддинг до длины L с float32 возвратом."""
        if x.size == 0:
            return np.zeros(L, dtype=np.float32)
        if x.size == L:
            return x.astype(np.float32, copy=False)
        # интерполяция по равномерной сетке
        xp = np.linspace(0.0, 1.0, num=x.size, endpoint=True)
        fp = x.astype(np.float64)
        xq = np.linspace(0.0, 1.0, num=L, endpoint=True)
        yq = np.interp(xq, xp, fp)
        return yq.astype(np.float32)