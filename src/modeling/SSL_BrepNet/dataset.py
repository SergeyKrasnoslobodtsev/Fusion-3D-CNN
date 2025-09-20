import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

from .normlization import BrepNetStandardizer
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .model.encoder import CustomBRepEncoder

@dataclass
class BRepData:
    """
    Структура данных для B-Rep представления CAD модели.
    """
    vertices: torch.Tensor      # [num_vertices, v_in_width]
    edges: torch.Tensor         # [num_edges, e_in_width]
    faces: torch.Tensor         # [num_faces, f_in_width]
    edge_to_vertex: torch.Tensor # [2, num_edges]
    face_to_edge: torch.Tensor   # [2, num_face_edges]
    face_to_face: torch.Tensor   # [2, num_face_connections]
    # Поле для батчинга
    face_batch_idx: Optional[torch.Tensor] = None

def batch_brep_data(brep_data_list: List[BRepData]) -> BRepData:
    """
    Функция для объединения нескольких BRepData в один батч.
    """
    vertices, edges, faces = [], [], []
    edge_to_vertex, face_to_edge, face_to_face = [], [], []
    face_batch_idx = []

    v_offset, e_offset, f_offset = 0, 0, 0

    for i, data in enumerate(brep_data_list):
        vertices.append(data.vertices)
        edges.append(data.edges)
        faces.append(data.faces)

        edge_to_vertex.append(data.edge_to_vertex + v_offset)
        face_to_edge.append(data.face_to_edge + e_offset)
        face_to_face.append(data.face_to_face + f_offset)
        
        face_batch_idx.append(torch.full((data.faces.shape[0],), i, dtype=torch.long))

        v_offset += data.vertices.shape[0]
        e_offset += data.edges.shape[0]
        f_offset += data.faces.shape[0]

    return BRepData(
        vertices=torch.cat(vertices, dim=0),
        edges=torch.cat(edges, dim=0),
        faces=torch.cat(faces, dim=0),
        edge_to_vertex=torch.cat(edge_to_vertex, dim=1),
        face_to_edge=torch.cat(face_to_edge, dim=1),
        face_to_face=torch.cat(face_to_face, dim=1),
        face_batch_idx=torch.cat(face_batch_idx, dim=0)
    )


class ReconstructionDataset(Dataset):
    """
    Загружает данные для одной CAD-модели и выбирает одну случайную грань с точками.
    Применяет стандартизацию только к признакам граней и ребер.
    """
    def __init__(self, data_dir: Path, stats_path: Path, points_per_sample: int = 1024):
        # Collect .npz files that include 'face_samples' or 'face_point_grids'
        self.data_files = []
        for p in sorted(data_dir.glob("*.npz")):
            try:
                with np.load(p, allow_pickle=True) as tmp:
                    if 'face_samples' in tmp.files or 'face_point_grids' in tmp.files:
                        self.data_files.append(p)
            except Exception:
                continue
        self.standardizer = BrepNetStandardizer(str(stats_path))
        self.points_per_sample = points_per_sample

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx) -> Dict[str, Any]:
        file_path = self.data_files[idx]
        with np.load(file_path, allow_pickle=True) as data:
            # Стандартизируем только признаки граней и ребер
            # Поддержка разных ключей: 'edges'/'faces' либо 'edge_features'/'face_features'
            # Предпочитаем BrepNet-ключи '*_features', затем fallback на 'edges'/'faces'
            if 'edge_features' in data.files:
                edges_arr = data['edge_features']
            elif 'edges' in data.files:
                edges_arr = data['edges']
            else:
                raise AssertionError(f"No edge features in {file_path}")

            if 'face_features' in data.files:
                faces_arr = data['face_features']
            elif 'faces' in data.files:
                faces_arr = data['faces']
            else:
                raise AssertionError(f"No face features in {file_path}")

            assert edges_arr.shape[0] > 0, f"Edge features are empty in {file_path}"
            assert faces_arr.shape[0] > 0, f"Face features are empty in {file_path}"
            features_to_standardize = {
                'edges': edges_arr,
                'faces': faces_arr
            }
            std_features = self.standardizer.transform(features_to_standardize)

            # Подготовка графовых связей
            # vertices/edge_to_vertex могут отсутствовать в BrepNet npz — создаём заглушки
            if 'vertices' in data.files:
                vertices = torch.from_numpy(data['vertices']).float()
            else:
                num_edges = std_features['edges'].shape[0]
                vertices = torch.zeros((num_edges, 3), dtype=torch.float32)

            if 'edge_to_vertex' in data.files:
                edge_to_vertex = torch.from_numpy(data['edge_to_vertex']).long()
            else:
                num_edges = std_features['edges'].shape[0]
                _e_idx = torch.arange(num_edges, dtype=torch.long)
                edge_to_vertex = torch.stack([_e_idx, _e_idx], dim=0)  # self-map заглушка

            if 'face_to_edge' in data.files:
                face_to_edge = torch.from_numpy(data['face_to_edge']).long()
            else:
                # Строим face->edge из coedge связей
                if 'face' in data.files and 'edge' in data.files:
                    coface = data['face']
                    coedge = data['edge']
                    faces_to_edges = self.standardizer.find_faces_to_edges(coface, coedge)
                    # Преобразуем в [2, num_face_edges]
                    f_idx = []
                    e_idx = []
                    for f, e_set in enumerate(faces_to_edges):
                        for e in e_set:
                            f_idx.append(f)
                            e_idx.append(int(e))
                    if len(f_idx) == 0:
                        face_to_edge = torch.zeros((2, 0), dtype=torch.long)
                    else:
                        face_to_edge = torch.stack([torch.tensor(e_idx, dtype=torch.long), torch.tensor(f_idx, dtype=torch.long)], dim=0)
                else:
                    face_to_edge = torch.zeros((2, 0), dtype=torch.long)

            if 'face_to_face' in data.files:
                face_to_face = torch.from_numpy(data['face_to_face']).long()
            else:
                face_to_face = torch.zeros((2, 0), dtype=torch.long)

            brep_data = BRepData(
                vertices=vertices,
                edges=torch.from_numpy(std_features['edges']).float(),
                faces=torch.from_numpy(std_features['faces']).float(),
                edge_to_vertex=edge_to_vertex,
                face_to_edge=face_to_edge,
                face_to_face=face_to_face
            )

            valid_face_indices: List[int] = []
            face_id: Optional[int] = None
            points_on_face = None
            # 1) Если есть готовые face_samples
            if 'face_samples' in data.files:
                face_samples = data['face_samples']
                valid_face_indices = [i for i, s in enumerate(face_samples) if getattr(s, 'shape', [0])[0] > 0]
                if valid_face_indices:
                    face_id = int(np.random.choice(valid_face_indices))
                    points_on_face = face_samples[face_id]
            
            # 2) Иначе пытаемся построить из face_point_grids (UV->XYZ)
            if points_on_face is None and 'face_point_grids' in data.files:
                fpg = data['face_point_grids']  # ожидаем форму [F, C, H, W] или объектный массив
                # кандидаты — те лица, где можно извлечь XYZ
                candidate_indices = []
                for i in range(len(fpg)):
                    gi = fpg[i]
                    if isinstance(gi, np.ndarray) and gi.size > 0:
                        candidate_indices.append(i)
                if candidate_indices:
                    face_id = int(np.random.choice(candidate_indices))
                    g = fpg[face_id]
                    if isinstance(g, np.ndarray):
                        if g.ndim == 3 and g.shape[-1] == 3:
                            # [H, W, 3]
                            H, W, _ = g.shape
                            uu = np.linspace(0.0, 1.0, num=W, dtype=np.float32) if W > 1 else np.array([0.0], dtype=np.float32)
                            vv = np.linspace(0.0, 1.0, num=H, dtype=np.float32) if H > 1 else np.array([0.0], dtype=np.float32)
                            U, V = np.meshgrid(uu, vv)
                            uv = np.stack([U, V], axis=-1).reshape(-1, 2)
                            xyz = g.reshape(-1, 3).astype(np.float32)
                            points_on_face = np.concatenate([uv, xyz], axis=1)
                        elif g.ndim == 3 and g.shape[0] >= 3:
                            # [C>=3, H, W] — каналы-первый (например, x,y,z, ...)
                            C, H, W = g.shape
                            xyz = np.stack([g[0], g[1], g[2]], axis=-1).reshape(-1, 3).astype(np.float32)
                            uu = np.linspace(0.0, 1.0, num=W, dtype=np.float32) if W > 1 else np.array([0.0], dtype=np.float32)
                            vv = np.linspace(0.0, 1.0, num=H, dtype=np.float32) if H > 1 else np.array([0.0], dtype=np.float32)
                            U, V = np.meshgrid(uu, vv)
                            uv = np.stack([U, V], axis=-1).reshape(-1, 2)
                            points_on_face = np.concatenate([uv, xyz], axis=1)
                        elif g.ndim == 4 and g.shape[1] >= 3:
                            # [F?, C, H, W] — маловероятно для одно-лицевого среза, но на всякий случай
                            C = g.shape[1]
                            H, W = g.shape[2], g.shape[3]
                            xyz = np.stack([g[0, 0], g[0, 1], g[0, 2]], axis=-1).reshape(-1, 3).astype(np.float32)
                            uu = np.linspace(0.0, 1.0, num=W, dtype=np.float32) if W > 1 else np.array([0.0], dtype=np.float32)
                            vv = np.linspace(0.0, 1.0, num=H, dtype=np.float32) if H > 1 else np.array([0.0], dtype=np.float32)
                            U, V = np.meshgrid(uu, vv)
                            uv = np.stack([U, V], axis=-1).reshape(-1, 2)
                            points_on_face = np.concatenate([uv, xyz], axis=1)

        if points_on_face is None or face_id is None:
            # Если не удалось найти валидные точки, пробуем другой файл
            return self.__getitem__(np.random.randint(0, len(self)))

        # Сэмплирование точек на выбранной грани
        num_points = points_on_face.shape[0]
        indices = np.random.choice(num_points, self.points_per_sample, replace=(num_points < self.points_per_sample))
        sampled_points = points_on_face[indices]

        return {
            "brep_data": brep_data,
            "face_id": int(face_id),
            "uv_coords": torch.from_numpy(sampled_points[:, :2]).float(),
            "xyz_coords": torch.from_numpy(sampled_points[:, 2:]).float()
        }

class ReconstructionDataCollator:
    """
    Собирает батч, может запускать энкодер (c/без градиента) и формирует данные для декодера.
    """
    def __init__(self, encoder: 'CustomBRepEncoder', with_grad: bool = False):
        self.encoder = encoder
        self.with_grad = with_grad
        # Режим модели зададим во время вызова, чтобы поддерживать train/eval

    def __call__(self, batch: List[Dict[str, Any]]) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        # Отфильтровываем None элементы, которые могли появиться из-за рекурсии в __getitem__
        batch = [item for item in batch if item is not None]
        if not batch:
            return None, None, None

        brep_data_list = [item["brep_data"] for item in batch]
        batched_brep_data = batch_brep_data(brep_data_list)

        device = next(self.encoder.parameters()).device
        for key, value in batched_brep_data.__dict__.items():
            if isinstance(value, torch.Tensor):
                batched_brep_data.__dict__[key] = value.to(device)

        # Вычисляем эмбеддинги граней, при необходимости с градиентами
        if self.with_grad:
            self.encoder.train()
            all_face_embeddings = self.encoder(batched_brep_data)
        else:
            self.encoder.eval()
            with torch.no_grad():
                all_face_embeddings = self.encoder(batched_brep_data)

        face_embeddings_batch = []
        uv_coords_batch = torch.stack([item["uv_coords"] for item in batch])
        xyz_coords_batch = torch.stack([item["xyz_coords"] for item in batch])

        for i, item in enumerate(batch):
            graph_face_embeddings = all_face_embeddings[batched_brep_data.face_batch_idx == i]
            face_embedding = graph_face_embeddings[item["face_id"]]
            face_embeddings_batch.append(face_embedding)

        return torch.stack(face_embeddings_batch), uv_coords_batch.to(device), xyz_coords_batch.to(device)


class BatchOnlyCollator:
    """
    Коллатор без вызова энкодера, формирует батч BRepData и сопутствующие тензоры.
    Используется с PyTorch Lightning: модель сама вызывает энкодер в training_step.
    """
    def __call__(self, batch: List[Dict[str, Any]]):
        batch = [item for item in batch if item is not None]
        if not batch:
            return None

        brep_data_list = [item["brep_data"] for item in batch]
        batched_brep_data = batch_brep_data(brep_data_list)
        face_ids = torch.tensor([item["face_id"] for item in batch], dtype=torch.long)
        uv_coords = torch.stack([item["uv_coords"] for item in batch])
        xyz_coords = torch.stack([item["xyz_coords"] for item in batch])
        return batched_brep_data, face_ids, uv_coords, xyz_coords


class BrepNetFeaturesDataset(Dataset):
    """
    Простой датасет для .npz с ключами BrepNet: face_features/edge_features/coedge_features/face_point_grids/face/edge.
    Возвращает стандартизованные признаки и (uv, xyz) выборки, не формируя полный граф для энкодера.
    """
    def __init__(self, data_dir: Path, stats_path: Path, points_per_sample: int = 512):
        self.data_files = sorted([p for p in data_dir.glob('*.npz')])
        self.standardizer = BrepNetStandardizer(str(stats_path))
        self.points_per_sample = points_per_sample

    def __len__(self):
        return len(self.data_files)

    def _build_samples_from_grids(self, grids: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        samples: List[Tuple[np.ndarray, np.ndarray]] = []
        for g in grids:
            if not isinstance(g, np.ndarray) or g.size == 0:
                continue
            if g.ndim == 3 and g.shape[-1] == 3:
                H, W, _ = g.shape
                uu = np.linspace(0.0, 1.0, num=W, dtype=np.float32) if W > 1 else np.array([0.0], dtype=np.float32)
                vv = np.linspace(0.0, 1.0, num=H, dtype=np.float32) if H > 1 else np.array([0.0], dtype=np.float32)
                U, V = np.meshgrid(uu, vv)
                uv = np.stack([U, V], axis=-1).reshape(-1, 2)
                xyz = g.reshape(-1, 3).astype(np.float32)
                N = uv.shape[0]
                if N == 0:
                    continue
                idx = np.random.choice(N, size=min(self.points_per_sample, N), replace=False)
                samples.append((uv[idx], xyz[idx]))
            elif g.ndim == 2 and g.shape[-1] == 3:
                N = g.shape[0]
                if N == 0:
                    continue
                u = np.linspace(0.0, 1.0, num=N, dtype=np.float32)
                uv = np.stack([u, np.zeros_like(u)], axis=-1)
                idx = np.random.choice(N, size=min(self.points_per_sample, N), replace=False)
                samples.append((uv[idx], g[idx].astype(np.float32)))
        return samples

    def __getitem__(self, idx):
        p = self.data_files[idx]
        with np.load(p, allow_pickle=True) as d:
            # Standardize features if present
            out: Dict[str, Any] = {"path": str(p)}
            if all(k in d.files for k in ("face_features","edge_features","coedge_features")):
                raw = {
                    'face_features': d['face_features'],
                    'edge_features': d['edge_features'],
                    'coedge_features': d['coedge_features'],
                }
                standardized = self.standardizer.standardize_data(raw)
                out['face_features'] = standardized['face_features']
                out['edge_features'] = standardized['edge_features']
                out['coedge_features'] = standardized['coedge_features']
                if 'face' in d.files and 'edge' in d.files:
                    out['pooled_face_edge'] = self.standardizer.pool_edge_data_onto_faces({
                        'face_features': standardized['face_features'],
                        'edge_features': standardized['edge_features'],
                        'face': d['face'],
                        'edge': d['edge'],
                    })
            # UV/XYZ from grids if present
            if 'face_point_grids' in d.files:
                uv_xyz = self._build_samples_from_grids(d['face_point_grids'])
                out['uv_xyz'] = uv_xyz
            print("face_features shape:", out['face_features'].shape)
        return out