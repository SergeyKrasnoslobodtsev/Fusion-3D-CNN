from typing import Optional
import numpy as np
import tqdm
from pathlib import Path
from torch.utils.data import Dataset
from ..utils.running_stats import RunningStats

def stats_to_json(stats):
    return [{"mean": s.mean(), "standard_deviation": s.standard_deviation()} for s in stats]

def append_to_stats(arr, stats):
    num_entities, num_features = arr.shape
    if not stats:
        stats.extend([RunningStats() for _ in range(num_features)])
    
    assert len(stats) == num_features, "Несоответствие количества признаков"

    for i in range(num_entities):
        for j in range(num_features):
            stats[j].push(arr[i, j])

# This function to standardize features is similar to what we do in the dataloader
def standardize_features(feature_tensor, stats):
    num_features = len(stats)
    assert feature_tensor.shape[1] == num_features
    means = np.zeros(num_features)
    sds = np.zeros(num_features)
    eps = 1e-7
    for index, s in enumerate(stats):
        assert s["standard_deviation"] > eps, "Feature has zero standard deviation"
        means[index] = s["mean"]
        sds[index] = s["standard_deviation"]

    # We need to broadcast means and sds over the number of entities
    means_x = np.expand_dims(means, axis=0)
    sds_x = np.expand_dims(sds, axis=0)
    feature_tensor_zero_mean = feature_tensor - means_x
    feature_tensor_standadized = feature_tensor_zero_mean / sds_x

    # Convert the tensors to floats after standardization 
    return feature_tensor_standadized
    
def standarize_data(data, feature_standardization):
    data["face_features"] = standardize_features(data["face_features"], feature_standardization["face_features"])
    data["edge_features"] = standardize_features(data["edge_features"], feature_standardization["edge_features"])
    data["coedge_features"] = standardize_features(data["coedge_features"], feature_standardization["coedge_features"])
    return data
    
def find_faces_to_edges(coedge_to_face, coedge_to_edge):
    faces_to_edges_dict = {}
    for coedge_index in range(coedge_to_face.shape[0]):
        edge = coedge_to_edge[coedge_index]
        face = coedge_to_face[coedge_index]
        if not face in faces_to_edges_dict:
            faces_to_edges_dict[face] = set()
        faces_to_edges_dict[face].add(edge)
    faces_to_edges = []
    for i in range(len(faces_to_edges_dict)):
        assert i in faces_to_edges_dict
        faces_to_edges.append(faces_to_edges_dict[i])
    return faces_to_edges

# This function pools edges features onto all faces which
# are adjacent to that edge
def pool_edge_data_onto_faces(data):
    face_features = data["face_features"]
    edge_features = data["edge_features"]
    coedge_to_face = data["coedge_to_face"] 
    coedge_to_edge = data["coedge_to_edge"]
    for edge in coedge_to_edge:
        assert edge < edge_features.shape[0]
    faces_to_edges = find_faces_to_edges(coedge_to_face, coedge_to_edge)
    face_edge_features = []
    for face_edge_set in faces_to_edges:
        edge_features_for_face  = []
        for edge in face_edge_set:
            edge_features_for_face.append(edge_features[edge])
        pooled_edge_features = np.max(np.stack(edge_features_for_face), axis = 0)
        face_edge_features.append(pooled_edge_features)
    assert len(face_edge_features) == face_features.shape[0]
    face_edge_features = np.stack(face_edge_features)
    return np.concatenate([face_features, face_edge_features], axis = 1)

def load_npz_brepnet(npz_file):
    with np.load(npz_file) as data:
        npz_data = {
            "face_features": data["face_features"], # признаки граней
            "face_point_grids": data["face_point_grids"], # координаты точек на гранях
            "edge_features": data["edge_features"], # признаки рёбер
            "edge_point_grids": data["edge_point_grids"], # координаты точек на рёбрах
            "coedge_features": data["coedge_features"], # признаки ко-рёбер
            "coedge_point_grids": data["coedge_point_grids"], # координаты точек на ко-рёбрах
            "coedge_lcs": data["coedge_lcs"], # локальные системы координат ко-рёбер
            "coedge_scale_factors": data["coedge_scale_factors"], # масштабные коэффициенты ко-рёбер
            "coedge_reverse_flags": data["coedge_reverse_flags"], # флаги реверса ко-рёбер
            "coedge_to_next": data["next"],  # индексы следующих ко-рёбер
            "coedge_to_mate": data["mate"],  # индексы сопряжённых ко-рёбер
            "coedge_to_face": data["face"],  # индексы граней для ко-рёбер
            "coedge_to_edge": data["edge"]   # индексы рёбер для ко-рёбер
        }
    return npz_data

def load_npz_dino(npz_file):
        with np.load(npz_file) as data:
            npz_data = data["views"]
        return npz_data

class CustomDataset(Dataset):
    """
    Датасет, объединяющий признаки из BRepNet и DINO для каждой модели.
    """
    def __init__(
        self,
        step_data: Path,
        brepnet_data: Path,
        dino_data: Path,
        compute_stats: bool = False,
        feature_stats: Optional[dict] = None,
        apply_standardization: bool = False,
    ):
        self.step_data = step_data
        self.brepnet_data = brepnet_data
        self.dino_data = dino_data
        self.apply_standardization = apply_standardization

        def get_clean_id(filename: str):
            name = Path(filename).stem
            if name.endswith('.prt'):
                name = name[:-4]
            return name

        step_files = {get_clean_id(f.name): f for f in step_data.glob("*.stp")}
        brepnet_files = {get_clean_id(f.name): f for f in brepnet_data.glob("*.npz")}
        dino_files = {get_clean_id(f.name): f for f in dino_data.glob("*.npz")}
        common_ids = sorted(set(step_files) & set(brepnet_files) & set(dino_files))

        face_stats, edge_stats, coedge_stats = [], [], []
        self.data = []
        for model_id in tqdm.tqdm(common_ids, desc="Loading dataset"):
            brepnet_path = brepnet_files[model_id]
            step_path = step_files[model_id]
            dino_path = dino_files[model_id]
            brepnet_feats = load_npz_brepnet(brepnet_path)
            dino_feats = load_npz_dino(dino_path)
            self.data.append(
                {
                    "model_id": model_id,
                    "step_path": step_path,
                    "brepnet_features": brepnet_feats,
                    "dino_features": dino_feats,
                }
            )
            if compute_stats:
                if "face_features" in brepnet_feats:
                    append_to_stats(brepnet_feats["face_features"], face_stats)
                if "edge_features" in brepnet_feats:
                    append_to_stats(brepnet_feats["edge_features"], edge_stats)
                if "coedge_features" in brepnet_feats:
                    append_to_stats(brepnet_feats["coedge_features"], coedge_stats)

        if compute_stats:
            self.std_brepnet = {
                "face_features": stats_to_json(face_stats),
                "edge_features": stats_to_json(edge_stats),
                "coedge_features": stats_to_json(coedge_stats),
            }
        else:
            self.std_brepnet = feature_stats

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        brepnet_features = sample["brepnet_features"].copy()

        # Стандартизация только если есть статистика и включён флаг
        if self.apply_standardization and self.std_brepnet is not None:
            data = standarize_data(brepnet_features, self.std_brepnet)
        else:
            data = brepnet_features

        pooled_face_edge_features = pool_edge_data_onto_faces(data)

        return {
            "model_id": sample["model_id"],
            "step_path": sample["step_path"],
            "brepnet_features": pooled_face_edge_features,
            "dino_features": sample["dino_features"],
        }