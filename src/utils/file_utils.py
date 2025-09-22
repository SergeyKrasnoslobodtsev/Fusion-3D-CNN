import json
import numpy as np
from pathlib import Path

def get_files(step_path: Path, extensions=("stp", "step")):
    """
    Получает список файлов в заданной директории.

    Args:
        step_path (Path): Путь к директории с 3D моделями.
        extensions (tuple): Расширения файлов для поиска.

    Returns:
        list[Path]: Список найденных файлов.
    """
    return [f for ext in extensions for f in step_path.glob(f"**/*.{ext}")]

def load_json(pathname):
    with open(pathname, "r") as fp:
        return json.load(fp)
    

def filter_unconverted_files(files: list[Path], output_path: Path):
    """
    Фильтрует файлы, которые уже были конвертированы в формат .npz.

    Args:
        files (list[Path]): Список входных файлов.
        output_path (Path): Директория с выходными файлами.

    Returns:
        list[Path]: Файлы, которые нужно конвертировать.
    """
    return [file for file in files if not (output_path / f"{file.stem}.npz").exists()]


def save_npz_data_without_uvnet_features(output_pathname, data):
    num_faces = data["face_features"].shape[0]
    num_coedges = data["coedge_features"].shape[0]

    dummy_face_point_grids = np.zeros((num_faces, 10, 10, 7))
    dummy_coedge_point_grids = np.zeros((num_coedges, 10, 12))
    dummy_coedge_lcs = np.zeros((num_coedges, 4, 4))
    dummy_coedge_scale_factors = np.zeros((num_coedges))
    dummy_coedge_reverse_flags = np.zeros((num_coedges))
    np.savez(
        output_pathname, 
        face_features = data["face_features"],
        face_point_grids = dummy_face_point_grids,
        edge_features = data["edge_features"],
        coedge_features = data["coedge_features"], 
        coedge_point_grids = dummy_coedge_point_grids,
        coedge_lcs = dummy_coedge_lcs,
        coedge_scale_factors = dummy_coedge_scale_factors,
        coedge_reverse_flags = dummy_coedge_reverse_flags,
        next = data["coedge_to_next"],
        mate = data["coedge_to_mate"],
        face = data["coedge_to_face"],
        edge = data["coedge_to_edge"],
        savez_compressed = True
    ) 

def load_npz_data(npz_file):
    with np.load(npz_file) as data:
        npz_data = {
            "face_features": data["face_features"],
            "face_point_grids": data["face_point_grids"],
            "edge_features": data["edge_features"],
            "coedge_features": data["coedge_features"], 
            "coedge_point_grids": data["coedge_point_grids"],
            "coedge_lcs": data["coedge_lcs"],
            "coedge_scale_factors": data["coedge_scale_factors"],
            "coedge_reverse_flags": data["coedge_reverse_flags"],
            "coedge_to_next": data["next"], 
            "coedge_to_mate": data["mate"], 
            "coedge_to_face": data["face"], 
            "coedge_to_edge": data["edge"]
        }
    return npz_data


def load_labels(label_pathname):
    labels = np.loadtxt(label_pathname, dtype=np.int64)
    if labels.ndim == 0:
        labels = np.expand_dims(labels, 0)
    return labels