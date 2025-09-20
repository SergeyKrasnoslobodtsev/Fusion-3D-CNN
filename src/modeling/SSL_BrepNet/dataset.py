import os
import json
import torch
import numpy as np
from types import SimpleNamespace
from torch.utils.data import Dataset

def standardize_features(feature_tensor, stats):
    # num_features = len(stats)
    means = np.array([s["mean"] for s in stats])
    sds = np.array([s["standard_deviation"] for s in stats])
    eps = 1e-7
    assert np.all(sds > eps), "Feature has zero standard deviation"
    means_x = np.expand_dims(means, axis=0)
    sds_x = np.expand_dims(sds, axis=0)
    feature_tensor_zero_mean = feature_tensor - means_x
    feature_tensor_standardized = feature_tensor_zero_mean / sds_x
    return feature_tensor_standardized.astype(np.float32)

def standarize_data(data, feature_standardization):
    data["face_features"] = standardize_features(data["face_features"], feature_standardization["face_features"])
    data["edge_features"] = standardize_features(data["edge_features"], feature_standardization["edge_features"])
    data["coedge_features"] = standardize_features(data["coedge_features"], feature_standardization["coedge_features"])
    return data

class BrepNetDataset(Dataset):
    def __init__(self, json_path, feats_path_brep, feats_path_sdf, split="training_set"):
        with open(json_path, encoding="utf-8") as f:
            stats = json.load(f)
        self.features_dir = stats["brepnet_features_dir"]
        self.files = stats[split]
        self.feature_standardization = stats["feature_standardization"] if split == "training_set" else None
        self.train_stats = stats["feature_standardization"]
        self.sdf_dir = feats_path_sdf
        self.brep_dir = feats_path_brep

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_name = self.files[idx]
        brep_path = os.path.join(self.brep_dir, file_name + ".npz")
        sdf_path = os.path.join(self.sdf_dir, file_name + ".npz")
        D = np.load(brep_path, allow_pickle=True)
        S = np.load(sdf_path, allow_pickle=True)

        data = {
            "vertex": D["vertex"],
            "edge_features": D["edge_features"],
            "face_features": D["face_features"],
            "coedge_features": D["coedge_features"],
            "edge_to_vertex": D["edge_to_vertex"],
            "face_to_edge": D["face_to_edge"],
            "face_to_face": D["face_to_face"],
        }
        # Стандартизация только для обучающей выборки
        if self.feature_standardization is not None:
            data = standarize_data(data, self.feature_standardization)
        return SimpleNamespace(
            name=file_name,
            vertices=torch.from_numpy(data["vertex"].astype(np.float32)),
            edges=torch.from_numpy(data["edge_features"].astype(np.float32)),
            faces=torch.from_numpy(data["face_features"].astype(np.float32)),
            edge_to_vertex=torch.from_numpy(data["edge_to_vertex"].astype(np.int64)),
            face_to_edge=torch.from_numpy(data["face_to_edge"][::-1].astype(np.int64)),
            face_to_face=torch.from_numpy(data["face_to_face"].astype(np.int64)),
            sdf_uv=torch.from_numpy(S["uv_faces"].astype(np.float32)),        # [n_faces, n_samples, 2]
            sdf_vals=torch.from_numpy(S["sdf_faces"].astype(np.float32))      # [n_faces, n_samples]
        )