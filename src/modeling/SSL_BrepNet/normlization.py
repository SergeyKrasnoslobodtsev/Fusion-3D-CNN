import json
import numpy as np
from typing import Dict, List

class BrepNetStandardizer:
    """
    Стандартизация BrepNet признаков Z_score
    """
    
    def __init__(self, standardization_stats_path: str) -> None:
        """
        Args:
            standardization_stats_path: Путь к JSON файлу со статистикой
        """
        with open(standardization_stats_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.feature_standardization = data['feature_standardization']
        self._stats_edges = self.feature_standardization.get('edge_features') 
        self._stats_faces = self.feature_standardization.get('face_features') 
        self._stats_coedges = self.feature_standardization.get('coedge_features')

    def standardize_coedge_features(
        self, 
        feature_tensor: np.ndarray
    ) -> np.ndarray:
        """
        Стандартизация признаков для coedge - точная копия из вашего блокнота

        Args:
            feature_tensor: [num_entities, num_features] - сырые признаки

        Returns:
            standardized_tensor: [num_entities, num_features] - стандартизированные признаки
        """
        return self.standardize_features(feature_tensor, self._stats_coedges)
    
    def standardize_edge_features(
        self, 
        feature_tensor: np.ndarray
    ) -> np.ndarray:
        """
        Стандартизация признаков для edge - точная копия из вашего блокнота

        Args:
            feature_tensor: [num_entities, num_features] - сырые признаки

        Returns:
            standardized_tensor: [num_entities, num_features] - стандартизированные признаки
        """
        return self.standardize_features(feature_tensor, self._stats_edges)
    
    def standardize_face_features(
        self, 
        feature_tensor: np.ndarray
    ) -> np.ndarray:
        """
        Стандартизация признаков для face - точная копия из вашего блокнота

        Args:
            feature_tensor: [num_entities, num_features] - сырые признаки

        Returns:
            standardized_tensor: [num_entities, num_features] - стандартизированные признаки
        """
        return self.standardize_features(feature_tensor, self._stats_faces)

    def standardize_features(
        self, 
        feature_tensor: np.ndarray, 
        stats: List[Dict]
    ) -> np.ndarray:
        """
        Стандартизация признаков - точная копия из вашего блокнота
        
        Args:
            feature_tensor: [num_entities, num_features] - сырые признаки
            stats: Список статистик для каждого признака
            
        Returns:
            standardized_tensor: [num_entities, num_features] - стандартизированные признаки
        """
        num_features = len(stats)
        assert feature_tensor.shape[1] == num_features
        
        means = np.zeros(num_features)
        sds = np.zeros(num_features)
        eps = 1e-7
        
        for index, s in enumerate(stats):
            assert s['standard_deviation'] < eps, "Feature has zero standard deviation"
            means[index] = s['mean']
            sds[index] = s['standard_deviation']
        
        # Broadcast means и sds для всех объектов
        means_x = np.expand_dims(means, axis=0)
        sds_x = np.expand_dims(sds, axis=0)
        
        # z-score нормализация
        feature_tensor_zero_mean = feature_tensor - means_x
        feature_tensor_standardized = feature_tensor_zero_mean / sds_x
        
        return feature_tensor_standardized
    
    def standardize_data(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Стандартизация всех BrepNet данных - точная копия из блокнота
        
        Args:
            data: Словарь с ключами 'face_features', 'edge_features', 'coedge_features'
            
        Returns:
            standardized_data: Стандартизированные данные
        """
        standardized_data = data.copy()
        
        # Стандартизация каждого типа признаков
        standardized_data['face_features'] = self.standardize_features(
            data['face_features'], 
            self.feature_standardization['face_features']
        )
        
        standardized_data['edge_features'] = self.standardize_features(
            data['edge_features'],
            self.feature_standardization['edge_features'] 
        )
        
        standardized_data['coedge_features'] = self.standardize_features(
            data['coedge_features'],
            self.feature_standardization['coedge_features']
        )
        
        return standardized_data
    
    def find_faces_to_edges(
        self, 
        coedge_to_face: np.ndarray, 
        coedge_to_edge: np.ndarray
    ) -> List[set]:
        """
        Поиск связей граней к рёбрам - точная копия из блокнота
        """
        faces_to_edges_dict = {}
        
        for coedge_index in range(coedge_to_face.shape[0]):
            edge = coedge_to_edge[coedge_index]
            face = coedge_to_face[coedge_index]
            
            if face not in faces_to_edges_dict:
                faces_to_edges_dict[face] = set()
            faces_to_edges_dict[face].add(edge)
        
        faces_to_edges = []
        for i in range(len(faces_to_edges_dict)):
            assert i in faces_to_edges_dict
            faces_to_edges.append(faces_to_edges_dict[i])
            
        return faces_to_edges
    
    def pool_edge_data_onto_faces(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Пулинг рёберных признаков на грани - точная копия из блокнота
        """
        face_features = data['face_features']
        edge_features = data['edge_features']  
        coedge_to_face = data['face']  # coedge -> face mapping
        coedge_to_edge = data['edge']  # coedge -> edge mapping
        
        # Проверяем валидность индексов рёбер
        for edge in coedge_to_edge:
            assert edge < edge_features.shape[0]
        
        # Находим связи граней к рёбрам
        faces_to_edges = self.find_faces_to_edges(coedge_to_face, coedge_to_edge)
        
        face_edge_features = []
        for face_edge_set in faces_to_edges:
            edge_features_for_face = []
            for edge in face_edge_set:
                edge_features_for_face.append(edge_features[edge])
            
            # Max pooling по рёбрам для данной грани
            pooled_edge_features = np.max(np.stack(edge_features_for_face), axis=0)
            face_edge_features.append(pooled_edge_features)
        
        assert len(face_edge_features) == face_features.shape[0]
        face_edge_features = np.stack(face_edge_features)
        
        # Конкатенация face + pooled edge признаков
        return np.concatenate([face_features, face_edge_features], axis=1)
