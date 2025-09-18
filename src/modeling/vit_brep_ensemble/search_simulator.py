import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json

from .models.self_supervised_ensemble import SelfSupervisedFusionModel
from .data_module.enhanced_data_loader import EnhancedFusionDataModule

class SearchSimulator:
    """Симулятор поиска похожих 3D моделей"""
    
    def __init__(
        self, 
        model: SelfSupervisedFusionModel, 
        data_module: EnhancedFusionDataModule,
        device: torch.device
    ):
        self.model = model
        self.data_module = data_module
        self.device = device
        
        # Кеш для эмбеддингов
        self.embeddings_cache: Optional[Dict[str, torch.Tensor]] = None
        self.item_ids: Optional[List[str]] = None
        
    def build_index(self, use_train: bool = True, use_val: bool = True) -> None:
        """Строит индекс эмбеддингов для всех моделей"""
        
        print("🔄 Строим индекс эмбеддингов...")
        
        embeddings = []
        item_ids = []
        
        # Собираем данные из train и val
        loaders = []
        if use_train:
            loaders.append(("train", self.data_module.train_dataloader()))
        if use_val:
            loaders.append(("val", self.data_module.val_dataloader()))
        
        self.model.eval()
        with torch.no_grad():
            for split_name, loader in loaders:
                print(f"  Обрабатываем {split_name} split...")
                
                for batch in tqdm(loader, desc=f"Extracting {split_name} embeddings"):
                    # Переносим на device
                    batch_device = {
                        k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()
                    }
                    
                    # Получаем эмбеддинги
                    outputs = self.model(batch_device)
                    batch_embeddings = outputs['projections'].cpu()  # Используем нормализованные проекции
                    
                    embeddings.append(batch_embeddings)
                    item_ids.extend(batch['item_id'])
        
        # Объединяем все эмбеддинги
        all_embeddings = torch.cat(embeddings, dim=0)
        
        # Сохраняем в кеш
        self.embeddings_cache = {
            item_id: embedding for item_id, embedding in zip(item_ids, all_embeddings)
        }
        self.item_ids = item_ids
        
        print(f"✅ Индекс построен: {len(self.item_ids)} моделей, размерность {all_embeddings.shape[1]}")
    
    def search_similar(
        self, 
        query_item_id: str, 
        top_k: int = 10,
        exclude_self: bool = True
    ) -> List[Tuple[str, float]]:
        """
        Поиск топ-K похожих моделей
        
        Args:
            query_item_id: ID модели для поиска
            top_k: Количество возвращаемых результатов
            exclude_self: Исключить саму модель из результатов
            
        Returns:
            List[(item_id, similarity_score)] отсортированный по убыванию схожести
        """
        
        if self.embeddings_cache is None:
            raise RuntimeError("Сначала постройте индекс с помощью build_index()")
        
        if query_item_id not in self.embeddings_cache:
            raise KeyError(f"Модель '{query_item_id}' не найдена в индексе")
        
        # Получаем эмбеддинг запроса
        query_embedding = self.embeddings_cache[query_item_id]
        
        # Вычисляем сходство со всеми моделями
        similarities = []
        for item_id, embedding in self.embeddings_cache.items():
            if exclude_self and item_id == query_item_id:
                continue
                
            # Косинусное сходство (эмбеддинги уже нормализованы)
            similarity = torch.dot(query_embedding, embedding).item()
            similarities.append((item_id, similarity))
        
        # Сортируем по убыванию сходства
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def search_by_embedding(
        self, 
        query_embedding: torch.Tensor, 
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """Поиск по готовому эмбеддингу"""
        
        if self.embeddings_cache is None:
            raise RuntimeError("Сначала постройте индекс с помощью build_index()")
        
        # Нормализуем запрос
        query_embedding = F.normalize(query_embedding, p=2, dim=0)
        
        # Вычисляем сходство
        similarities = []
        for item_id, embedding in self.embeddings_cache.items():
            similarity = torch.dot(query_embedding, embedding).item()
            similarities.append((item_id, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def evaluate_retrieval(
        self, 
        test_queries: Optional[List[str]] = None,
        top_k: int = 10
    ) -> Dict[str, float]:
        """
        Оценка качества поиска (если есть ground truth)
        """
        
        if test_queries is None:
            # Используем случайную выборку из валидации
            test_queries = np.random.choice(self.item_ids, size=min(50, len(self.item_ids)), replace=False).tolist() # type: ignore
        
        print(f"🧪 Оцениваем качество поиска на {len(test_queries)} запросах...") # type: ignore 
        
        # Простая метрика: разнообразность результатов
        diversity_scores = []
        avg_similarities = []
        
        for query_id in tqdm(test_queries):
            results = self.search_similar(query_id, top_k=top_k)
            
            if not results:
                continue
                
            # Средняя схожесть
            similarities = [score for _, score in results]
            avg_similarities.append(np.mean(similarities))
            
            # Разнообразность (стандартное отклонение схожести)
            diversity_scores.append(np.std(similarities))
        
        metrics = {
            'avg_similarity': np.mean(avg_similarities),
            'avg_diversity': np.mean(diversity_scores),
            'num_queries': len(test_queries) # type: ignore
        }
        
        return metrics
    
    def visualize_search_results(
        self, 
        query_item_id: str, 
        top_k: int = 10,
        save_path: Optional[Path] = None
    ) -> None:
        """Визуализация результатов поиска"""
        
        results = self.search_similar(query_item_id, top_k=top_k)
        
        # Создаём DataFrame для удобства
        df = pd.DataFrame(results, columns=['item_id', 'similarity'])
        df['rank'] = range(1, len(df) + 1)
        
        # Строим график
        plt.figure(figsize=(12, 6))
        
        # График 1: Распределение схожести
        plt.subplot(1, 2, 1)
        sns.barplot(data=df, x='rank', y='similarity', palette='viridis')
        plt.title(f'Топ-{top_k} похожих моделей для {query_item_id}')
        plt.xlabel('Ранг')
        plt.ylabel('Косинусная схожесть')
        plt.xticks(rotation=45)
        
        # График 2: Таблица результатов
        plt.subplot(1, 2, 2)
        plt.axis('tight')
        plt.axis('off')
        table_data = df[['rank', 'item_id', 'similarity']].round(3)
        table = plt.table(
            cellText=table_data.values,
            colLabels=['Ранг', 'ID модели', 'Схожесть'],
            cellLoc='center',
            loc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 График сохранён: {save_path}")
        
        plt.show()
    
    def save_index(self, path: Path) -> None:
        """Сохранить индекс эмбеддингов"""
        if self.embeddings_cache is None:
            raise RuntimeError("Индекс не построен")
            
        # Преобразуем в numpy для сериализации
        embeddings_np = {
            item_id: embedding.numpy() 
            for item_id, embedding in self.embeddings_cache.items()
        }
        
        index_data = {
            'embeddings': embeddings_np,
            'item_ids': self.item_ids,
            'embedding_dim': list(embeddings_np.values())[0].shape[0]
        }
        
        np.savez_compressed(path, **index_data)
        print(f"💾 Индекс сохранён: {path}")
    
    def load_index(self, path: Path) -> None:
        """Загрузить индекс эмбеддингов"""
        data = np.load(path, allow_pickle=True)
        
        # Восстанавливаем эмбеддинги
        embeddings_np = data['embeddings'].item()
        self.embeddings_cache = {
            item_id: torch.from_numpy(embedding) 
            for item_id, embedding in embeddings_np.items()
        }
        self.item_ids = data['item_ids'].tolist()

        print(f"📂 Индекс загружен: {len(self.item_ids)} моделей") # type: ignore