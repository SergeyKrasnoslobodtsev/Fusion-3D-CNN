import torch
import numpy as np
from pathlib import Path
import json
import tempfile
import traceback

from src.SSL_BrepNet.dataset_loader import BRepData
from src.SSL_BrepNet.normlization import BrepNetStandardizer
from src.SSL_BrepNet.model.encoder import CustomBRepEncoder
from src.SSL_BrepNet.model.decoder import ConditionalDecoder

def test_ssl_encoder() -> bool:
    """Тест SSL энкодера с правильными индексами"""
    try:
        print("🔧 Тестирование SSL энкодера...")
        
        encoder = CustomBRepEncoder(
            v_in_width=3,
            e_in_width=2, 
            f_in_width=5,  
            out_width=64,
            num_layers=1,
            use_attention=False
        )
        
        # Правильные тестовые данные с валидными индексами
        num_vertices = 4
        num_edges = 6
        num_faces = 3
        
        brep_data = BRepData(
            vertices=torch.randn(num_vertices, 3),
            edges=torch.randn(num_edges, 2),
            faces=torch.randn(num_faces, 5),
            # Рёбра соединяют вершины в пределах [0, num_vertices)
            edge_to_vertex=torch.tensor([
                [0, 1, 2, 0, 1, 2],  # src vertices
                [1, 2, 3, 3, 0, 1]   # dst vertices  
            ], dtype=torch.long),
            # Грани соединяются с рёбрами в пределах [0, num_edges)
            face_to_edge=torch.tensor([
                [0, 0, 1, 1, 2, 2],  # face indices (в пределах num_faces)
                [0, 1, 2, 3, 4, 5]   # edge indices (в пределах num_edges)
            ], dtype=torch.long),
            # Грани соединяются с гранями в пределах [0, num_faces)
            face_to_face=torch.tensor([
                [0, 1, 2],  # src faces
                [1, 2, 0]   # dst faces
            ], dtype=torch.long)
        )
        
        # Прямой проход
        with torch.no_grad():
            output = encoder(brep_data)
        
        # Проверки
        assert output.shape == (num_faces, 64), f"Ожидали ({num_faces}, 64), получили {output.shape}"
        assert not torch.isnan(output).any(), "Энкодер выдает NaN"
        assert not torch.isinf(output).any(), "Энкодер выдает Inf"
        
        print("✅ SSL энкодер работает корректно")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка в SSL энкодере: {e}")
        traceback.print_exc()
        return False

def test_ssl_decoder() -> bool:
    """Тест SSL декодера"""
    try:
        print("🔧 Тестирование SSL декодера...")
        
        decoder = ConditionalDecoder(
            latent_size=64,
            hidden_dims=[128, 128, 64],
            uv_input_dim=2,
            output_dim=4
        )
        
        # Тестовые данные
        uv_coords = torch.randn(100, 2)
        latent_vector = torch.randn(64)
        
        # Прямой проход
        with torch.no_grad():
            output = decoder(uv_coords, latent_vector)
        
        # Проверки
        assert output.shape == (100, 4), f"Ожидали (100, 4), получили {output.shape}"
        assert not torch.isnan(output).any(), "Декодер выдает NaN"
        
        print("✅ SSL декодер работает корректно")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка в SSL декодере: {e}")
        traceback.print_exc()
        return False

def test_standardizer() -> bool:
    """Тест стандартизатора"""
    try:
        print("🔧 Тестирование стандартизатора...")
        
        # Создаем временную статистику
        stats = {
            "feature_standardization": {
                "face_features": [
                    {"mean": 0.5, "standard_deviation": 1.2},
                    {"mean": -0.3, "standard_deviation": 0.8},
                    {"mean": 1.1, "standard_deviation": 2.1}
                ],
                "edge_features": [
                    {"mean": 0.2, "standard_deviation": 0.9},
                    {"mean": -0.1, "standard_deviation": 1.5}
                ],
                "coedge_features": [
                    {"mean": 0.0, "standard_deviation": 1.0}
                ]
            }
        }
        
        # Сохраняем во временный файл
        temp_dir = Path(tempfile.mkdtemp())
        stats_path = temp_dir / "stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f)
        
        standardizer = BrepNetStandardizer(str(stats_path))
        
        # Тестовые данные
        test_data = {
            'face_features': np.random.randn(5, 3),
            'edge_features': np.random.randn(8, 2),
            'coedge_features': np.random.randn(10, 1)
        }
        
        # Стандартизация
        standardized = standardizer.standardize_data(test_data)
        
        # Проверки
        assert standardized['face_features'].shape == (5, 3)
        assert standardized['edge_features'].shape == (8, 2)
        
        print("✅ Стандартизатор работает корректно")
        
        # Очистка
        import shutil
        shutil.rmtree(temp_dir)
        return True
        
    except Exception as e:
        print(f"❌ Ошибка в стандартизаторе: {e}")
        traceback.print_exc()
        return False

def test_pooling() -> bool:
    """Тест пулинга рёбер на грани"""
    try:
        print("🔧 Тестирование пулинга рёбер...")
        
        # Создаем временную статистику (как выше)
        stats = {
            "feature_standardization": {
                "face_features": [{"mean": 0.0, "standard_deviation": 1.0}] * 3,
                "edge_features": [{"mean": 0.0, "standard_deviation": 1.0}] * 2,
                "coedge_features": [{"mean": 0.0, "standard_deviation": 1.0}]
            }
        }
        
        temp_dir = Path(tempfile.mkdtemp())
        stats_path = temp_dir / "stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f)
        
        standardizer = BrepNetStandardizer(str(stats_path))
        
        # Тестовые данные
        data = {
            'face_features': np.random.randn(3, 3),  # 3 грани
            'edge_features': np.random.randn(5, 2),  # 5 рёбер
            'face': np.array([0, 0, 1, 1, 2]),       # coedge->face
            'edge': np.array([0, 1, 2, 3, 4])        # coedge->edge
        }
        
        # Пулинг
        pooled = standardizer.pool_edge_data_onto_faces(data)
        
        # Проверки
        expected_shape = (3, 5)  # 3 face + 2 edge = 5
        assert pooled.shape == expected_shape, f"Ожидали {expected_shape}, получили {pooled.shape}"
        
        print("✅ Пулинг рёбер работает корректно")
        
        # Очистка
        import shutil
        shutil.rmtree(temp_dir)
        return True
        
    except Exception as e:
        print(f"❌ Ошибка в пулинге: {e}")
        traceback.print_exc()
        return False

def test_integration() -> bool:
    """Исправленный интеграционный тест"""
    try:
        print("🔧 Интеграционный тест...")
        
        # Энкодер и декодер
        encoder = CustomBRepEncoder(3, 2, 5, 64, 1)
        decoder = ConditionalDecoder(64, [128, 64])
        
        # Правильные тестовые данные
        num_vertices = 4
        num_edges = 5  
        num_faces = 3
        
        brep_data = BRepData(
            vertices=torch.randn(num_vertices, 3),
            edges=torch.randn(num_edges, 2), 
            faces=torch.randn(num_faces, 5),
            # Валидные индексы для edge_to_vertex
            edge_to_vertex=torch.tensor([
                [0, 1, 2, 3, 0],  # src: все < num_vertices
                [1, 2, 3, 0, 1]   # dst: все < num_vertices
            ], dtype=torch.long),
            # Валидные индексы для face_to_edge
            face_to_edge=torch.tensor([
                [0, 0, 1, 1, 2],  # face: все < num_faces
                [0, 1, 2, 3, 4]   # edge: все < num_edges
            ], dtype=torch.long),
            # Валидные индексы для face_to_face
            face_to_face=torch.tensor([
                [0, 1],  # src: все < num_faces
                [1, 2]   # dst: все < num_faces
            ], dtype=torch.long)
        )
        
        with torch.no_grad():
            # Энкодирование
            face_embeddings = encoder(brep_data)
            model_embedding = face_embeddings.mean(dim=0)
            
            # Декодирование
            uv_coords = torch.rand(50, 2)
            reconstructed = decoder(uv_coords, model_embedding)
        
        assert reconstructed.shape == (50, 4)
        assert not torch.isnan(reconstructed).any()
        
        print("✅ Интеграция работает корректно")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка в интеграции: {e}")
        traceback.print_exc()
        return False
def create_valid_brep_data(num_vertices: int, num_edges: int, num_faces: int) -> BRepData:
    """
    Создание валидных BRepData с правильными индексами
    """
    return BRepData(
        vertices=torch.randn(num_vertices, 3),
        edges=torch.randn(num_edges, 2),
        faces=torch.randn(num_faces, 5),
        
        # edge_to_vertex: каждое ребро соединяет 2 вершины
        edge_to_vertex=torch.stack([
            torch.randint(0, num_vertices, (num_edges,)),  # src vertices
            torch.randint(0, num_vertices, (num_edges,))   # dst vertices
        ]),
        
        # face_to_edge: некоторые грани соединяются с некоторыми рёбрами
        face_to_edge=torch.stack([
            torch.randint(0, num_faces, (min(num_faces * 3, 20),)),  # face indices
            torch.randint(0, num_edges, (min(num_faces * 3, 20),))   # edge indices
        ]),
        
        # face_to_face: некоторые грани соединяются с другими гранями  
        face_to_face=torch.stack([
            torch.randint(0, num_faces, (min(num_faces * 2, 10),)),  # src faces
            torch.randint(0, num_faces, (min(num_faces * 2, 10),))   # dst faces
        ])
    )

def run_corrected_tests() -> None:
    """Запуск исправленных тестов"""
    print("🚀 Запуск исправленных тестов SSL компонентов...\n")
    
    # Тест энкодера с валидными данными
    print("🔧 Тестирование SSL энкодера (исправлено)...")
    try:
        encoder = CustomBRepEncoder(3, 2, 5, 64, 1, False)
        brep_data = create_valid_brep_data(num_vertices=6, num_edges=8, num_faces=4)
        
        with torch.no_grad():
            output = encoder(brep_data)
        
        assert output.shape == (4, 64)
        print("✅ SSL энкодер работает корректно")
        
    except Exception as e:
        print(f"❌ Ошибка в SSL энкодере: {e}")
        traceback.print_exc()
    
    print()
    
    # Тест интеграции с валидными данными
    print("🔧 Интеграционный тест (исправлено)...")
    try:
        encoder = CustomBRepEncoder(3, 2, 5, 64, 1)
        decoder = ConditionalDecoder(64, [128, 64])
        brep_data = create_valid_brep_data(num_vertices=5, num_edges=7, num_faces=3)
        
        with torch.no_grad():
            face_embeddings = encoder(brep_data)
            model_embedding = face_embeddings.mean(dim=0)
            
            uv_coords = torch.rand(50, 2)
            reconstructed = decoder(uv_coords, model_embedding)
        
        assert reconstructed.shape == (50, 4)
        print("✅ Интеграция работает корректно")
        
    except Exception as e:
        print(f"❌ Ошибка в интеграции: {e}")
        traceback.print_exc()
    
    print("\n🎉 Исправленные тесты завершены!")

def run_all_tests() -> None:
    """Запуск всех тестов"""
    print("🚀 Запуск тестов SSL компонентов...\n")
    
    tests = [
        test_ssl_encoder,
        test_ssl_decoder, 
        test_standardizer,
        test_pooling,
        test_integration
    ]
    
    results = []
    for test_func in tests:
        success = test_func()
        results.append(success)
        print()
    
    # Итоги
    passed = sum(results)
    total = len(results)
    
    print(f"📊 Результаты: {passed}/{total} тестов пройдено")
    
    if passed == total:
        print("🎉 Все тесты успешно пройдены!")
    else:
        print("⚠️ Есть проблемы, которые нужно исправить")

# Запуск тестов
run_all_tests()
run_corrected_tests()
