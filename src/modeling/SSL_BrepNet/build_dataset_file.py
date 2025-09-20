"""
Скрипт для создания файла описания комбинированного набора данных
из признаков B-repNet и DINO-ViT.

1. Находит общие модели в папках с признаками brepnet и vit.
2. Разделяет модели на обучающую, валидационную и тестовую выборки.
3. Вычисляет статистики стандартизации (mean/std) *только* для признаков B-repNet
   и *только* по обучающей выборке.
4. Сохраняет итоговый .json файл с описанием набора данных.
"""
import json
import sys
from pathlib import Path
import typer
import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from loguru import logger

from .running_stats import RunningStats 

from ...config import PROCESSED_DATA_DIR, RAW_DATA_DIR



app = typer.Typer()

def save_json_data(pathname, data):
    """Export a data to a json file"""
    with open(pathname, 'w', encoding='utf8') as fp:
        json.dump(data, fp, indent=4, ensure_ascii=False, sort_keys=False)

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

def find_brepnet_standardization(train_model_ids, brepnet_folder: Path):
    """Вычисляет статистики только для признаков B-repNet."""
    face_stats, edge_stats, coedge_stats = [], [], []
    
    logger.info("Вычисление статистик стандартизации для B-repNet...")
    for model_id in tqdm.tqdm(train_model_ids):
        file_path = brepnet_folder / f"{model_id}.npz"
        data = np.load(file_path)
        
        
        if "face_features" in data:
            append_to_stats(data["face_features"], face_stats)
        if "edge_features" in data:
            append_to_stats(data["edge_features"], edge_stats)
        if "coedge_features" in data:
            append_to_stats(data["coedge_features"], coedge_stats)

    return {
        "face_features": stats_to_json(face_stats),
        "edge_features": stats_to_json(edge_stats),
        "coedge_features": stats_to_json(coedge_stats),
    }

# команда для запуска python -m src.pipelines.build_dataset_file

@app.command()
def run(
    brepnet_dir: Path = typer.Option(PROCESSED_DATA_DIR / 'dataset' / 'features' / 'brepnet', help="Путь к папке с признаками B-repNet (*.npz)."),
    output_file: Path = typer.Option(PROCESSED_DATA_DIR / 'dataset' / 'combined_dataset.json', help="Путь к выходному JSON файлу набора данных."),
    validation_split: float = typer.Option(0.15, help="Доля обучающих данных для валидации."),
    test_split: float = typer.Option(0.15, help="Доля всех данных для тестирования."),
    random_seed: int = typer.Option(42, help="Случайное число для воспроизводимости разделения."),
):
    """Создает файл набора данных для комбинированной модели B-repNet + DINO."""


    stems = {p.stem for p in brepnet_dir.glob("*.npz")}
    
    stems = list(stems)


    logger.info(f"Найдено {len(stems)} общих моделей.")

    train_val_ids, test_ids = train_test_split(
        stems, test_size=test_split, random_state=random_seed
    )
    train_ids, val_ids = train_test_split(
        train_val_ids, test_size=validation_split, random_state=random_seed
    )

    logger.info(f"Разделение: {len(train_ids)} train, {len(val_ids)} validation, {len(test_ids)} test.")

    # Вычислить стандартизацию для B-repNet по обучающей выборке
    standardization_data = find_brepnet_standardization(train_ids, brepnet_dir)

    
    dataset_json = {
        "brepnet_features_dir": str(brepnet_dir.resolve()),
        "training_set": train_ids,
        "validation_set": val_ids,
        "test_set": test_ids,
        "feature_standardization": standardization_data
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    save_json_data(output_file, dataset_json)

    logger.info(f"\nФайл набора данных успешно создан: {output_file}")



if __name__ == "__main__":
    app()