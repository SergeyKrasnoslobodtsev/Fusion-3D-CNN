from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import gc
from tqdm import tqdm
import typer

from .features import BRepNetExtractor
from ...utils import file_utils
from loguru import logger

app = typer.Typer()


def extract_brepnet_features(file, output_path, feature_schema):
    extractor = BRepNetExtractor(file, output_path, feature_schema)
    extractor.process()

def run_worker(worker_args):
    """
    Обрабатывает один файл для извлечения признаков BRepNet.

    Args:
        worker_args (tuple): Кортеж с параметрами:
            - file (Path): Путь к файлу.
            - output_path (Path): Директория для сохранения результата.
            - feature_schema (dict): Схема признаков.
    """
    file = worker_args[0]
    output_path = worker_args[1]
    feature_schema = worker_args[2]
    extract_brepnet_features(file, output_path, feature_schema)




def extract_brepnet_data_from_step(
        step_path, 
        output_path, 
        feature_list_path,
        force_regeneration=False,
        num_workers=1
    ):
    logger.info(f"Начало обработки: force_regeneration={force_regeneration}, num_workers={num_workers}")
    feature_schema = file_utils.load_json(feature_list_path)
    files = file_utils.get_files(step_path)

    if not force_regeneration:
        files = file_utils.filter_unconverted_files(files, output_path)

    use_many_threads = num_workers > 1
    if use_many_threads:
        worker_args = [(f, output_path, feature_schema) for f in files]
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            list(tqdm(executor.map(run_worker, worker_args), total=len(worker_args)))
    else:
        for file in tqdm(files):
            extract_brepnet_features(file, output_path, feature_schema)

    gc.collect()
    logger.success("Обработка завершена.")



from ...config import PROCESSED_DATA_DIR

# command python -m src.pipelines.extr_feats_brepnet

@app.command()
def run(
    step_path_dir: Path = typer.Option(PROCESSED_DATA_DIR / "test" / 'Затвор', help="Путь к директории с 3D моделями"),
    feature_list_path: Path = typer.Option(PROCESSED_DATA_DIR / "test" / "feature_lists/all.json", help="Путь к файлу со списком признаков"),
    num_workers: int = typer.Option(1, help="Количество потоков для обработки"),
    force_regeneration: bool = typer.Option(False, help="Перегенерировать признаки, даже если они уже существуют"),

):
    if not step_path_dir.exists():
        logger.error(f"Директория {step_path_dir} не существует.")
        raise FileNotFoundError(f"Директория {step_path_dir} не найдена.")
    
    if not feature_list_path.exists():
        logger.error(f"Файл {feature_list_path} не существует.")
        raise FileNotFoundError(f"Файл {feature_list_path} не найден.")
    
    output_path = step_path_dir.parent / "features" 

    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)


    extract_brepnet_data_from_step(
        step_path=step_path_dir,
        output_path=output_path,
        feature_list_path=feature_list_path,
        force_regeneration=force_regeneration,
        num_workers=num_workers
    )

if __name__ == "__main__":
    app()