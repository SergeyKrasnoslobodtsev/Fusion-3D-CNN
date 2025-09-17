from pathlib import Path
from tqdm import tqdm

import torch
import numpy as np
import typer

from ..config import PROCESSED_DATA_DIR, RAW_DATA_DIR

from ..utils import file_utils
from ..features.dino_extractor import DINOFeatureExtractor

from loguru import logger

app = typer.Typer()


def extractor(model: DINOFeatureExtractor, in_folder: Path, out_path: Path):
    img_files = file_utils.get_files(in_folder, ("png", "jpg", "jpeg", "bmp"))
    views = []
    for img_file in img_files:
        feats = model(str(img_file))          # (1, D)
        views.append(feats.squeeze(0))        # (D,)

    if not views:
        raise RuntimeError(f"No images found in {in_folder} {img_files}")

    views = torch.stack(views, dim=0)                     # (V, D)


    out_path.parent.mkdir(parents=True, exist_ok=True)
    np_names = [p.name for p in img_files]
    np.savez_compressed(
        out_path,
        views=views.detach().cpu().numpy().astype("float32"), 
        names=np.array(np_names, dtype=object),
    )
    
def worker(args):
    in_folder, out_path = args
    # Инициализируем модель и pooling здесь, внутри дочернего процесса
    model = DINOFeatureExtractor()
    extractor(model=model, in_folder=in_folder, out_path=out_path)

# Команда для запуска python -m src.pipelines.extr_feats_dino
@app.command()
def run(
    images_dir: Path = typer.Option(PROCESSED_DATA_DIR / "dataset_129" / '2D', help="Путь к директориям с 2D изображениями"),
    num_workers: int = typer.Option(1, help="Количество рабочих процессов для извлечения признаков"),
    force_regeneration: bool = typer.Option(False, help="Перегенерировать признаки, даже если они уже существуют"),
):
    """ Извлечение признаков из 2D изображений с использованием DINO """
    output_dir = images_dir.parent / "features" / "dino"
    output_dir.mkdir(parents=True, exist_ok=True)

    model_folders = [f for f in images_dir.iterdir() if f.is_dir()]

    if not force_regeneration:
        model_folders = file_utils.filter_unconverted_files(model_folders, output_dir)

    use_many_threads = num_workers > 1

    if use_many_threads:
        from concurrent.futures import ProcessPoolExecutor

        # Передаем только простые аргументы
        worker_args = [(mf, output_dir / f"{mf.name}.npz") for mf in model_folders]


        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Используем новую функцию-обертку
            list(tqdm(executor.map(worker, worker_args), total=len(worker_args)))
    else:
        # Для однопоточного режима инициализируем модель один раз
        model = DINOFeatureExtractor()
        for mf in tqdm(model_folders, desc="Extracting DINO features"):
            model_id = mf.name
            out_path = output_dir / f"{model_id}.npz"
            # пропускаем если файл уже существует
            if out_path.exists():
                continue
            extractor(model=model, in_folder=mf, out_path=out_path)

if __name__ == "__main__":
    app()