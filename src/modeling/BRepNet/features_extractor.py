from pathlib import Path
from src.modeling.BRepNet.eval import BRepNetEmbeddingExtractor
from pathlib import Path
import numpy as np
from loguru import logger
from tqdm import tqdm
import typer

from ...config import EXTERNAL_DATA_DIR, PROCESSED_DATA_DIR
from ...utils.file_utils import get_files
app = typer.Typer()

# python -m src.modeling.BRepNet.features_extractor

@app.command()
def run(
    checkpoint: Path = typer.Option(EXTERNAL_DATA_DIR / 'pretrained_s2.0.0_extended_step_uv_net_features_0816_183419.ckpt', help="Путь к чекпоинту модели BRepNet"),
    feature_standardization: Path = typer.Option(EXTERNAL_DATA_DIR / 's2.0.0_step_all_features.json', help="Путь к файлу стандартизации признаков"),
    segment_files: Path = typer.Option(EXTERNAL_DATA_DIR / 'segment_names.json', help="Путь к файлу с именами сегментов"),
    kernel_file: Path = typer.Option(EXTERNAL_DATA_DIR / 'kernels/winged_edge.json', help="Путь к файлу с ядром"),
    features_list: Path = typer.Option(EXTERNAL_DATA_DIR / 'feature_lists/all.json', help="Путь к файлу со списком признаков"),
    brepnet_dir: Path = typer.Option(PROCESSED_DATA_DIR / "dataset_129" / 'features' / 'brep', help="Путь к директории с npz файлами BRepNet"),
):

    embedding_extractor = BRepNetEmbeddingExtractor(
        checkpoint_path=checkpoint,
        feature_standardization=feature_standardization,
        segment_files=segment_files,
        kernel_file=kernel_file,
        features_list=features_list,
    )
    brepnet_files = get_files(brepnet_dir, ("npz",))
    
    logger.info(f"Найдено {len(brepnet_files)} файлов BRepNet")
    
    out_face_dir = brepnet_dir / "embeddings_faces"
    out_model_dir = brepnet_dir / "embeddings_models"
    out_face_dir.mkdir(exist_ok=True)
    out_model_dir.mkdir(exist_ok=True)

    feats = np.load(brepnet_files[0])
    for k, v in feats.items():
        print(k, v.shape)

    for npz_path in tqdm(brepnet_files, desc="Извлечение эмбеддингов BRepNet"):
        face_embs, model_emb = embedding_extractor.extract_from_npz(npz_path, pool="mean")
        np.save(out_face_dir / (npz_path.stem + "_faces.npy"), face_embs.detach().cpu().numpy())
        np.save(out_model_dir / (npz_path.stem + "_model.npy"), model_emb.detach().cpu().numpy())
    
    logger.success("Эмбеддинги BRepNet успешно извлечены и сохранены.")

if __name__ == "__main__":
    app()





