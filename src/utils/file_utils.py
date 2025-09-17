import json
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