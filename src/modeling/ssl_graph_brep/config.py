from dataclasses import dataclass

@dataclass
class SSLConfig:
    proj_dim: int = 128       # размер проекции для контраста
    lr: float = 1e-3          # скорость обучения
    hidden: int = 128         # размер скрытых слоёв GNN
    tau: float = 0.08          # температура InfoNCE
    lambda_topo_next: float = 12.0  #  вес топологических предтекстов
    lambda_topo_mate: float = 12.0  # вес топологических предтекстов
    aug_p: float = 0.15       # вероятность маскирования атрибутов
    batch_size: int = 32      # размер батча
    num_workers: int = 0      # число потоков загрузки данных