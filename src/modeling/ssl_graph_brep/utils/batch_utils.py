from __future__ import annotations
from torch_geometric.data import HeteroData

def num_graphs_in_batch(batch: HeteroData) -> int:
    """
    Возвращает число графов в HeteroDataBatch через ptr любого типа узла.
    По умолчанию используем 'coedge', иначе пробуем другие типы.
    """
    for node_type in ["coedge", "face", "edge"]:
        if node_type in batch and hasattr(batch[node_type], "ptr") and batch[node_type].ptr is not None:
            return int(batch[node_type].ptr.numel() - 1)
    return 1