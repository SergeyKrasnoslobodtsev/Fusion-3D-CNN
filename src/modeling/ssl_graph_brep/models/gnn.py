import torch
from torch import nn
from torch_geometric.nn import HeteroConv, SAGEConv

class BRepHeteroGNN(nn.Module):
    """
    Гетерограф: узлы 'face'/'edge'/'coedge'; ребра:
      ('coedge','next','coedge'), ('coedge','mate','coedge'),
      ('coedge','to_face','face'), ('coedge','to_edge','edge').
    """
    def __init__(self, hidden: int = 128, out_dim: int = 256) -> None:
        super().__init__()
        self.conv1 = HeteroConv({
            ('coedge', 'next', 'coedge'): SAGEConv((-1, -1), hidden),
            ('coedge', 'mate', 'coedge'): SAGEConv((-1, -1), hidden),
            ('coedge', 'to_face', 'face'): SAGEConv((-1, -1), hidden),
            ('coedge', 'to_edge', 'edge'): SAGEConv((-1, -1), hidden),
        }, aggr='sum')
        self.conv2 = HeteroConv({
            ('coedge', 'next', 'coedge'): SAGEConv((-1, -1), hidden),
            ('coedge', 'mate', 'coedge'): SAGEConv((-1, -1), hidden),
            ('coedge', 'to_face', 'face'): SAGEConv((-1, -1), hidden),
            ('coedge', 'to_edge', 'edge'): SAGEConv((-1, -1), hidden),
        }, aggr='sum')
        self.readout = nn.ModuleDict({
            'coedge': nn.Linear(hidden, out_dim),
            'face': nn.Linear(hidden, out_dim),
            'edge': nn.Linear(hidden, out_dim),
        })

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {k: torch.relu(v) for k, v in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {k: torch.relu(v) for k, v in x_dict.items()}
        z_dict = {k: self.readout[k](v) for k, v in x_dict.items()}
        return z_dict  # per-type embeddings