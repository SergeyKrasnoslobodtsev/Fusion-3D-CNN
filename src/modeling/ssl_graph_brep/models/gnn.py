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
        self.dropout = nn.Dropout(0.1)

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {k: torch.relu(v) for k, v in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {k: torch.relu(v) for k, v in x_dict.items()}
        z_dict = {k: self.readout[k](v) for k, v in x_dict.items()}
        
        return z_dict  # per-type embeddings
    


class AttnReadout(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, 1)
        )
    def forward(self, x, batch):
        # x: [N,C], batch: [N] -> graph ids
        w = self.proj(x)                      # [N,1]
        w = torch.exp(w - w.max())            # стабилизация
        num_g = int(batch.max()) + 1
        out = x.new_zeros((num_g, x.size(1)))
        den = x.new_zeros((num_g, 1))
        out.index_add_(0, batch, w * x)
        den.index_add_(0, batch, w)
        return out / (den + 1e-6)