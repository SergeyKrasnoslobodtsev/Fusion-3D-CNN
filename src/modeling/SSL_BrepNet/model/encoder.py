
import torch
from torch.nn import Linear, Sequential, ModuleList, BatchNorm1d, Dropout, LeakyReLU, ReLU
import torch_geometric.nn.conv.gat_conv as gat_conv

class CustomBRepEncoder(torch.nn.Module):
    def __init__(self, v_in_width, e_in_width, f_in_width, out_width, num_layers, use_attention=False):
        super().__init__()
        self.use_attention = use_attention

        # Initial feature embedding layers for vertices, edges, and faces
        self.embed_v_in = LinearBlock(v_in_width, out_width)
        self.embed_e_in = LinearBlock(e_in_width, out_width)
        self.embed_f_in = LinearBlock(f_in_width, out_width)

        # Message passing layers for encoding hierarchical structure
        self.V2E = BipartiteResMRConv(out_width)
        self.E2F = BipartiteResMRConv(out_width)

        # Additional message passing layers to refine features
        self.message_layers = ModuleList([BipartiteResMRConv(out_width) for _ in range(num_layers)])

        # Attention mechanism for handling varied neighborhood sizes
        if self.use_attention:
            self.attention_layers = ModuleList([gat_conv.GATConv(out_width, out_width//4, heads=4) for _ in range(num_layers)])

    def forward(self, data):
        # эмбеддинги
        x_v = self.embed_v_in(data['vertices'])
        x_e = self.embed_e_in(data['edges'])
        x_f = self.embed_f_in(data['faces'])

        # ➜ санитайз сразу после эмбеддингов
        def _sanitize(x: torch.Tensor) -> torch.Tensor:
            # заменяем NaN/Inf и мягко клипуем, чтобы не закипала softmax в GAT
            return torch.clamp(torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4), -1e4, 1e4)

        x_v = _sanitize(x_v)
        x_e = _sanitize(x_e)
        x_f = _sanitize(x_f)

        # индексы
        e_v = data['edge_to_vertex'].long().contiguous()   # [2, Ne]
        f_e = data['face_to_edge'].long().contiguous()     # [2, Nfe]
        f_f = data['face_to_face'].long().contiguous()     # [2, Nff]

        # ➜ self-loops для лиц (защита внимания от пустых соседств)
        nf = x_f.size(0)
        if f_f.numel() == 0:
            f_f = torch.empty(2, 0, dtype=torch.long, device=x_f.device)
        loops = torch.arange(nf, device=x_f.device, dtype=torch.long)
        f_f = torch.cat([f_f, torch.stack([loops, loops])], dim=1)

        # подъём признаков
        x_e = _sanitize(self.V2E(x_v, x_e, e_v[[1, 0]]))
        x_f = _sanitize(self.E2F(x_e, x_f, f_e[[1, 0]]))

        # рефайнмент / внимание
        for i, layer in enumerate(self.message_layers):
            if self.use_attention:
                x_f = _sanitize(self.attention_layers[i](_sanitize(x_f), f_f[:2, :]))
            else:
                x_f = _sanitize(layer(x_f, x_f, f_f[:2, :]))

        return _sanitize(x_f) 

class BipartiteResMRConv(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.mlp = LinearBlock(2 * width, width)

    def forward(self, x_src, x_dst, e):
        diffs = torch.index_select(x_dst, 0, e[1]) - torch.index_select(x_src, 0, e[0])
        maxes = torch.full((x_dst.shape[0], diffs.shape[1]), float('-inf'), device=diffs.device)
        maxes = maxes.scatter_reduce(0, e[1].unsqueeze(-1).expand_as(diffs), diffs, reduce="amax", include_self=True)
        maxes = torch.nan_to_num(maxes, neginf=0.0, posinf=0.0)
        return x_dst + self.mlp(torch.cat([x_dst, maxes], dim=1))

# LinearBlock with flexibility for configurations
class LinearBlock(torch.nn.Module):
    def __init__(self, *layer_sizes, batch_norm=False, dropout=0.0, last_linear=False, leaky=True):
        super().__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            c_in = layer_sizes[i]
            c_out = layer_sizes[i + 1]
            layers.append(Linear(c_in, c_out))
            if last_linear and i + 1 >= len(layer_sizes) - 1:
                break
            if batch_norm:
                layers.append(BatchNorm1d(c_out))
            if dropout > 0:
                layers.append(Dropout(p=dropout))
            layers.append(LeakyReLU() if leaky else ReLU())
        self.f = Sequential(*layers)

    def forward(self, x):
        return self.f(x)