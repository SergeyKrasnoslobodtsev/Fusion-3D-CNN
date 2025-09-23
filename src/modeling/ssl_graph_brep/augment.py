import torch
from torch_geometric.data import HeteroData
from typing import Tuple

def random_rotate_3d(xyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Случайная ротация 3D координат и возврат матрицы ротации.
    xyz: [*, 3, *] - любые размерности с 3D координатами во втором измерении
    """
    device = xyz.device
    # Случайные углы Эйлера
    alpha = torch.rand(1, device=device) * 2 * torch.pi  # вокруг Z
    beta = torch.rand(1, device=device) * 2 * torch.pi   # вокруг Y  
    gamma = torch.rand(1, device=device) * 2 * torch.pi  # вокруг X
    
    # Матрицы ротации
    Rz = torch.tensor([
        [torch.cos(alpha), -torch.sin(alpha), 0],
        [torch.sin(alpha), torch.cos(alpha), 0], 
        [0, 0, 1]
    ], device=device)
    
    Ry = torch.tensor([
        [torch.cos(beta), 0, torch.sin(beta)],
        [0, 1, 0],
        [-torch.sin(beta), 0, torch.cos(beta)]
    ], device=device)
    
    Rx = torch.tensor([
        [1, 0, 0],
        [0, torch.cos(gamma), -torch.sin(gamma)],
        [0, torch.sin(gamma), torch.cos(gamma)]
    ], device=device)
    
    R = Rz @ Ry @ Rx
    
    # Применяем ротацию
    original_shape = xyz.shape
    if len(original_shape) == 3:  # [N, 3, M]
        rotated = torch.einsum('ij,njm->nim', R, xyz)
    elif len(original_shape) == 4:  # [N, 3, M, K] 
        rotated = torch.einsum('ij,njmk->nimk', R, xyz)
    else:
        raise ValueError(f"Unsupported shape: {original_shape}")
        
    return rotated, R

def augment_coedge_grid(grid: torch.Tensor, p: float = 0.2) -> torch.Tensor:
    """
    Правильная аугментация coedge grid [C, 12, 10].
    Каналы: [xyz, tangents, nL, nR] = [0:3, 3:6, 6:9, 9:12]
    """
    if torch.rand(1) > p:
        return grid
        
    augmented = grid.clone()
    
    # Ротация координат и всех направлений
    xyz, R = random_rotate_3d(grid[:, 0:3, :])  # [C, 3, 10]
    tangents, _ = random_rotate_3d(grid[:, 3:6, :])  # [C, 3, 10] 
    nL, _ = random_rotate_3d(grid[:, 6:9, :])  # [C, 3, 10]
    nR, _ = random_rotate_3d(grid[:, 9:12, :])  # [C, 3, 10]
    
    # Применяем ту же ротацию ко всем направлениям
    tangents = torch.einsum('ij,cjk->cik', R, grid[:, 3:6, :])
    nL = torch.einsum('ij,cjk->cik', R, grid[:, 6:9, :]) 
    nR = torch.einsum('ij,cjk->cik', R, grid[:, 9:12, :])
    
    augmented[:, 0:3, :] = xyz
    augmented[:, 3:6, :] = tangents
    augmented[:, 6:9, :] = nL
    augmented[:, 9:12, :] = nR
    
    return augmented

def augment_face_uv(uv: torch.Tensor, p: float = 0.2) -> torch.Tensor:
    """
    Правильная аугментация face UV [F, 7, 10, 10].
    Каналы: [xyz, normals, mask] = [0:3, 3:6, 6:7]
    """
    if torch.rand(1) > p:
        return uv
        
    augmented = uv.clone()
    
    # Ротация координат и нормалей
    xyz, R = random_rotate_3d(uv[:, 0:3, :, :])  # [F, 3, 10, 10]
    normals = torch.einsum('ij,fjkl->fikl', R, uv[:, 3:6, :, :])
    
    augmented[:, 0:3, :, :] = xyz
    augmented[:, 3:6, :, :] = normals
    # Маску не трогаем: augmented[:, 6:7, :, :] остается как есть
    
    return augmented

def geometric_augment(data: HeteroData, p: float = 0.15) -> HeteroData:
    """
    Правильная геометрическая аугментация B-Rep данных.
    Применяется ДО нормализации CoedgeLCSNormalize.
    """
    augmented = data.clone()
    
    # Аугментация coedge grid
    if "grid" in augmented["coedge"]:
        augmented["coedge"].grid = augment_coedge_grid(
            augmented["coedge"].grid, p=p
        )
    
    # Аугментация face UV
    if "uv" in augmented["face"]:
        augmented["face"].uv = augment_face_uv(
            augmented["face"].uv, p=p
        )
    
    # Лёгкий шум на признаки (безопасно)
    if torch.rand(1) < p * 0.5:  # реже
        noise_scale = 0.01
        augmented["face"].x += noise_scale * torch.randn_like(augmented["face"].x)
        augmented["edge"].x += noise_scale * torch.randn_like(augmented["edge"].x)
        
    return augmented

def two_views_corrected(data: HeteroData, p: float = 0.15) -> Tuple[HeteroData, HeteroData]:
    """
    Создает два правильно аугментированных представления.
    """
    view1 = geometric_augment(data, p=p)
    view2 = geometric_augment(data, p=p) 
    return view1, view2