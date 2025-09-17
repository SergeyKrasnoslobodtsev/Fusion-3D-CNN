import torch
import torchvision.transforms as T
from PIL import Image
from typing import Any

class DINOFeatureExtractor:
    def __init__(self) -> None:
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.model: Any = torch.hub.load(
            'facebookresearch/dinov2', 'dinov2_vits14', pretrained=True
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        self.transform: T.Compose = make_transform()

    @torch.no_grad()
    def __call__(self, image_path: str) -> torch.Tensor:
        img: Image.Image = Image.open(image_path).convert("RGB")
        img_tensor: torch.Tensor = self.transform(img)  # type: ignore
        img_tensor = img_tensor.unsqueeze(0).to(self.device)  
        feats: torch.Tensor = self.model(img_tensor)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats

def make_transform(resize_size: int = 224) -> T.Compose:
    to_tensor: T.ToTensor = T.ToTensor()
    resize: T.Resize = T.Resize((resize_size, resize_size))
    normalize: T.Normalize = T.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return T.Compose([to_tensor, resize, normalize])
