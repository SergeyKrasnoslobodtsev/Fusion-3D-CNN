import torch
from ..models.clip import clip

from typing import cast
from PIL import Image

class CLIPFeatureExtractor:
    def __init__(self, model_name: str = "ViT-B/32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(model_name, self.device)
        self.model.eval()


    @torch.no_grad()
    def __call__(self, image_path:str):
        img = Image.open(image_path).convert("RGB")
        img_tensor = self.preprocess(img)  # (1,C,H,W)
        img_tensor = cast(torch.Tensor, img_tensor).unsqueeze(0).to(self.device)  # (1,C,H,W)
        feats = self.model.encode_image(img_tensor)        
        feats = feats / feats.norm(dim=-1, keepdim=True)

        return feats