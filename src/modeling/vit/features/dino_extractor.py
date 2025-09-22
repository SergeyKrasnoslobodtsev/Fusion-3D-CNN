import torch
import torchvision.transforms as T

from typing import cast

from PIL import Image

class DINOFeatureExtractor:
    def __init__(self, model_name: str = "dinov2_vits14", image_size: int = 224):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = torch.hub.load('facebookresearch/dinov2', model_name, pretrained=True).to(self.device) # type: ignore
        self.model.eval()
        self.transform = _transform(image_size)

    @torch.no_grad()
    def __call__(self, image_path:str):
        img_tensor = self.transform(Image.open(image_path))  # (1,C,H,W)
        img_tensor = cast(torch.Tensor, img_tensor).unsqueeze(0).to(self.device)  # (1,C,H,W)
        feats = self.model(img_tensor)         # ожидаем (1,D)
        feats = feats / feats.norm(dim=-1, keepdim=True)

        return feats

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def _transform(resize_size: int = 224):
   transform = T.Compose([
       T.Resize((resize_size, resize_size), interpolation=T.InterpolationMode.BICUBIC),
       T.CenterCrop((resize_size, resize_size)),
         _convert_image_to_rgb,
       T.ToTensor(),
       T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
   ])
   return transform