import math
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    import pytorch_lightning as pl
    from pytorch_lightning.utilities.types import STEP_OUTPUT
except ImportError as e:
    raise ImportError("PyTorch Lightning is required. Please install 'pytorch-lightning'.")

from .dataset_loader import ReconstructionDataset, BatchOnlyCollator
from .model.encoder import CustomBRepEncoder
from .model.decoder import ConditionalDecoder


def l2_xyz_loss(pred: torch.Tensor, target_xyz: torch.Tensor) -> torch.Tensor:
    """
    Compute L2 loss on predicted xyz vs ground-truth.

    pred: [B, P, 4] where last channel is [x,y,z,sdf]
    target_xyz: [B, P, 3]
    """
    pred_xyz = pred[..., :3]
    return F.mse_loss(pred_xyz, target_xyz)


def sdf_l1_loss(pred: torch.Tensor, target_sdf: torch.Tensor) -> torch.Tensor:
    """Optional auxiliary SDF loss if available.

    pred: [B, P, 4]
    target_sdf: [B, P, 1]
    """
    pred_sdf = pred[..., 3:4]
    return F.l1_loss(pred_sdf, target_sdf)

class SSLDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: Path, stats_path: Path, batch_size: int, points_per_sample: int):
        super().__init__()
        self.data_dir = data_dir
        self.stats_path = stats_path
        self.batch_size = batch_size
        self.points_per_sample = points_per_sample
        self.dataset: Optional[ReconstructionDataset] = None
        self.collate_fn = BatchOnlyCollator()

    def setup(self, stage: Optional[str] = None) -> None:
        self.dataset = ReconstructionDataset(self.data_dir, self.stats_path, self.points_per_sample)

    def train_dataloader(self):
        assert self.dataset is not None
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=self.collate_fn,
        )


class SSLModule(pl.LightningModule):
    def __init__(
        self,
        v_in: int,
        e_in: int,
        f_in: int,
        out_width: int,
        num_layers: int,
        use_attention: bool,
        decoder_hidden: Optional[str],
        lr: float,
        use_sdf_loss: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        hidden_dims = [int(x) for x in decoder_hidden.split(",")] if decoder_hidden else [1024, 1024, 1024, 1024]
        self.encoder = CustomBRepEncoder(v_in, e_in, f_in, out_width, num_layers, use_attention)
        self.decoder = ConditionalDecoder(latent_size=out_width, hidden_dims=hidden_dims, uv_input_dim=2, output_dim=4)
        self.lr = lr
        self.use_sdf_loss = use_sdf_loss

    def configure_optimizers(self):
        return torch.optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=self.lr)

    def forward(self, uv_coords: torch.Tensor, face_embedding: torch.Tensor) -> torch.Tensor:
        return self.decoder(uv_coords, face_embedding)

    def training_step(self, batch, batch_idx: int) -> STEP_OUTPUT:
        data, face_ids, uv_coords, xyz_coords = batch  # data: BRepData

        # Move batched BRepData tensors to device
        for k, v in data.__dict__.items():
            if isinstance(v, torch.Tensor):
                data.__dict__[k] = v.to(self.device)
        face_ids = face_ids.to(self.device)
        uv_coords = uv_coords.to(self.device)
        xyz_coords = xyz_coords.to(self.device)

        # Encode faces embeddings
        all_face_embeddings = self.encoder(data)  # [num_faces_total, out_width]

        # Gather face embeddings per sample in batch using stored indices
        embeddings = []
        for i in range(uv_coords.shape[0]):
            mask = (data.face_batch_idx == i)
            face_emb_i = all_face_embeddings[mask][face_ids[i]]  # [out_width]
            embeddings.append(face_emb_i)
        face_embeddings = torch.stack(embeddings, dim=0)  # [B, out_width]

        # Decode per sample
        preds = []
        for i in range(uv_coords.shape[0]):
            preds.append(self(uv_coords[i], face_embeddings[i]))  # [P, 4]
        pred = torch.stack(preds, dim=0)  # [B, P, 4]

        loss = l2_xyz_loss(pred, xyz_coords)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=uv_coords.shape[0])
        return loss


def train_ssl(
    data_dir: Path,
    stats_path: Path,
    save_dir: Path,
    epochs: int = 20,
    batch_size: int = 8,
    points_per_sample: int = 1024,
    lr: float = 1e-3,
    device: Optional[str] = None,
    v_in: int = 0,
    e_in: int = 0,
    f_in: int = 0,
    out_width: int = 64,
    num_layers: int = 2,
    use_attention: bool = False,
    decoder_hidden: Optional[str] = "1024,1024,1024,1024",
    use_sdf_loss: bool = False,
) -> None:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Datamodule
    dm = SSLDataModule(data_dir=data_dir, stats_path=stats_path, batch_size=batch_size, points_per_sample=points_per_sample)
    dm.setup()
    assert dm.dataset is not None
    if len(dm.dataset) == 0:
        raise RuntimeError(f"No .npz files with 'face_samples' found in {data_dir}")

    # Infer feature dims if needed
    try:
        if v_in <= 0 or e_in <= 0 or f_in <= 0:
            first_file = dm.dataset.data_files[0]
            import numpy as _np
            with _np.load(first_file, allow_pickle=True) as _data:
                if v_in <= 0:
                    v_in = int(_data['vertices'].shape[1])
                if e_in <= 0:
                    e_in = int(_data['edges'].shape[1])
                if f_in <= 0:
                    f_in = int(_data['faces'].shape[1])
    except Exception:
        pass

    model = SSLModule(
        v_in=v_in,
        e_in=e_in,
        f_in=f_in,
        out_width=out_width,
        num_layers=num_layers,
        use_attention=use_attention,
        decoder_hidden=decoder_hidden,
        lr=lr,
        use_sdf_loss=use_sdf_loss,
    )

    from pytorch_lightning.callbacks import ModelCheckpoint
    checkpoint_cb = ModelCheckpoint(
        monitor="train_loss",
        mode="min",
        save_top_k=1,
        filename="ssl-{epoch:02d}-{train_loss:.4f}",
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        default_root_dir=str(save_dir),
        accelerator="gpu" if device == "cuda" and torch.cuda.is_available() else "cpu",
        devices=1,
        log_every_n_steps=10,
        callbacks=[checkpoint_cb],
    )

    trainer.fit(model, dm)

    # Save checkpoints (Lightning also manages checkpoints if callbacks used)
    torch.save({
        "encoder_state": model.encoder.state_dict(),
        "decoder_state": model.decoder.state_dict(),
    }, save_dir / "ssl_last.pt")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Self-supervised training for BRepNet features (encoder-decoder L2 reconstruction)")
    parser.add_argument("--data_dir", type=Path, required=True, help="Directory with .npz samples that include vertices/edges/faces/connectivity and face_samples")
    parser.add_argument("--stats_path", type=Path, required=True, help="Path to BrepNet standardization stats JSON")
    parser.add_argument("--save_dir", type=Path, default=Path("runs/ssl"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--points_per_sample", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--v_in", type=int, default=3, help="Vertex feature width; 0=auto")
    parser.add_argument("--e_in", type=int, default=10, help="Edge feature width; 0=auto")
    parser.add_argument("--f_in", type=int, default=7, help="Face feature width; 0=auto")
    parser.add_argument("--out_width", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--use_attention", action="store_true")
    parser.add_argument("--decoder_hidden", type=str, default="1024,1024,1024,1024")
    parser.add_argument("--use_sdf_loss", action="store_true")

    args = parser.parse_args()

    train_ssl(
        data_dir=args.data_dir,
        stats_path=args.stats_path,
        save_dir=args.save_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        points_per_sample=args.points_per_sample,
        lr=args.lr,
        device=args.device,
        v_in=args.v_in,
        e_in=args.e_in,
        f_in=args.f_in,
        out_width=args.out_width,
        num_layers=args.num_layers,
        use_attention=args.use_attention,
        decoder_hidden=args.decoder_hidden,
        use_sdf_loss=args.use_sdf_loss,
    )
