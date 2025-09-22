import torch
import pytorch_lightning as pl
import copy
import torch.nn.functional as F

from .encoder import CustomBRepEncoder
from .decoder import ConditionalDecoder

from typing import Dict, Iterable

def rmse_from_mse(mse: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(mse.clamp_min(1e-12))

def finite_ratio(t: torch.Tensor) -> torch.Tensor:
    return torch.isfinite(t).float().mean()

def contrast_summary_from_logits(logits: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    logits: [B, 1+K] — 0-й столбец это позитив, остальные — негативы.
    Возвращает con_acc, pos, neg_mean, neg_std, margin.
    """
    pos = logits[:, 0]
    neg = logits[:, 1:]
    con_acc = (logits.argmax(dim=1) == 0).float().mean()
    neg_mean = neg.mean(dim=1)
    neg_std  = neg.std(dim=1, unbiased=False)
    margin   = pos - neg_mean
    return {
        "con_acc":  con_acc,
        "pos":      pos.mean(),
        "neg_mean": neg_mean.mean(),
        "neg_std":  neg_std.mean(),
        "margin":   margin.mean(),
    }

def recall_at_k_from_logits(logits: torch.Tensor, ks: Iterable[int] = (1, 5, 10)) -> Dict[int, torch.Tensor]:
    """
    Recall@K: входит ли позитив (столбец 0) в топ-K по убыванию логитов.
    logits: [B, 1+K]
    """
    max_k = max(ks)
    values, indices = torch.topk(logits, k=max_k, dim=1, largest=True)
    # позиции позитивов вдоль оси признаков (dim=1)
    pos_ranks = (indices == 0).nonzero(as_tuple=False)[:, 1]  # [B]
    out = {}
    for k in ks:
        out[k] = (pos_ranks < k).float().mean()
    return out


class BRepAutoEncoderModule(pl.LightningModule):
    def __init__(self,
                 n_layers: int = 2,
                 use_attention: bool = True,
                 lr=3e-4,
                 tau: float = 0.07, 
                 m: float = 0.99, 
                 queue_size: int = 4096,
                 w_rec: float = 1.0, 
                 w_con: float = 1.0, 
                 points_per_face: int = 500
                 ):
        super().__init__()
        self.encoder = CustomBRepEncoder(
                v_in_width=3,
                e_in_width=10,
                f_in_width=7,
                out_width=64,
                num_layers=n_layers,
                use_attention=use_attention
            )
        self.decoder = ConditionalDecoder(
            latent_size=64, 
            hidden_dims=[1024, 1024, 1024, 1024]
            )
        self.lr = lr
        self.tau = tau          # температура InfoNCE
        self.m = m              # коэффициент EMA для momentum-энкодера
        self.w_rec = w_rec      # вес реконструктивного лосса
        self.w_con = w_con      # вес контрастивного лосса
        self.points_per_face = points_per_face

        # momentum-энкодер (без градиентов)
        self.encoder_m = copy.deepcopy(self.encoder)
        for p in self.encoder_m.parameters():
            p.requires_grad_(False)

        # очередь негативов для глобальных эмбеддингов (d x K)
        d = int(getattr(self.encoder, "out_width", 64))
        self.queue_size = queue_size
        self.register_buffer("z_queue", F.normalize(torch.randn(d, queue_size), dim=0))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))


        self.save_hyperparameters(ignore=['encoder', 'decoder'])

    @torch.no_grad()
    def _momentum_update(self):
        # EMA-обновление весов momentum-энкодера
        for p_q, p_k in zip(self.encoder.parameters(), self.encoder_m.parameters()):
            p_k.data.mul_(self.m).add_(p_q.data, alpha=1.0 - self.m)

    @torch.no_grad()
    def _enqueue(self, k: torch.Tensor):
        # k: [D] или [1, D] — глобальный эмбеддинг ключа
        k = k.flatten()
        ptr = int(self.queue_ptr)
        self.z_queue[:, ptr] = k
        self.queue_ptr[0] = (ptr + 1) % self.queue_size

    @staticmethod
    def _safe_mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        mask = torch.isfinite(a) & torch.isfinite(b)
        denom = mask.float().sum().clamp(min=1.0)
        diff = (torch.where(mask, a, torch.zeros_like(a)) -
                torch.where(mask, b, torch.zeros_like(b)))
        return (diff.pow(2) * mask).sum() / denom

    def compute_loss(self, predicted, target_xyz, target_sdf):
        pred_xyz, pred_sdf = predicted[:, :3], predicted[:, 3]
        xyz_loss = torch.nn.functional.mse_loss(pred_xyz, target_xyz)
        sdf_loss = torch.nn.functional.mse_loss(pred_sdf, target_sdf)
        return xyz_loss + sdf_loss
    
    def _common_step(self, batch):
        device = self.device
        if isinstance(batch, (list, tuple)):
            batch = batch[0]

        # ===== 2.1. Эмбеддинги лиц энкодером (query) =====
        z_faces_q = self.encoder(batch)                 # [F, D]
        z_faces_q = torch.nan_to_num(z_faces_q)
        if z_faces_q.size(0) == 0:
            # нет граней → пропускаем пример (нулевые лоссы)
            zero = torch.zeros((), device=self.device)
            return zero, zero, zero

        z_faces_q = F.normalize(z_faces_q, dim=-1, eps=1e-6)
        z_q = F.normalize(z_faces_q.mean(dim=0), dim=0, eps=1e-6)  # [D]

        # ===== 2.2. Слабая аугментация и momentum-энкодер (key) =====
        with torch.no_grad():
            self._momentum_update()
            batch_aug = {}
            for k, v in batch.items():
                if torch.is_tensor(v) and v.dtype.is_floating_point:
                    batch_aug[k] = (v + 0.01 * torch.randn_like(v)).to(device)
                else:
                    batch_aug[k] = v

            z_faces_k = self.encoder_m(batch_aug)      # [F, D]
            z_faces_k = torch.nan_to_num(z_faces_k)
            if z_faces_k.size(0) == 0:
                # ключ пуст тоже пропускаем пример
                zero = torch.zeros((), device=self.device)
                return zero, zero, zero

            z_faces_k = F.normalize(z_faces_k, dim=-1, eps=1e-6)
            z_k = F.normalize(z_faces_k.mean(dim=0), dim=0, eps=1e-6)      # [D]

        # контраст: перед умножением — "чистим" очередь и сводим тип/девайс
        queue = torch.nan_to_num(self.z_queue.detach()).to(z_q.device, dtype=z_q.dtype)
        assert z_q.shape[0] == queue.shape[0], f"MoCo dim mismatch: z_q={z_q.shape}, queue={queue.shape}"

        l_pos = torch.matmul(z_q.unsqueeze(0), z_k.unsqueeze(1))   # [1,1]
        l_neg = torch.matmul(z_q.unsqueeze(0), queue)              # [1,K]

        logits = torch.cat([l_pos, l_neg], dim=1) / self.tau
        # если вдруг прилетели NaN — «обнулим» вклад контраста в этот шаг
        if not torch.isfinite(logits).all():
            loss_con = torch.zeros((), device=logits.device)
        else:
            labels = torch.zeros(1, dtype=torch.long, device=logits.device)
            loss_con = F.cross_entropy(logits, labels)

        self._enqueue(z_k.detach())

        # ===== 2.4. Реконструктив — как у тебя, по всем лицам =====
        sampled_points = batch['sdf_uv']      # [F, S, 2]
        sampled_sdf   = batch['sdf_vals'] 
        # уменьшаем количество точек в 2 раза, чтобы не закипел VRAM
        # sampled_points = sampled_points[:, ::2, :].contiguous()
        # sampled_sdf    = sampled_sdf[:, ::2].contiguous()    # [F, S]
        
        num_faces = sampled_points.shape[0]
        if num_faces == 0:
            # пустая геометрия — только контраст
            return self.w_con * loss_con, torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

        total_loss_rec = torch.zeros((), device=device)
        total_xyz_loss = torch.zeros((), device=device)
        total_sdf_loss = torch.zeros((), device=device)

        for i in range(num_faces):
            uv  = torch.nan_to_num(sampled_points[i]).to(device).float()
            sdf = torch.nan_to_num(sampled_sdf[i]).to(device).float()
            emb_face = torch.nan_to_num(z_faces_q[i]).float()  # !!! используем query-эмбеддинг лиц

            pred = torch.nan_to_num(self.decoder(uv, emb_face))      # [S, 4]
            target_xyz = self.compute_xyz_from_uv(uv).float()

            xyz_loss = self._safe_mse(pred[:, :3], target_xyz)
            sdf_loss = self._safe_mse(pred[:, 3],  sdf)

            total_xyz_loss += xyz_loss
            total_sdf_loss += sdf_loss
            total_loss_rec += xyz_loss + sdf_loss
            # по факту первые 6 граней - это уже само тело остальные - это вырезы
            if i == 6:
                break  # ограничим первыми гранями для экономии VRAM

        avg_xyz_loss = total_xyz_loss / num_faces
        avg_sdf_loss = total_sdf_loss / num_faces
        loss_rec = total_loss_rec / num_faces

        # ===== 2.5. Совместный лосс =====
        loss_total = self.w_rec * loss_rec + self.w_con * loss_con
        return loss_total, avg_xyz_loss, avg_sdf_loss, logits.detach(), finite_ratio(z_faces_q).detach()

    def training_step(self, batch, batch_idx):
        loss, xyz_mse, sdf_mse, logits, fin_z = self._common_step(batch)
        # базовые
        self.log('train_loss', loss, batch_size=1)
        self.log('train_xyz_loss', xyz_mse, batch_size=1)
        self.log('train_sdf_loss', sdf_mse, batch_size=1)
        self.log('train_rmse_xyz', rmse_from_mse(xyz_mse), batch_size=1)
        self.log('train_rmse_sdf', rmse_from_mse(sdf_mse), batch_size=1)
        self.log('train_finite_z', fin_z, batch_size=1)

        # контраст
        c = contrast_summary_from_logits(logits)
        self.log('train_con_acc',  c['con_acc'],  batch_size=1, prog_bar=True)
        self.log('train_con_pos',  c['pos'],      batch_size=1)
        self.log('train_con_negm', c['neg_mean'], batch_size=1)
        self.log('train_con_negs', c['neg_std'],  batch_size=1)
        self.log('train_con_margin', c['margin'], batch_size=1)

        # Recall@K
        r = recall_at_k_from_logits(logits, ks=(1, 5, 10))
        self.log('train_recall@1',  r[1],  batch_size=1, prog_bar=True)
        self.log('train_recall@5',  r[5],  batch_size=1)
        self.log('train_recall@10', r[10], batch_size=1)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, xyz_mse, sdf_mse, logits, fin_z = self._common_step(batch)
        self.log('val_loss', loss, prog_bar=True, batch_size=1)
        self.log('val_xyz_loss', xyz_mse, batch_size=1)
        self.log('val_sdf_loss', sdf_mse, batch_size=1)
        self.log('val_rmse_xyz', rmse_from_mse(xyz_mse), batch_size=1)
        self.log('val_rmse_sdf', rmse_from_mse(sdf_mse), batch_size=1)
        self.log('val_finite_z', fin_z, batch_size=1)

        c = contrast_summary_from_logits(logits)
        self.log('val_con_acc',  c['con_acc'],  batch_size=1, prog_bar=True)
        self.log('val_con_pos',  c['pos'],      batch_size=1)
        self.log('val_con_negm', c['neg_mean'], batch_size=1)
        self.log('val_con_negs', c['neg_std'],  batch_size=1)
        self.log('val_con_margin', c['margin'], batch_size=1)

        r = recall_at_k_from_logits(logits, ks=(1, 5, 10))
        self.log('val_recall@1',  r[1],  batch_size=1, prog_bar=True)
        self.log('val_recall@5',  r[5],  batch_size=1)
        self.log('val_recall@10', r[10], batch_size=1)

        return loss

    def compute_xyz_from_uv(self, uv_coords):
        """ Простейшая проекция UV в 3D пространство (z=0).
            спиздили у китайцев
        """
        x = uv_coords[:, 0]  # x координата
        y = uv_coords[:, 1]  # y координата
        z = torch.zeros_like(x)  # z координата

        return torch.stack([x, y, z], dim=-1)

    def configure_optimizers(self):
        return torch.optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=self.lr)
    