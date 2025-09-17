from pathlib import Path
import typer
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from ..ssl_graph_brep.models.ssl_brep import SSLBRepModule
from ..ssl_graph_brep.data_module.brep_data_loader import BRepDataModule
from ...config import REPORTS_DIR, MODELS_DIR
app = typer.Typer()



@app.command()
def run(
    npz_brep_dir: Path = typer.Option(Path("./features"), help="Путь к директории с 3D моделями"),
    batch_size: int = typer.Option(16, help="Размер батча"),
    epochs: int = typer.Option(50, help="Количество эпох"),
    num_workers: int = typer.Option(4, help="Количество воркеров для DataLoader"),
    proj_dim: int = typer.Option(128, help="Размер проекции"),
    lr: float = typer.Option(1e-3, help="Learning rate"),
    hidden: int = typer.Option(128, help="Размер скрытого слоя"),
    tau: float = typer.Option(0.1, help="Температура для контрастной потери"),
    lambda_topo_next: float = typer.Option(1.5, help="Вес топологической потери для next"),
    lambda_topo_mate: float = typer.Option(1.5, help="Вес топологической потери для mate"),
    aug_p: float = typer.Option(0.15, help="Вероятность аугментации"),
    topo_tau: float = typer.Option(0.07, help="Температура для топологической потери"),
    weight_decay: float = typer.Option(1e-4, help="Вес L2-регуляризации"),
    validation_ratio: float = typer.Option(0.1, help="Доля валидационного сета"),
    test_ratio: float = typer.Option(0.1, help="Доля тестового сета"),
    ):

    csv_logger = CSVLogger(save_dir=REPORTS_DIR, name="ssl_brep")

    dm = BRepDataModule(
        data_dir=str(npz_brep_dir),
        batch_size=batch_size, 
        num_workers=num_workers,
        val_ratio=validation_ratio,
        test_ratio=test_ratio)
    dm.setup()


    model = SSLBRepModule(
        proj_dim=proj_dim,
        lr=lr,
        hidden=hidden,
        tau=tau,
        lambda_topo_next=lambda_topo_next,
        lambda_topo_mate=lambda_topo_mate,
        aug_p=aug_p,
        topo_tau=topo_tau,
        weight_decay=weight_decay,
    )

    ckpt = ModelCheckpoint(
        monitor="val_infoNCE_acc", 
        mode="max", 
        save_top_k=1, 
        save_last=True,
        dirpath=MODELS_DIR / "ssl_brep",
        filename="ssl-brep-{epoch:02d}-{val_infoNCE_acc:.3f}"
    )
    trainer = pl.Trainer(max_epochs=epochs, callbacks=[ckpt], logger=[csv_logger], log_every_n_steps=10)

    trainer.fit(model, dm)



if __name__ == "__main__":
    app()