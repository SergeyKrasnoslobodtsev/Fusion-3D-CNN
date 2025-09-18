import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from .models.ensemble import ContrastiveFusionModel
from .data_module.data_loader import FusionDataModule
from ...config import INTERIM_DATA_DIR, MODELS_DIR, REPORTS_DIR

def main():
    # 1. Конфигурация
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    EMBED_DIM = 256
    EPOCHS = 100

    # Пути к данным
    brep_features_dir = INTERIM_DATA_DIR / "features/brepnet"
    dino_features_dir = INTERIM_DATA_DIR / "features/dino"
    stats_path = INTERIM_DATA_DIR / "features/pooled_brep.json"

    # 2. Инициализация DataModule
    data_module = FusionDataModule(
        brep_features_dir=brep_features_dir,
        dino_features_dir=dino_features_dir,
        stats_path=stats_path,
        batch_size=BATCH_SIZE
    )

    # 3. Инициализация Модели
    model = ContrastiveFusionModel(
        embed_dim=EMBED_DIM,
        learning_rate=LEARNING_RATE
    )

    # 4. Настройка логгера и колбэков (сохранение лучшей модели)

    csv_logger = CSVLogger(save_dir=REPORTS_DIR, name="fusion_model")
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss", # Можно заменить на val_loss, когда добавите validation_step
        dirpath=MODELS_DIR / "vit_brep_ensemble",
        filename="fusion-model-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        mode="min",
    )

    # 5. Инициализация Trainer
    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        logger=csv_logger,
        callbacks=[checkpoint_callback],
        
    )

    # 6. Запуск обучения
    trainer.fit(model, datamodule=data_module)

if __name__ == "__main__":
    main()