import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger, WandbLogger

from .models.self_supervised_ensemble import SelfSupervisedFusionModel
from .data_module.enhanced_data_loader import EnhancedFusionDataModule
from .utils.clustering_callback import PseudoLabelingCallback
from ...config import INTERIM_DATA_DIR, MODELS_DIR, REPORTS_DIR

def main():
    # Конфигурация
    config = {
        'batch_size': 64,  # Увеличиваем для лучшей кластеризации
        'learning_rate': 5e-5,  # Уменьшаем для стабильности
        'embed_dim': 256,
        'temperature': 0.1,
        'num_clusters': 50,
        'epochs': 200,
        'num_workers': 8,
    }

    # Пути к данным
    brep_features_dir = INTERIM_DATA_DIR / "features/brepnet"
    dino_features_dir = INTERIM_DATA_DIR / "features/dino"
    stats_path = INTERIM_DATA_DIR / "features/pooled_brep_enhanced.json"

    # DataModule
    data_module = EnhancedFusionDataModule(
        brep_features_dir=brep_features_dir,
        dino_features_dir=dino_features_dir,
        stats_path=stats_path,
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )

    # Модель
    model = SelfSupervisedFusionModel(
        embed_dim=config['embed_dim'],
        learning_rate=config['learning_rate'],
        temperature=config['temperature'],
        num_clusters=config['num_clusters']
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=MODELS_DIR / "enhanced_fusion",
        filename="fusion-{epoch:02d}-{val_loss:.3f}",
        save_top_k=3,
        mode="min",
        save_last=True
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        min_delta=0.001,
        patience=20,
        mode="min"
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # Callback для псевдо-меток
    pseudo_labeling = PseudoLabelingCallback(
        num_clusters=config['num_clusters'],
        update_freq=5,
        start_epoch=3
    )

    # Логгеры
    csv_logger = CSVLogger(
        save_dir=REPORTS_DIR, 
        name="enhanced_fusion_model"
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=config['epochs'],
        logger=csv_logger,
        callbacks=[
            checkpoint_callback, 
            early_stopping, 
            lr_monitor,
            pseudo_labeling
        ],
        gradient_clip_val=1.0,  # Стабилизация градиентов
        accumulate_grad_batches=2,  # Эффективный batch_size = 64*2 = 128
        precision=16,  # Экономия памяти
        deterministic=True,
        log_every_n_steps=10,
        check_val_every_n_epoch=2
    )

    # Обучение
    trainer.fit(model, datamodule=data_module)

if __name__ == "__main__":
    pl.seed_everything(42, workers=True)
    main()