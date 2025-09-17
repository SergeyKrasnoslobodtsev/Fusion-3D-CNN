from typing import List
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def load_and_clean_csv(path: str, train_keys: List[str], val_keys: List[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Разделяем строки train/val по наличию ключевых столбцов
    df_train = df.dropna(subset=[k for k in train_keys if k in df.columns]).copy()
    df_val   = df.dropna(subset=[k for k in val_keys if k in df.columns]).copy()
    # Берём последнее значение на эпоху (Lightning накапливает и пишет на epoch_end)
    agg_train = df_train.groupby("epoch").last().reset_index()
    agg_val   = df_val.groupby("epoch").last().reset_index()
    # Оставляем только нужные столбцы
    keep_train = ["epoch"] + [k for k in train_keys if k in agg_train.columns]
    keep_val   = ["epoch"] + [k for k in val_keys   if k in agg_val.columns]
    agg_train = agg_train[keep_train]
    agg_val   = agg_val[keep_val]
    # Сливаем train и val в одну таблицу
    merged = pd.merge(agg_train, agg_val, on="epoch", how="outer").sort_values("epoch")
    return merged

def plot_training_curves(metrics_csv: Path):

    train_keys = ["train_infoNCE_acc","train_loss","train_loss_con","train_loss_next","train_loss_mate",
              "train_topo_next_top1","train_topo_mate_top1"]

    val_keys   = ["val_infoNCE_acc","val_loss", "val_topo_next_top1","val_topo_mate_top1"]


    clean = load_and_clean_csv(str(metrics_csv), train_keys, val_keys)

    fig, axs = plt.subplots(2, 2, figsize=(10,8), constrained_layout=True)

    axs[0,0].plot(clean["epoch"], clean["train_loss"], label="train")
    axs[0,0].plot(clean["epoch"], clean.get("val_loss"), label="val")
    axs[0,0].set_title("Train Loss"); axs[0,0].legend()

    axs[0,1].plot(clean["epoch"], clean["val_infoNCE_acc"], label="val")
    axs[0,1].plot(clean["epoch"], clean["train_infoNCE_acc"], label="train")
    axs[0,1].set_title("Accuracy"); axs[0,1].legend()

    axs[1,0].plot(clean["epoch"], clean["train_topo_mate_top1"], label="train/topo_mate_top1")
    axs[1,0].plot(clean["epoch"], clean["val_topo_mate_top1"], label="val/topo_mate_top1")
    axs[1,0].set_title("Topo Mate@1"); axs[1,0].legend()

    axs[1,1].plot(clean["epoch"], clean["train_topo_next_top1"], label="train/topo_next_top1")
    axs[1,1].plot(clean["epoch"], clean["val_topo_next_top1"], label="val/topo_next_top1")
    axs[1,1].set_title("Topo Next@1"); axs[1,1].legend()