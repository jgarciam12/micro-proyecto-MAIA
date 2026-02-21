from __future__ import annotations

from data.load_data import DatasetConfig, load_train_val_datasets, compute_class_weights
from data.preprocessing import apply_preprocessing


def build_dataloaders(cfg: DatasetConfig):
    train_ds, val_ds, class_names = load_train_val_datasets(cfg)

    train_ds = apply_preprocessing(train_ds, img_size=cfg.img_size, augment=True)
    val_ds = apply_preprocessing(val_ds, img_size=cfg.img_size, augment=False)

    class_weights = compute_class_weights(train_ds, num_classes=len(class_names))
    return train_ds, val_ds, class_names, class_weights