
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import image_dataset_from_directory

AUTOTUNE = tf.data.AUTOTUNE


@dataclass(frozen=True)
class DatasetConfig:
    data_dir: str
    img_size: Tuple[int, int] = (224, 224)
    batch_size: int = 32
    val_split: float = 0.2
    seed: int = 123
    shuffle_buffer: int = 1000
    cache: bool = True


def _list_subdirs(path: str) -> list[str]:
   
    return sorted(
        d
        for d in os.listdir(path)
        if os.path.isdir(os.path.join(path, d)) and not d.startswith(".")
    )


def resolve_data_dir(data_dir: str) -> str:

    if not os.path.isdir(data_dir):
        raise ValueError(f"data_dir no existe o no es un directorio: {data_dir}")

    # Si ya tiene 2+ subcarpetas, asumimos que son clases
    subdirs = _list_subdirs(data_dir)
    if len(subdirs) >= 2:
        return data_dir

    # Si tiene exactamente 1 subcarpeta, bajamos un nivel si allí hay >=2 subcarpetas
    if len(subdirs) == 1:
        candidate = os.path.join(data_dir, subdirs[0])
        inner = _list_subdirs(candidate)
        if len(inner) >= 2:
            return candidate

    # Si no cumple, lo dejamos igual (y más adelante validamos clases)
    return data_dir


def load_train_val_datasets(cfg: DatasetConfig):

    data_dir = resolve_data_dir(cfg.data_dir)

    # Validación: asegurar que hay 2+ clases (subcarpetas)
    class_dirs = _list_subdirs(data_dir)
    if len(class_dirs) < 2:
        raise ValueError(
            f"Dataset inválido: se detectaron {len(class_dirs)} subcarpetas en '{data_dir}'. "
            f"Se esperaban >=2 clases (ej: Normal/Scol/Spond). "
            f"Subcarpetas encontradas: {class_dirs}"
        )

    train_ds = image_dataset_from_directory(
        data_dir,
        validation_split=cfg.val_split,
        subset="training",
        seed=cfg.seed,
        image_size=cfg.img_size,
        batch_size=cfg.batch_size,
        label_mode="int",
    )

    val_ds = image_dataset_from_directory(
        data_dir,
        validation_split=cfg.val_split,
        subset="validation",
        seed=cfg.seed,
        image_size=cfg.img_size,
        batch_size=cfg.batch_size,
        label_mode="int",
    )

    class_names: List[str] = list(train_ds.class_names)

    if cfg.cache:
        train_ds = train_ds.cache()
        val_ds = val_ds.cache()

    train_ds = train_ds.shuffle(cfg.shuffle_buffer, seed=cfg.seed).prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)

    return train_ds, val_ds, class_names


def compute_class_weights(train_ds, num_classes: int) -> dict[int, float]:
  
    labels = []
    for _, y in train_ds.unbatch():
        labels.append(int(y.numpy()))
    y = np.array(labels, dtype=np.int64)

    classes = np.arange(num_classes)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return {int(c): float(w) for c, w in zip(classes, weights)}
