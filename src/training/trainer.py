from __future__ import annotations

import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


def build_callbacks(out_dir: str):

    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, "best_model.keras")

    return [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
        ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True),
    ]


def train_model(
    model: tf.keras.Model,
    train_ds,
    val_ds,
    epochs: int,
    out_dir: str,
    class_weights: dict[int, float] | None = None,
):
    """
    Entrena el modelo y retorna el history.
    """
    callbacks = build_callbacks(out_dir)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights,
    )
    return history