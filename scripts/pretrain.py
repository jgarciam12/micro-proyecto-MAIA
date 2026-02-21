# scripts/pretrain.py
from __future__ import annotations

import argparse
import json
import os

import tensorflow as tf

from data.load_data import DatasetConfig
from data.loader import build_dataloaders
from training.trainer import train_model
from evaluation.metrics import compute_metrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--val_split", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--out_dir", type=str, default="artifacts/pretrain")
    return p.parse_args()


def build_baseline_model(num_classes: int, img_size: int):
    """
    Baseline robusto: MobileNetV2 frozen + head.
    Luego el equipo puede extenderlo a fine-tuning / tuner.
    """
    inputs = tf.keras.Input(shape=(img_size, img_size, 3))
    x = tf.keras.layers.Rescaling(1.0 / 255)(inputs)

    base = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_tensor=x,
    )
    base.trainable = False

    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    cfg = DatasetConfig(
        data_dir=args.data_dir,
        img_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        val_split=args.val_split,
        seed=args.seed,
    )

    train_ds, val_ds, class_names, class_weights = build_dataloaders(cfg)

    model = build_baseline_model(num_classes=len(class_names), img_size=args.img_size)

    _ = train_model(
        model=model,
        train_ds=train_ds,
        val_ds=val_ds,
        epochs=args.epochs,
        out_dir=args.out_dir,
        class_weights=class_weights,
    )

    # Guardar modelo final
    model_path = os.path.join(args.out_dir, "final_model.keras")
    model.save(model_path)

    # Métricas
    metrics = compute_metrics(model, val_ds, class_names)
    metrics["class_names"] = class_names
    metrics["model_path"] = model_path

    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("OK - artifacts guardados en:", args.out_dir)


if __name__ == "__main__":
    main()