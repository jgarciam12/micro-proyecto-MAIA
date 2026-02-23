
from __future__ import annotations

import tensorflow as tf


def build_augmentation():

    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomContrast(0.1),
        ],
        name="augmentation",
    )


def apply_preprocessing(ds, img_size=(224, 224), augment: bool = True):

    aug = build_augmentation() if augment else None
    rescale = tf.keras.layers.Rescaling(1.0 / 255)

    def _map(x, y):
        x = tf.image.resize(x, img_size)  # por seguridad si llega tamaño distinto
        x = rescale(x)
        if aug is not None:
            x = aug(x, training=True)
        return x, y

    return ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)