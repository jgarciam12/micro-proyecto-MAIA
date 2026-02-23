import numpy as np
import tensorflow as tf

class Predictor:
    def __init__(self, model_path: str, img_size: int = 224):
        self.model = tf.keras.models.load_model(model_path)
        self.img_size = img_size
        self.classes = ["Normal","Scol", "Spond"]

    def predict(self, image: np.ndarray) -> np.ndarray:
        image = image.resize((self.img_size, self.img_size))
        x = np.expand_dims(np.array(image), axis=0)

        pred = self.model.predict(x)
        idx = int(np.argmax(pred))

        return {
            "class_id": idx,
            "class_name": self.classes[idx],
            "confidence": float(pred[0][idx])
        }