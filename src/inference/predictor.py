import numpy as np
import tensorflow as tf

class Predictor:
    def __init__(self, model_path: str, img_size: int = 224):
        self.model = tf.keras.models.load_model(model_path)
        self.img_size = img_size
        self.classes = ["Normal","Scol", "Spond"]

    def predict(self, image):

        image = image.resize((self.img_size, self.img_size))

        x = np.array(image, dtype=np.float32) / 255.0
        x = np.expand_dims(x, axis=0)

        pred = self.model.predict(x)[0]

        idx = int(np.argmax(pred))

        probs = {
            self.classes[i]: float(pred[i]) for i in range(len(self.classes))
        }

        return {
            "class_id": idx,
            "class_name": self.classes[idx],
            "confidence": float(pred[idx]),
            "probabilities": probs
        }