from PIL import Image
from src.inference.predictor import Predictor
import os

def main():
    predictor = Predictor(
        model_path = "artifacts/models/scoliosis_model.keras",
        img_size = 224
    )
    PATH = "data/data-xray/224/224/Scol"
    imagenes = os.listdir(PATH)

    for imagename in imagenes:
        img = Image.open(os.path.join(PATH, imagename))

    #img = Image.open("/N15, Rt TAIS, F, 14 yrs.jpg")#)Normal/N1,N,40,M_1_0.jpg")

        result = predictor.predict(img)

        print("\n==== RESULTADO ====")
        print(result)

if __name__ == "__main__":
    main()
# Ejecutar: python -m scripts.test_predict