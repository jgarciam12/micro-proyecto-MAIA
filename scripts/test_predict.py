from PIL import Image
from src.inference.predictor import Predictor

def main():
    predictor = Predictor(
        model_path = "artifacts/models/scoliosis_model.keras",
        img_size = 224
    )

    img = Image.open("data/data-xray/224/224/Scol/N15, Rt TAIS, F, 14 yrs.jpg")#)Normal/N1,N,40,M_1_0.jpg")

    result = predictor.predict(img)

    print("\n==== RESULTADO ====")
    print(result)

if __name__ == "__main__":
    main()
# Ejecutar: python -m scripts.test_predict