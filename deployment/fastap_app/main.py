
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import io

from src.inference.predictor import Predictor

app = FastAPI(title="API clasificadora de enfermedad de columna")

predictor = Predictor(
    model_path = "artifacts/models/scoliosis_model.keras",
    img_size = 224
)

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/jpg", "image/png"]:
        raise HTTPException(status_code=400, detail="Formato inválido")

    content = await file.read()
    img = Image.open(io.BytesIO(content))

    return predictor.predict(img)

# Ejecutar: uvicorn deployment.fastap_app.main:app --reload

