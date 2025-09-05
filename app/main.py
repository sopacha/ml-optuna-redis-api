from fastapi import FastAPI, UploadFile, File
from app.predict import load_model, preprocess_image
import torch.nn.functional as F
import torch

app = FastAPI(title="Optuna + Redis ML API")

model = load_model() # Load model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Run inference on an uploaded image."""

    # Save uploaded file temporarily
    image_path = f"temp_{file.filename}"
    with open(image_path, "wb") as f:
        f.write(await file.read())

    # Preprocess
    img_tensor = preprocess_image(image_path).to(DEVICE)

    # Run inference
    with torch.no_grad():
        output = model(img_tensor)
        probs = F.softmax(output, dim=1)
        pred_class = CLASSES[probs.argmax().item()]
        confidence = probs.max().item()

    return {"class": pred_class, "confidence": round(confidence, 4)}
