from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from app.optuna_tuner import tune_hyperparameters
from app.predict import load_model, preprocess_image
import torch.nn.functional as F
import torch
import redis
import hashlib
import json

# Connect to Redis
r = redis.Redis(host="localhost", port=6379, db=0)

app = FastAPI(title="Optuna + Redis ML API")

model = load_model() # Load model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Check cache
    file_bytes = await file.read()

    file_hash = hashlib.md5(file_bytes).hexdigest()

    if r.exists(file_hash):
        cached = json.loads(r.get(file_hash))
        return {"cached": True, **cached}

    # Otherwise, run inference
    image_path = f"temp_{file.filename}"
    with open(image_path, "wb") as f:
        f.write(file_bytes)

    img_tensor = preprocess_image(image_path).to(DEVICE)

    with torch.no_grad():
        output = model(img_tensor)
        probs = F.softmax(output, dim=1)
        pred_class = CLASSES[probs.argmax().item()]
        confidence = probs.max().item()

    result = {"class": pred_class, "confidence": round(confidence, 4)}

    # Save to Redis with expiration (e.g., 1 hour)
    r.setex(file_hash, 3600, json.dumps(result))

    return {"cached": False, **result}

@app.post("/train")
async def train(background_tasks: BackgroundTasks, n_trials: int = 5):
    
    def training_job(n_trials: int):
        result = tune_hyperparameters(n_trials=n_trials)
        global model
        model = load_model()
        print("Training finished:", result)

    background_tasks.add_task(training_job, n_trials)

    return {"status": "training started" }