import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from app.models import SimpleCNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "best_model.pth"

# CIFAR-10 classes
CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

def load_model():
    model = SimpleCNN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

def preprocess_image(image_path: str):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # CIFAR10 size
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0)  

def predict(image_path: str):
    model = load_model()
    img_tensor = preprocess_image(image_path).to(DEVICE)

    with torch.no_grad():
        output = model(img_tensor)
        probs = F.softmax(output, dim=1)
        pred_class = CLASSES[probs.argmax().item()]
        confidence = probs.max().item()
    print(f"Predicted class: {pred_class} with confidence {confidence:.4f}")
    return {"class": pred_class, "confidence": round(confidence, 4)}


if __name__ == "__main__":
    result = predict("data/0013.jpg")
