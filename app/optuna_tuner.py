import torch
import torch.nn as nn
import torch.optim as optim
from app.models import SimpleCNN
from app.utils import get_cifar10_loaders
import optuna
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "best_model.pth"

def train_and_evaluate(model, train_loader, test_loader, lr, epochs=1):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for _ in range(epochs):
        model.train()
        for data, target in train_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)

    return correct / total


def objective(trial):
    lr = trial.suggest_loguniform("lr", 1e-4, 1e-1)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

    train_loader, test_loader = get_cifar10_loaders(batch_size=batch_size)
    model = SimpleCNN().to(DEVICE)

    accuracy = train_and_evaluate(model, train_loader, test_loader, lr)
    return 1 - accuracy  # minimize error


def tune_hyperparameters(n_trials=10):
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    print("Best hyperparameters:", study.best_params)

    # Retrain final model with best params
    best_params = study.best_params
    train_loader, test_loader = get_cifar10_loaders(batch_size=best_params["batch_size"])
    model = SimpleCNN().to(DEVICE)
    _ = train_and_evaluate(model, train_loader, test_loader, best_params["lr"], epochs=3)

    # Save model
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Best model saved to {MODEL_PATH}")
    
    return {
        "best_params": best_params,
        "best_acc": 1 - study.best_value,
        "model_path": MODEL_PATH
    }


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    tune_hyperparameters(n_trials=10)
