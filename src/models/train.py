import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import mlflow
import mlflow.pytorch
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import argparse

# ---------- Модели ----------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

def get_model(name: str, num_classes=10, pretrained=True):
    if name == "simple_cnn":
        return SimpleCNN(num_classes)
    elif name == "resnet18":
        model = models.resnet18(weights="DEFAULT" if pretrained else None)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    elif name == "mobilenet_v2":
        model = models.mobilenet_v2(weights="DEFAULT" if pretrained else None)
        model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        model.classifier[1] = nn.Linear(model.last_channel, num_classes)
        return model
    else:
        raise ValueError(f"Unknown model: {name}")

# ---------- Обучение ----------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    preds, targets = [], []
    for x, y in tqdm(loader, leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x.size(0)
        preds.extend(out.argmax(1).cpu().numpy())
        targets.extend(y.cpu().numpy())
    acc = accuracy_score(targets, preds)
    return running_loss / len(loader.dataset), acc

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    preds, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            running_loss += loss.item() * x.size(0)
            preds.extend(out.argmax(1).cpu().numpy())
            targets.extend(y.cpu().numpy())
    acc = accuracy_score(targets, preds)
    return running_loss / len(loader.dataset), acc, preds, targets

def run_experiment(model_name: str, data_dir: str, use_augmented: bool,
                   epochs: int, lr: float, batch_size: int, experiment_name: str):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, Model: {model_name}, Augmented: {use_augmented}")
    
    transform = transforms.Compose([
        transforms.Resize(28),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    if use_augmented:
        # ImageFolder для аугментированных данных
        train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=transform)
        # Для простоты берём оригинальный test
        test_ds = datasets.FashionMNIST(root="data/raw", train=False, download=False, transform=transform)
    else:
        full = datasets.FashionMNIST(root="data/raw", train=True, download=False, transform=transform)
        train_size = int(0.9 * len(full))
        val_size = len(full) - train_size
        train_ds, _ = torch.utils.data.random_split(full, [train_size, val_size])
        test_ds = datasets.FashionMNIST(root="data/raw", train=False, download=False, transform=transform)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    
    model = get_model(model_name).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=f"{model_name}_{'aug' if use_augmented else 'orig'}"):
        mlflow.log_params({
            "model": model_name,
            "epochs": epochs,
            "lr": lr,
            "batch_size": batch_size,
            "augmented": use_augmented,
            "dataset": "Fashion-MNIST"
        })
        
        best_acc = 0.0
        for epoch in range(epochs):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            test_loss, test_acc, _, _ = evaluate(model, test_loader, criterion, device)
            
            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_loss": test_loss,
                "test_acc": test_acc
            }, step=epoch)
            
            print(f"Epoch {epoch+1}/{epochs} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
            
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(model.state_dict(), f"models/{model_name}_{'aug' if use_augmented else 'orig'}_best.pt")
        
        # Финальная оценка
        _, final_acc, preds, targets = evaluate(model, test_loader, criterion, device)
        mlflow.log_metric("final_test_acc", final_acc)
        
        report = classification_report(targets, preds, output_dict=True)
        mlflow.log_dict(report, "classification_report.json")
        
        # Логируем модель
        mlflow.pytorch.log_model(model, "model")
        
        print(f"Best Test Accuracy: {best_acc:.4f}")
        return best_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--augmented", action="store_true")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    
    os.makedirs("models", exist_ok=True)
    
    experiment = "fashion_mnist_aug" if args.augmented else "fashion_mnist_orig"
    data_dir = "data/augmented" if args.augmented else "data/raw"
    
    run_experiment(
        model_name=args.model,
        data_dir=data_dir,
        use_augmented=args.augmented,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        experiment_name=experiment
    )