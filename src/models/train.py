"""Обучение моделей на Fashion-MNIST (исходные и аугментированные данные).

Каждый запуск сохраняет:
  * метрики  -> reports/metrics_{model}_{orig|aug}.json  (их читает src/evaluate.py
                и отслеживает DVC через metrics в dvc.yaml);
  * чекпоинт -> models/{model}_{orig|aug}_best.pt;
  * трекинг эксперимента -> MLflow (локальная папка mlruns/).

Примеры:
  python src/models/train.py --model simple_cnn
  python src/models/train.py --model resnet18 --augmented --epochs 5
"""
import argparse
import inspect
import json
from pathlib import Path

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from tqdm import tqdm

FASHION_MNIST_CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


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


# ---------- Данные ----------
def load_params(path="params.yaml"):
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def limit_dataset(dataset, limit):
    """Ограничение размера набора (для быстрых проверок пайплайна)."""
    if limit and 0 < limit < len(dataset):
        return Subset(dataset, list(range(int(limit))))
    return dataset


def build_datasets(use_augmented: bool, image_size: int):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        # страховка: любые входные изображения приводятся к 1 каналу
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    test_ds = datasets.FashionMNIST(root="data/raw", train=False,
                                    download=False, transform=transform)

    if use_augmented:
        train_root = Path("data/augmented/train")
        if not train_root.exists():
            raise FileNotFoundError(
                f"Не найден {train_root}. Сначала выполните: python src/data/augment.py")
        train_ds = datasets.ImageFolder(train_root, transform=transform)

        expected = {f"{i:02d}": i for i in range(len(FASHION_MNIST_CLASSES))}
        if dict(train_ds.class_to_idx) != expected:
            raise ValueError(
                "Порядок классов в data/augmented не совпадает с FashionMNIST: "
                f"{train_ds.class_to_idx}. Перегенерируйте датасет: python src/data/augment.py")
    else:
        full = datasets.FashionMNIST(root="data/raw", train=True,
                                     download=False, transform=transform)
        split_info = Path("data/processed/split_info.pt")
        if split_info.exists():
            indices = torch.load(split_info, weights_only=False)["train_indices"]
            train_ds = Subset(full, list(indices))
        else:
            # тот же сплит, что и в src/data/preprocess.py
            train_size = int(0.9 * len(full))
            train_ds, _ = torch.utils.data.random_split(
                full, [train_size, len(full) - train_size],
                generator=torch.Generator().manual_seed(42))

    return train_ds, test_ds


# ---------- Эксперимент ----------
def run_experiment(model_name: str, use_augmented: bool, epochs: int, lr: float,
                   batch_size: int, num_workers: int, image_size: int,
                   experiment_name: str, limit=None, models_dir="models",
                   reports_dir="reports", params_file="params.yaml"):
    suffix = "aug" if use_augmented else "orig"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Model: {model_name} | "
          f"Dataset: {'augmented' if use_augmented else 'original'}")

    train_ds, test_ds = build_datasets(use_augmented, image_size)
    train_ds = limit_dataset(train_ds, limit)
    test_ds = limit_dataset(test_ds, limit)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers)

    model = get_model(model_name).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Пример входа нужен MLflow для сериализации модели (формат pt2)
    sample_input, _ = next(iter(test_loader))

    Path(models_dir).mkdir(parents=True, exist_ok=True)
    Path(reports_dir).mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(models_dir) / f"{model_name}_{suffix}_best.pt"
    metrics_path = Path(reports_dir) / f"metrics_{model_name}_{suffix}.json"

    mlflow.set_experiment(experiment_name)
    history = []
    best_acc = 0.0

    with mlflow.start_run(run_name=f"{model_name}_{suffix}"):
        mlflow.log_params({
            "model": model_name,
            "epochs": epochs,
            "lr": lr,
            "batch_size": batch_size,
            "image_size": image_size,
            "augmented": use_augmented,
            "train_size": len(train_ds),
            "dataset": "Fashion-MNIST",
        })
        if Path(params_file).exists():
            mlflow.log_artifact(params_file)

        for epoch in range(epochs):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            test_loss, test_acc, _, _ = evaluate(model, test_loader, criterion, device)

            history.append({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
            })
            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
            }, step=epoch)

            print(f"Epoch {epoch + 1}/{epochs} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")

            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(model.state_dict(), ckpt_path)

        # Финальная оценка
        _, final_acc, preds, targets = evaluate(model, test_loader, criterion, device)
        report = classification_report(targets, preds, output_dict=True)
        mlflow.log_metric("final_test_acc", final_acc)
        mlflow.log_metric("best_test_acc", best_acc)
        mlflow.log_dict(report, "classification_report.json")

        if "name" in inspect.signature(mlflow.pytorch.log_model).parameters:
            # MLflow 3.x: по умолчанию формат pt2 требует TensorSpec-сигнатуру,
            # для учебного проекта достаточно pickle
            mlflow.pytorch.log_model(model, name="model", input_example=sample_input,
                                     serialization_format="pickle")
        else:
            mlflow.pytorch.log_model(model, artifact_path="model",
                                     input_example=sample_input)  # MLflow 2.x

        metrics_record = {
            "model": model_name,
            "augmented": use_augmented,
            "dataset": "augmented" if use_augmented else "original",
            "epochs": epochs,
            "lr": lr,
            "batch_size": batch_size,
            "image_size": image_size,
            "num_workers": num_workers,
            "limit": limit,
            "train_size": len(train_ds),
            "test_size": len(test_ds),
            "best_test_acc": best_acc,
            "final_test_acc": final_acc,
            "history": history,
            "class_names": FASHION_MNIST_CLASSES,
            "classification_report": report,
        }
        metrics_path.write_text(json.dumps(metrics_record, indent=2, ensure_ascii=False),
                                encoding="utf-8")
        mlflow.log_artifact(str(metrics_path))
        if ckpt_path.exists():
            mlflow.log_artifact(str(ckpt_path))

    print(f"Best Test Accuracy: {best_acc:.4f}")
    print(f"Метрики сохранены: {metrics_path}")
    return best_acc


def main():
    params = load_params()
    data_params = params.get("data", {})
    train_params = params.get("train", {})

    parser = argparse.ArgumentParser(description="Обучение моделей на Fashion-MNIST")
    parser.add_argument("--model", type=str, required=True,
                        choices=["simple_cnn", "resnet18", "mobilenet_v2"])
    parser.add_argument("--augmented", action="store_true",
                        help="использовать аугментированный датасет (data/augmented)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None,
                        help="ограничить число изображений (быстрая проверка пайплайна)")
    parser.add_argument("--experiment", type=str, default=None,
                        help="имя эксперимента MLflow")
    parser.add_argument("--params", type=str, default="params.yaml")
    args = parser.parse_args()

    epochs = args.epochs if args.epochs is not None else train_params.get("epochs", 5)
    lr = args.lr if args.lr is not None else train_params.get("lr", 1e-3)
    batch_size = args.batch_size if args.batch_size is not None else data_params.get("batch_size", 64)
    num_workers = args.num_workers if args.num_workers is not None else data_params.get("num_workers", 2)
    image_size = args.image_size if args.image_size is not None else data_params.get("image_size", 28)

    experiment = args.experiment or (
        "fashion_mnist_aug" if args.augmented else "fashion_mnist_orig")

    run_experiment(
        model_name=args.model,
        use_augmented=args.augmented,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=image_size,
        experiment_name=experiment,
        limit=args.limit,
        params_file=args.params,
    )


if __name__ == "__main__":
    main()