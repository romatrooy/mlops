"""Предобработка данных: трансформации, train/val сплит и DataLoader-ы.

Сплит сохраняется в data/processed/split_info.pt и переиспользуется в
src/models/train.py, чтобы эксперименты были воспроизводимыми.
"""
import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def load_params(path="params.yaml"):
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_transforms(image_size=28):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])


def prepare_dataloaders(raw_dir="data/raw", processed_dir="data/processed",
                        batch_size=64, num_workers=2, image_size=28,
                        val_split=0.1, seed=42):
    processed = Path(processed_dir)
    processed.mkdir(parents=True, exist_ok=True)

    train_tf = get_transforms(image_size)
    test_tf = get_transforms(image_size)

    full_train = datasets.FashionMNIST(root=raw_dir, train=True,
                                       download=False, transform=train_tf)
    test = datasets.FashionMNIST(root=raw_dir, train=False,
                                 download=False, transform=test_tf)

    val_size = int(len(full_train) * val_split)
    train_size = len(full_train) - val_size
    train, val = random_split(full_train, [train_size, val_size],
                              generator=torch.Generator().manual_seed(seed))

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers)

    # Сохраняем информацию о сплите (используется при обучении)
    torch.save({
        "train_indices": list(train.indices),
        "val_indices": list(val.indices),
        "val_split": val_split,
        "seed": seed,
    }, processed / "split_info.pt")
    torch.save({"classes": list(full_train.classes)}, processed / "classes.pt")

    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    return train_loader, val_loader, test_loader


def main():
    params = load_params()
    data_params = params.get("data", {})

    parser = argparse.ArgumentParser(description="Предобработка Fashion-MNIST")
    parser.add_argument("--raw_dir", type=str, default="data/raw")
    parser.add_argument("--processed_dir", type=str, default="data/processed")
    parser.add_argument("--batch_size", type=int, default=data_params.get("batch_size", 64))
    parser.add_argument("--num_workers", type=int, default=data_params.get("num_workers", 2))
    parser.add_argument("--image_size", type=int, default=data_params.get("image_size", 28))
    parser.add_argument("--val_split", type=float, default=data_params.get("val_split", 0.1))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    prepare_dataloaders(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        val_split=args.val_split,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()