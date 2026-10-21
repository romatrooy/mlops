import os
import torch
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import yaml

def get_transforms(image_size=28, train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

def prepare_dataloaders(raw_dir="data/raw", processed_dir="data/processed",
                        batch_size=64, val_split=0.1, seed=42):
    Path(processed_dir).mkdir(parents=True, exist_ok=True)
    
    full_train = datasets.FashionMNIST(
        root=raw_dir, train=True, download=False,
        transform=get_transforms(train=True)
    )
    test = datasets.FashionMNIST(
        root=raw_dir, train=False, download=False,
        transform=get_transforms(train=False)
    )
    
    val_size = int(len(full_train) * val_split)
    train_size = len(full_train) - val_size
    train, val = random_split(full_train, [train_size, val_size],
                              generator=torch.Generator().manual_seed(seed))
    
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Сохраняем информацию о сплитах
    torch.save({
        "train_indices": train.indices,
        "val_indices": val.indices,
    }, os.path.join(processed_dir, "split_info.pt"))
    
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    prepare_dataloaders(batch_size=params["data"]["batch_size"])