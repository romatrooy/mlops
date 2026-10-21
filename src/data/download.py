import os
from pathlib import Path
from torchvision import datasets

def download_fashion_mnist(raw_dir: str = "data/raw"):
    Path(raw_dir).mkdir(parents=True, exist_ok=True)
    
    train = datasets.FashionMNIST(
        root=raw_dir, train=True, download=True
    )
    test = datasets.FashionMNIST(
        root=raw_dir, train=False, download=True
    )
    print(f"Train samples: {len(train)}, Test samples: {len(test)}")
    return train, test

if __name__ == "__main__":
    download_fashion_mnist()