import os
from pathlib import Path
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm
import torch

def create_augmented_dataset(
    raw_dir="data/raw",
    augmented_dir="data/augmented",
    num_aug_per_image=2
):
    Path(augmented_dir).mkdir(parents=True, exist_ok=True)
    
    # Более агрессивные аугментации
    aug_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
    ])
    
    base_transform = transforms.ToTensor()
    
    train_ds = datasets.FashionMNIST(root=raw_dir, train=True, download=False)
    
    # Создаём структуру папок по классам
    classes = train_ds.classes
    for cls in classes:
        (Path(augmented_dir) / "train" / cls).mkdir(parents=True, exist_ok=True)
    
    print("Генерация аугментированных изображений...")
    idx = 0
    for img, label in tqdm(train_ds):
        cls_name = classes[label]
        
        # Оригинал
        save_image(base_transform(img), 
                   Path(augmented_dir) / "train" / cls_name / f"orig_{idx:06d}.png")
        
        # Аугментации
        for a in range(num_aug_per_image):
            aug_img = aug_transform(img)
            save_image(aug_img, 
                       Path(augmented_dir) / "train" / cls_name / f"aug_{idx:06d}_{a}.png")
        idx += 1
    
    print(f"Готово. Аугментированный датасет сохранён в {augmented_dir}")
    print(f"Всего изображений примерно: {len(train_ds) * (1 + num_aug_per_image)}")

if __name__ == "__main__":
    create_augmented_dataset()