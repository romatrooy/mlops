"""Аугментация Fashion-MNIST: генерация дополнительных изображений и
сохранение нового датасета на диск.

Папки классов называются по индексу класса (00..09). Это важно: тогда
torchvision.datasets.ImageFolder выдаёт ровно тот же порядок меток, что и
FashionMNIST (T-shirt/top -> 0, ..., Ankle boot -> 9), и результаты
экспериментов с аугментацией сопоставимы с экспериментами без неё.
Соответствие индексов и названий классов сохраняется в classes.json.
"""
import argparse
import json
from pathlib import Path

import torch
import yaml
from torchvision import datasets, transforms
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm


def save_grayscale_image(image_tensor, path):
    """Сохраняет одноканальное изображение как grayscale-PNG.

    torchvision.utils.save_image прогоняет тензор через make_grid, который
    разворачивает 1 канал в 3, и на диск попадает RGB-файл (ломает обучение
    одноканальных моделей). to_pil_image сохраняет mode='L'.
    """
    to_pil_image(image_tensor.clamp(0, 1)).save(path)


def load_params(path="params.yaml"):
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


DEFAULT_TRANSFORMS = ["RandomHorizontalFlip", "RandomRotation", "ColorJitter", "RandomAffine"]

AVAILABLE_TRANSFORMS = {
    "RandomHorizontalFlip": lambda: transforms.RandomHorizontalFlip(p=0.5),
    "RandomRotation": lambda: transforms.RandomRotation(degrees=15),
    "ColorJitter": lambda: transforms.ColorJitter(brightness=0.3, contrast=0.3),
    "RandomAffine": lambda: transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    "RandomGrayscale": lambda: transforms.RandomGrayscale(p=0.2),
}


def get_aug_transform(transform_names=None):
    """Собирает трансформации по именам из params.yaml (augmentation.transforms)."""
    names = transform_names or DEFAULT_TRANSFORMS
    unknown = [n for n in names if n not in AVAILABLE_TRANSFORMS]
    if unknown:
        raise ValueError(f"Неизвестные аугментации {unknown}. "
                         f"Доступны: {sorted(AVAILABLE_TRANSFORMS)}")
    return transforms.Compose([AVAILABLE_TRANSFORMS[n]() for n in names]
                              + [transforms.ToTensor()])


def create_augmented_dataset(raw_dir="data/raw", augmented_dir="data/augmented",
                             num_aug_per_image=2, limit=None, seed=42,
                             transform_names=None):
    torch.manual_seed(seed)
    base_transform = transforms.ToTensor()
    aug_transform = get_aug_transform(transform_names)

    train_ds = datasets.FashionMNIST(root=raw_dir, train=True, download=False)
    classes = list(train_ds.classes)

    train_root = Path(augmented_dir) / "train"
    for idx in range(len(classes)):
        (train_root / f"{idx:02d}").mkdir(parents=True, exist_ok=True)

    total = len(train_ds) if limit is None else min(int(limit), len(train_ds))
    print(f"Генерация аугментированных изображений: {total} исходных x "
          f"{1 + num_aug_per_image} = {total * (1 + num_aug_per_image)} файлов")

    for idx in tqdm(range(total)):
        img, label = train_ds[idx]
        cls_dir = train_root / f"{label:02d}"

        # Оригинал
        save_grayscale_image(base_transform(img), cls_dir / f"orig_{idx:06d}.png")

        # Аугментированные копии
        for a in range(num_aug_per_image):
            save_grayscale_image(aug_transform(img), cls_dir / f"aug_{idx:06d}_{a}.png")

    mapping = {
        "class_to_idx": {f"{i:02d}": i for i in range(len(classes))},
        "class_names": classes,
        "num_augmented_per_image": num_aug_per_image,
        "transforms": transform_names or DEFAULT_TRANSFORMS,
        "source_samples": total,
        "seed": seed,
    }
    (Path(augmented_dir) / "classes.json").write_text(
        json.dumps(mapping, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Готово. Датасет сохранён в {augmented_dir}")
    print(f"Всего файлов изображений: {total * (1 + num_aug_per_image)}")


def main():
    params = load_params()
    aug_params = params.get("augmentation", {})

    parser = argparse.ArgumentParser(description="Аугментация Fashion-MNIST")
    parser.add_argument("--raw_dir", type=str, default="data/raw")
    parser.add_argument("--augmented_dir", type=str, default="data/augmented")
    parser.add_argument("--num_aug", type=int,
                        default=aug_params.get("num_augmented_per_image", 2))
    parser.add_argument("--limit", type=int, default=None,
                        help="ограничить число исходных изображений (для быстрой проверки)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    create_augmented_dataset(
        raw_dir=args.raw_dir,
        augmented_dir=args.augmented_dir,
        num_aug_per_image=args.num_aug,
        limit=args.limit,
        seed=args.seed,
        transform_names=aug_params.get("transforms"),
    )


if __name__ == "__main__":
    main()