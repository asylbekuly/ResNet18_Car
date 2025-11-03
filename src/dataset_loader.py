import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Tuple

# ImageNet статистики для предобученных моделей (ResNet18)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def get_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Возвращает трансформы для train и для val/test.
    Train: аугментации + Normalize
    Val/Test: только приведение к размеру + ToTensor + Normalize
    """
    train_tf = transforms.Compose([
        # Немного сильнее, чем просто Resize: учим модель быть устойчивой к кропам/зуму
        transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return train_tf, eval_tf


def get_dataloaders(data_dir: str, batch_size: int = 32, num_workers: int = 2):
    """
    Собирает датасеты и лоадеры для train/val/test из структуры:
    data/
      train/CLASS_NAME/*.jpg
      val/CLASS_NAME/*.jpg
      test/CLASS_NAME/*.jpg
    """
    train_tf, eval_tf = get_transforms()

    train_path = os.path.join(data_dir, "train")
    val_path   = os.path.join(data_dir, "val")
    test_path  = os.path.join(data_dir, "test")

    train_dataset = datasets.ImageFolder(root=train_path, transform=train_tf)
    val_dataset   = datasets.ImageFolder(root=val_path,   transform=eval_tf)
    test_dataset  = datasets.ImageFolder(root=test_path,  transform=eval_tf)

    # На Mac (MPS) pin_memory можно оставить False; для CUDA — True
    pin_mem = torch.cuda.is_available()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    class_names = train_dataset.classes  # имена классов в алфавитном порядке

    return train_loader, val_loader, test_loader, class_names, train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    from collections import Counter

    data_dir = "/Users/market/DocumentsR/Coding/GitHub/car_brand_classifier/data"

    train_loader, val_loader, test_loader, class_names, train_dataset, val_dataset, test_dataset = get_dataloaders(
        data_dir=data_dir,
        batch_size=32,
        num_workers=2,
    )

    print("Найдены классы (в алфавитном порядке):")
    for idx, class_name in enumerate(class_names):
        print(f"{idx}: {class_name}")

    print("\nРазмеры датасетов:")
    print(f"train: {len(train_dataset)} изображений")
    print(f"val:   {len(val_dataset)} изображений")
    print(f"test:  {len(test_dataset)} изображений")

    def print_class_distribution(dataset, class_names, title):
        counts = Counter([label for _, label in dataset.samples])
        print(f"\nИзображений на класс ({title}):")
        for idx, class_name in enumerate(class_names):
            print(f"{class_name}: {counts.get(idx, 0)}")

    print_class_distribution(train_dataset, class_names, "train")
    print_class_distribution(val_dataset, class_names, "val")
    print_class_distribution(test_dataset, class_names, "test")
