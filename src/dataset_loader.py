import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_dataloaders(data_dir, batch_size=32):

    # Аугментации только для train (чтобы увеличить разнообразие и помочь с маленькими классами типа lada_priora_kz)
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor()
    ])

    # Для val и test мы НЕ искажаем картинку, только приводим к одному размеру и тензору
    eval_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_path = os.path.join(data_dir, "train")
    val_path = os.path.join(data_dir, "val")
    test_path = os.path.join(data_dir, "test")

    # ImageFolder ожидает структуру data/split/class_name/*.jpg
    train_dataset = datasets.ImageFolder(root=train_path, transform=train_transforms)
    val_dataset   = datasets.ImageFolder(root=val_path,   transform=eval_transforms)
    test_dataset  = datasets.ImageFolder(root=test_path,  transform=eval_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    class_names = train_dataset.classes  # список имён классов по алфавиту

    return train_loader, val_loader, test_loader, class_names, train_dataset, val_dataset, test_dataset


if __name__ == "__main__":


    data_dir = "/Users/market/DocumentsR/Coding/GitHub/car_brand_classifier/data"


    train_loader, val_loader, test_loader, class_names, train_dataset, val_dataset, test_dataset = get_dataloaders(
        data_dir=data_dir,
        batch_size=32
    )

    print("Найдены классы (в алфавитном порядке):")
    for idx, class_name in enumerate(class_names):
        print(f"{idx}: {class_name}")

    print("\nРазмеры датасетов:")
    print(f"train: {len(train_dataset)} изображений")
    print(f"val:   {len(val_dataset)} изображений")
    print(f"test:  {len(test_dataset)} изображений")


    from collections import Counter
    train_counts = Counter([label for _, label in train_dataset.samples])
    print("\nИзображений на класс (train):")
    for idx, class_name in enumerate(class_names):
        count = train_counts.get(idx, 0)
        print(f"{class_name}: {count}")


    def print_class_distribution(dataset, class_names, title):
        counts = Counter([label for _, label in dataset.samples])
        print(f"\nИзображений на класс ({title}):")
        for idx, class_name in enumerate(class_names):
            print(f"{class_name}: {counts.get(idx, 0)}")


    print_class_distribution(train_dataset, class_names, "train")
    print_class_distribution(val_dataset, class_names, "val")
    print_class_distribution(test_dataset, class_names, "test")
