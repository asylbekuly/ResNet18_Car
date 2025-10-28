import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

from dataset_loader import get_dataloaders


########################
# CONFIG
########################
DATA_DIR = "/Users/market/DocumentsR/Coding/GitHub/car_brand_classifier/data"
BATCH_SIZE = 32

EPOCHS_STAGE1 = 7   # учим только fc
EPOCHS_STAGE2 = 5   # доучиваем layer4

LR_STAGE1 = 1e-3    # выше lr для нового fc
LR_STAGE2 = 1e-4    # ниже lr для тонкой настройки feature extractor
WEIGHT_DECAY = 1e-4

MODEL_PATH = "best_model_finetuned.pth"


########################
# DEVICE PICKER
########################
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

device = get_device()
print(f"[INFO] Using device: {device}")


########################
# LOAD DATA
########################
(
    train_loader,
    val_loader,
    test_loader,
    class_names,
    train_dataset,
    val_dataset,
    test_dataset,
) = get_dataloaders(
    data_dir=DATA_DIR,
    batch_size=BATCH_SIZE
)

num_classes = len(class_names)
print(f"[INFO] num_classes = {num_classes}")
print(f"[INFO] classes: {class_names}")


########################
# BUILD MODEL
########################
def build_model(num_classes: int):
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)

    # сначала заморозим все слои
    for p in model.parameters():
        p.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model

model = build_model(num_classes).to(device)
criterion = nn.CrossEntropyLoss()


########################
# TRAIN / EVAL LOOPS
########################
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        total_loss += loss.item() * images.size(0)
        total_correct += torch.sum(preds == labels).item()
        total_seen += labels.size(0)

    return total_loss / total_seen, total_correct / total_seen


@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        _, preds = torch.max(outputs, 1)
        total_loss += loss.item() * images.size(0)
        total_correct += torch.sum(preds == labels).item()
        total_seen += labels.size(0)

    return total_loss / total_seen, total_correct / total_seen


########################
# PHASE 1: train only fc
########################
print("\n======= STAGE 1: train classifier head only (fc) =======")

# Только последний слой обучаем
optimizer = optim.Adam(model.fc.parameters(), lr=LR_STAGE1, weight_decay=WEIGHT_DECAY)

best_val_acc = 0.0

for epoch in range(EPOCHS_STAGE1):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc = eval_one_epoch(model, val_loader, criterion, device)

    print(f"Stage1 Epoch [{epoch+1}/{EPOCHS_STAGE1}]")
    print(f"  Train Loss: {train_loss:.4f}  |  Train Acc: {train_acc*100:.2f}%")
    print(f"  Val   Loss: {val_loss:.4f}    |  Val   Acc: {val_acc*100:.2f}%")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"  [INFO] Saved best model so far to {MODEL_PATH} (stage1)")

print(f"[INFO] After Stage1 best Val Acc: {best_val_acc*100:.2f}%")


########################
# PHASE 2: fine-tune layer4 (+fc)
########################
print("\n======= STAGE 2: fine-tune layer4 + fc =======")

# подгружаем лучшую модель со Stage1 (стабильно)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)

# размораживаем самый верхний блок фичей resnet18
# структура resnet18:
# model.layer1, model.layer2, model.layer3, model.layer4, model.fc
for p in model.layer4.parameters():
    p.requires_grad = True
for p in model.fc.parameters():
    p.requires_grad = True

# теперь оптимизируем и layer4, и fc
params_to_update = list(model.layer4.parameters()) + list(model.fc.parameters())

optimizer = optim.Adam(params_to_update, lr=LR_STAGE2, weight_decay=WEIGHT_DECAY)

for epoch in range(EPOCHS_STAGE2):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc = eval_one_epoch(model, val_loader, criterion, device)

    print(f"Stage2 Epoch [{epoch+1}/{EPOCHS_STAGE2}]")
    print(f"  Train Loss: {train_loss:.4f}  |  Train Acc: {train_acc*100:.2f}%")
    print(f"  Val   Loss: {val_loss:.4f}    |  Val   Acc: {val_acc*100:.2f}%")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"  [INFO] Saved best model so far to {MODEL_PATH} (stage2)")

print(f"[INFO] After Stage2 best Val Acc: {best_val_acc*100:.2f}%")


########################
# TEST EVAL with best model
########################
print("\n======= FINAL EVAL ON TEST SET =======")

# грузим лучший чекпоинт (лучшая валидация, не обязательно последняя эпоха)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

print("\n=== Classification report (per-class Precision / Recall / F1) ===")
print(classification_report(
    all_labels,
    all_preds,
    target_names=class_names,
    digits=4
))

cm = confusion_matrix(all_labels, all_preds)
print("=== Confusion Matrix (rows=true, cols=pred) ===")
print(cm)

print(f"[INFO] Done. Best model (after fine-tuning) saved at: {MODEL_PATH}")
