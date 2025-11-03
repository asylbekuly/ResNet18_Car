import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter

from dataset_loader import get_dataloaders


########################
# CONFIG
########################
DATA_DIR = "/Users/market/DocumentsR/Coding/GitHub/car_brand_classifier/data"
BATCH_SIZE = 32

EPOCHS_STAGE1 = 7   # учим только fc
EPOCHS_STAGE2 = 8   # доучиваем layer3+layer4 подольше

LR_STAGE1 = 1e-3          # lr для нового fc
LR_STAGE2 = 5e-5          # меньший lr для тонкой настройки feature extractor
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

    # замораживаем все слои
    for p in model.parameters():
        p.requires_grad = False

    # меняем последний слой на наш (35 классов)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

model = build_model(num_classes).to(device)


########################
# CLASS WEIGHTS (для дисбаланса)
########################
# чем реже класс в train, тем больше вес его ошибки
counts = Counter([label for _, label in train_dataset.samples])
total = sum(counts.values())
class_weights = []
for idx in range(num_classes):
    freq = counts[idx]
    # простая инверсия частоты; можно нормировать, но не обязательно
    class_weights.append(total / (freq + 1e-8))
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)  # <— включены веса классов
# Если захочешь попробовать без весов: criterion = nn.CrossEntropyLoss()


########################
# TRAIN / EVAL LOOPS
########################
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_correct, total_seen = 0.0, 0, 0

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
    total_loss, total_correct, total_seen = 0.0, 0, 0

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
# LOG STRUCTURE FOR PLOTS
########################
history = {
    "epoch": [],
    "stage": [],
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": [],
}
global_epoch_idx = 0  # считаем эпохи подряд: 1..(EPOCHS_STAGE1+EPOCHS_STAGE2)


########################
# PHASE 1: train only fc
########################
print("\n======= STAGE 1: train classifier head only (fc) =======")

optimizer = optim.Adam(model.fc.parameters(), lr=LR_STAGE1, weight_decay=WEIGHT_DECAY)
best_val_acc = 0.0

for epoch in range(EPOCHS_STAGE1):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc = eval_one_epoch(model, val_loader, criterion, device)

    global_epoch_idx += 1
    print(f"Stage1 Epoch [{epoch+1}/{EPOCHS_STAGE1}]")
    print(f"  Train Loss: {train_loss:.4f}  |  Train Acc: {train_acc*100:.2f}%")
    print(f"  Val   Loss: {val_loss:.4f}    |  Val   Acc: {val_acc*100:.2f}%")

    history["epoch"].append(global_epoch_idx)
    history["stage"].append("stage1")
    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc * 100.0)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc * 100.0)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"  [INFO] Saved best model so far to {MODEL_PATH} (stage1)")

print(f"[INFO] After Stage1 best Val Acc: {best_val_acc*100:.2f}%")


########################
# PHASE 2: fine-tune layer3 + layer4 (+fc)
########################
print("\n======= STAGE 2: fine-tune layer3+layer4 + fc =======")

# загружаем лучшую модель со Stage1
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)

# размораживаем layer3, layer4 и fc
for p in model.layer3.parameters():
    p.requires_grad = True
for p in model.layer4.parameters():
    p.requires_grad = True
for p in model.fc.parameters():
    p.requires_grad = True

params_to_update = list(model.layer3.parameters()) + list(model.layer4.parameters()) + list(model.fc.parameters())
optimizer = optim.Adam(params_to_update, lr=LR_STAGE2, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_STAGE2)

for epoch in range(EPOCHS_STAGE2):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc = eval_one_epoch(model, val_loader, criterion, device)

    global_epoch_idx += 1
    print(f"Stage2 Epoch [{epoch+1}/{EPOCHS_STAGE2}]")
    print(f"  Train Loss: {train_loss:.4f}  |  Train Acc: {train_acc*100:.2f}%")
    print(f"  Val   Loss: {val_loss:.4f}    |  Val   Acc: {val_acc*100:.2f}%")

    history["epoch"].append(global_epoch_idx)
    history["stage"].append("stage2")
    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc * 100.0)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc * 100.0)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"  [INFO] Saved best model so far to {MODEL_PATH} (stage2)")

    scheduler.step()

print(f"[INFO] After Stage2 best Val Acc: {best_val_acc*100:.2f}%")


########################
# TEST EVAL with best model
########################
print("\n======= FINAL EVAL ON TEST SET =======")

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

all_preds, all_labels = [], []
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
    digits=4,
    zero_division=0,     # <<— без предупреждений
))

cm = confusion_matrix(all_labels, all_preds)
print("=== Confusion Matrix (rows=true, cols=pred) ===")
print(cm)

print(f"[INFO] Done. Best model (after fine-tuning) saved at: {MODEL_PATH}")


########################################
# VISUALIZATIONS FOR PRESENTATION
########################################
print("\n[INFO] Generating plots for presentation...")

# ---- 1) Accuracy & Loss curves ----
df_hist = pd.DataFrame(history)

plt.figure(figsize=(8,5))
plt.plot(df_hist["epoch"], df_hist["train_acc"], marker='o', label='Train Acc (%)')
plt.plot(df_hist["epoch"], df_hist["val_acc"], marker='o', label='Val Acc (%)')
plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)")
plt.title("Accuracy over Training (Stage1 → Stage2 Fine-Tuning)")
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig("accuracy_curve.png", dpi=200); plt.close()

plt.figure(figsize=(8,5))
plt.plot(df_hist["epoch"], df_hist["train_loss"], marker='o', label='Train Loss')
plt.plot(df_hist["epoch"], df_hist["val_loss"], marker='o', label='Val Loss')
plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.title("Loss over Training (Stage1 → Stage2 Fine-Tuning)")
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig("loss_curve.png", dpi=200); plt.close()

print("[INFO] Saved: accuracy_curve.png, loss_curve.png")


# ---- 2) Class balance (selected classes) ----
train_counts = Counter([label for _, label in train_dataset.samples])
cls_names_for_bar, cls_counts_for_bar = [], []
for idx, cls_name in enumerate(class_names):
    if cls_name in ["BMW", "Toyota", "Volkswagen", "Volvo", "camry70_kz", "lada_priora_kz"]:
        cls_names_for_bar.append(cls_name)
        cls_counts_for_bar.append(train_counts[idx])

plt.figure(figsize=(8,4))
plt.bar(cls_names_for_bar, cls_counts_for_bar)
plt.title("Train Data Balance (Selected Classes)")
plt.ylabel("Number of images")
plt.tight_layout()
plt.savefig("class_balance.png", dpi=200)
plt.close()

print("[INFO] Saved: class_balance.png")


# ---- 3) Gallery: sample predictions (Pred vs True) ----
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])

def imshow_denorm(img_tensor):
    """Показываем изображение, денормализуя из ImageNet-нормализации"""
    img = img_tensor.detach().cpu().numpy().transpose((1, 2, 0))
    img = (img * IMAGENET_STD) + IMAGENET_MEAN
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    plt.axis('off')

images_batch, labels_batch = next(iter(test_loader))
images_batch = images_batch.to(device)
labels_batch = labels_batch.to(device)
with torch.no_grad():
    outputs_batch = model(images_batch)
    _, preds_batch = torch.max(outputs_batch, 1)

plt.figure(figsize=(16,8))
for i in range(min(8, len(images_batch))):
    plt.subplot(2,4,i+1)
    imshow_denorm(images_batch[i])
    correct = preds_batch[i] == labels_batch[i]
    clr = "green" if correct else "red"
    plt.title(f"Pred: {class_names[preds_batch[i]]}\nTrue: {class_names[labels_batch[i]]}",
              fontsize=9, color=clr)
plt.tight_layout()
plt.savefig("sample_preds.png", dpi=200)
plt.close()

print("[INFO] Saved: sample_preds.png")


# ---- 4) Mini-confusion (Toyota vs camry70_kz vs lada_priora_kz) ----
key_classes = ["Toyota", "camry70_kz", "lada_priora_kz"]
key_idx = [class_names.index(c) for c in key_classes]
mini_cm = cm[np.ix_(key_idx, key_idx)].astype(float)

plt.figure(figsize=(5,4))
plt.imshow(mini_cm, interpolation='nearest')
plt.title("Confusion: Toyota vs Local Classes")
plt.colorbar()
ticks = np.arange(len(key_classes))
plt.xticks(ticks, key_classes, rotation=45, ha='right')
plt.yticks(ticks, key_classes)
plt.ylabel("True class"); plt.xlabel("Predicted class")
plt.tight_layout()
plt.savefig("confusion_mini.png", dpi=200)
plt.close()

print("[INFO] Saved: confusion_mini.png")

print("\n[INFO] All plots are ready: accuracy_curve.png, loss_curve.png, class_balance.png, sample_preds.png, confusion_mini.png")
