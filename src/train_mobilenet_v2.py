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

EPOCHS_STAGE1 = 7
EPOCHS_STAGE2 = 8

LR_STAGE1 = 1e-3
LR_STAGE2 = 3e-5
WEIGHT_DECAY = 1e-4

MODEL_PATH = "best_mobilenetv2_finetuned.pth"

########################
# DEVICE
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
(train_loader, val_loader, test_loader,
 class_names, train_dataset, val_dataset, test_dataset) = get_dataloaders(
    data_dir=DATA_DIR, batch_size=BATCH_SIZE
)

num_classes = len(class_names)
print(f"[INFO] num_classes = {num_classes}")
print(f"[INFO] classes: {class_names}")

########################
# BUILD MODEL
########################
def build_model(num_classes: int):
    weights = models.MobileNet_V2_Weights.DEFAULT
    model = models.mobilenet_v2(weights=weights)

    # замораживаем все слои
    for p in model.parameters():
        p.requires_grad = False

    # меняем последний классификатор
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model

model = build_model(num_classes).to(device)

########################
# CLASS WEIGHTS
########################
counts = Counter([label for _, label in train_dataset.samples])
total = sum(counts.values())
class_weights = torch.tensor(
    [total / (counts[idx] + 1e-8) for idx in range(num_classes)],
    dtype=torch.float32, device=device
)

criterion = nn.CrossEntropyLoss(weight=class_weights)

########################
# TRAIN / EVAL LOOPS
########################
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_correct, total_seen = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
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
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        total_loss += loss.item() * images.size(0)
        total_correct += torch.sum(preds == labels).item()
        total_seen += labels.size(0)
    return total_loss / total_seen, total_correct / total_seen

########################
# LOG FOR PLOTS
########################
history = {"epoch": [], "stage": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
global_epoch_idx = 0
best_val_acc = 0.0

########################
# STAGE 1: только classifier
########################
print("\n======= STAGE 1: train classifier head only (MobileNetV2) =======")
optimizer = optim.Adam(model.classifier[-1].parameters(), lr=LR_STAGE1, weight_decay=WEIGHT_DECAY)

for epoch in range(EPOCHS_STAGE1):
    tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
    va_loss, va_acc = eval_one_epoch(model, val_loader, criterion, device)
    global_epoch_idx += 1

    print(f"Stage1 Epoch [{epoch+1}/{EPOCHS_STAGE1}]")
    print(f"  Train Loss: {tr_loss:.4f}  |  Train Acc: {tr_acc*100:.2f}%")
    print(f"  Val   Loss: {va_loss:.4f}    |  Val   Acc: {va_acc*100:.2f}%")

    history["epoch"].append(global_epoch_idx)
    history["stage"].append("stage1")
    history["train_loss"].append(tr_loss)
    history["train_acc"].append(tr_acc * 100)
    history["val_loss"].append(va_loss)
    history["val_acc"].append(va_acc * 100)

    if va_acc > best_val_acc:
        best_val_acc = va_acc
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"  [INFO] Saved best model so far to {MODEL_PATH} (stage1)")

print(f"[INFO] After Stage1 best Val Acc: {best_val_acc*100:.2f}%")

########################
# STAGE 2: разморозим хвост + classifier
########################
print("\n======= STAGE 2: fine-tune last blocks + classifier (MobileNetV2) =======")
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)

# размораживаем последние inverted residual blocks, например с 14-го по конец
for idx, m in enumerate(model.features):
    if idx >= 14:
        for p in m.parameters():
            p.requires_grad = True
for p in model.classifier.parameters():
    p.requires_grad = True

params_to_update = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.Adam(params_to_update, lr=LR_STAGE2, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_STAGE2)

for epoch in range(EPOCHS_STAGE2):
    tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
    va_loss, va_acc = eval_one_epoch(model, val_loader, criterion, device)
    global_epoch_idx += 1

    print(f"Stage2 Epoch [{epoch+1}/{EPOCHS_STAGE2}]")
    print(f"  Train Loss: {tr_loss:.4f}  |  Train Acc: {tr_acc*100:.2f}%")
    print(f"  Val   Loss: {va_loss:.4f}    |  Val   Acc: {va_acc*100:.2f}%")

    history["epoch"].append(global_epoch_idx)
    history["stage"].append("stage2")
    history["train_loss"].append(tr_loss)
    history["train_acc"].append(tr_acc * 100)
    history["val_loss"].append(va_loss)
    history["val_acc"].append(va_acc * 100)

    if va_acc > best_val_acc:
        best_val_acc = va_acc
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"  [INFO] Saved best model so far to {MODEL_PATH} (stage2)")

    scheduler.step()

print(f"[INFO] After Stage2 best Val Acc: {best_val_acc*100:.2f}%")

########################
# TEST EVAL
########################
print("\n======= FINAL EVAL ON TEST SET (MobileNetV2) =======")
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

print("\n=== Classification report (per-class Precision / Recall / F1) ===")
print(classification_report(all_labels, all_preds, target_names=class_names, digits=4, zero_division=0))

cm = confusion_matrix(all_labels, all_preds)
print("=== Confusion Matrix (rows=true, cols=pred) ===")
print(cm)

print(f"[INFO] Done. Best model (MobileNetV2) saved at: {MODEL_PATH}")

########################
# PLOTS
########################
print("\n[INFO] Generating plots for presentation...")

df_hist = pd.DataFrame(history)

plt.figure(figsize=(8,5))
plt.plot(df_hist["epoch"], df_hist["train_acc"], marker='o', label='Train Acc (%)')
plt.plot(df_hist["epoch"], df_hist["val_acc"], marker='o', label='Val Acc (%)')
plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)")
plt.title("Accuracy over Training - MobileNetV2")
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig("accuracy_curve_mobilenetv2.png", dpi=200); plt.close()

plt.figure(figsize=(8,5))
plt.plot(df_hist["epoch"], df_hist["train_loss"], marker='o', label='Train Loss')
plt.plot(df_hist["epoch"], df_hist["val_loss"], marker='o', label='Val Loss')
plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.title("Loss over Training - MobileNetV2")
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig("loss_curve_mobilenetv2.png", dpi=200); plt.close()

print("[INFO] Saved: accuracy_curve_mobilenetv2.png, loss_curve_mobilenetv2.png")
print("[INFO] All plots are ready.")
