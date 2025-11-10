import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torchvision import models

from dataset_loader import get_dataloaders

DATA_DIR = "/Users/market/DocumentsR/Coding/GitHub/car_brand_classifier/data"
BATCH_SIZE = 32

RESNET_PATH = "best_model_finetuned.pth"
MNV2_PATH   = "best_mobilenetv2_finetuned.pth"

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

device = get_device()
print(f"[INFO] Using device: {device}")

(_, _, test_loader,
 class_names, _, _, _) = get_dataloaders(data_dir=DATA_DIR, batch_size=BATCH_SIZE)
num_classes = len(class_names)

def build_resnet18(num_classes):
    w = models.ResNet18_Weights.DEFAULT
    m = models.resnet18(weights=w)
    in_features = m.fc.in_features
    m.fc = nn.Linear(in_features, num_classes)
    return m

def build_mobilenet_v2(num_classes):
    w = models.MobileNet_V2_Weights.DEFAULT
    m = models.mobilenet_v2(weights=w)
    in_features = m.classifier[-1].in_features
    m.classifier[-1] = nn.Linear(in_features, num_classes)
    return m

def eval_model(model, loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            p = out.argmax(1)
            preds.extend(p.cpu().numpy())
            labels.extend(y.cpu().numpy())
    preds = np.array(preds); labels = np.array(labels)
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
    return acc, prec, rec, f1

# ResNet18
resnet = build_resnet18(num_classes).to(device)
resnet.load_state_dict(torch.load(RESNET_PATH, map_location=device))
res_acc, res_prec, res_rec, res_f1 = eval_model(resnet, test_loader, device)

# MobileNetV2
mnv2 = build_mobilenet_v2(num_classes).to(device)
mnv2.load_state_dict(torch.load(MNV2_PATH, map_location=device))
mn_acc, mn_prec, mn_rec, mn_f1 = eval_model(mnv2, test_loader, device)

# Table
df = pd.DataFrame([
    ["ResNet18",    res_acc, res_prec, res_rec, res_f1],
    ["MobileNetV2", mn_acc,  mn_prec,  mn_rec,  mn_f1],
], columns=["Model", "Accuracy", "Precision(w)", "Recall(w)", "F1(w)"])

print("\n=== Test Comparison ===")
print(df.to_string(index=False))

# Bar plot
plt.figure(figsize=(8,5))
x = np.arange(len(df))
width = 0.2
plt.bar(x - 1.5*width, df["Accuracy"], width, label="Acc")
plt.bar(x - 0.5*width, df["Precision(w)"], width, label="Prec")
plt.bar(x + 0.5*width, df["Recall(w)"], width, label="Rec")
plt.bar(x + 1.5*width, df["F1(w)"], width, label="F1")
plt.xticks(x, df["Model"])
plt.ylabel("Score")
plt.ylim(0, 1.0)
plt.title("ResNet18 vs MobileNetV2 - Test set")
plt.legend()
plt.tight_layout()
plt.savefig("comparison_resnet18_mnv2.png", dpi=200)
plt.close()
print("[INFO] Saved: comparison_resnet18_mnv2.png")

# Сохраним таблицу в csv для отчета
df.to_csv("comparison_resnet18_mnv2.csv", index=False)
print("[INFO] Saved: comparison_resnet18_mnv2.csv")
