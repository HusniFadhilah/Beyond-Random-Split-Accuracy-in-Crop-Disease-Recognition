"""Phase 6 — Baseline random split: ResNet-50 on PlantVillage.
Diagnostic run: 3 epochs, 80/10/10 stratified random split, 224px, batch 64.
Goal: in-domain reference accuracy for RQ1/RQ2 (random vs group/cross-dataset gap).
"""
import os, json, sys, time, hashlib, random

# Step 1: install GPU torch (Kaggle default is CPU)
print("Installing PyTorch GPU...")
import subprocess
r = subprocess.run(
    ["pip", "install", "--quiet", "torch==2.5.0", "torchvision==0.20.0",
     "--index-url", "https://download.pytorch.org/whl/cu121"],
    check=False, capture_output=True, text=True, timeout=300)
print("pip install rc:", r.returncode)
print(r.stdout[-300:])

# Verify CUDA
import torch
print(f"torch {torch.__version__}, cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU mem: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

t0 = time.time()
report = {"phase": "6_baseline_resnet50", "captured_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

from torchvision import datasets, transforms, models
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, balanced_accuracy_score

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

DATA_ROOT = "/kaggle/input/datasets/abdallahalidev/plantvillage-dataset/color"
if not os.path.isdir(DATA_ROOT):
    print("FATAL: PlantVillage not mounted at", DATA_ROOT)
    print("Datasets present:", os.listdir("/kaggle/input/datasets/") if os.path.isdir("/kaggle/input/datasets") else "NONE")
    raise SystemExit(1)

# Transforms
train_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
eval_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Build dataset
print("Loading full PlantVillage dataset...")
full_ds = datasets.ImageFolder(DATA_ROOT, transform=eval_tf)
n_total = len(full_ds)
classes = full_ds.classes
n_classes = len(classes)
print(f"Total images: {n_total}, classes: {n_classes}")

# Stratified random split 80/10/10
targets = np.array([s[1] for s in full_ds.samples])
idx = np.arange(n_total)
tr_idx, tmp_idx, tr_y, tmp_y = train_test_split(idx, targets, test_size=0.2, random_state=42, stratify=targets)
va_idx, te_idx, va_y, te_y = train_test_split(tmp_idx, tmp_y, test_size=0.5, random_state=42, stratify=tmp_y)
print(f"Split: train={len(tr_idx)}, val={len(va_idx)}, test={len(te_idx)}")

train_ds = datasets.ImageFolder(DATA_ROOT, transform=train_tf)
val_ds = datasets.ImageFolder(DATA_ROOT, transform=eval_tf)
test_ds = datasets.ImageFolder(DATA_ROOT, transform=eval_tf)
train_loader = DataLoader(Subset(train_ds, tr_idx.tolist()), batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(Subset(val_ds, va_idx.tolist()), batch_size=128, shuffle=False, num_workers=2, pin_memory=True)
test_loader = DataLoader(Subset(test_ds, te_idx.tolist()), batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

# Model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
model.fc = nn.Linear(model.fc.in_features, n_classes)
model = model.to(device)

# Class weights for imbalance
class_counts = np.bincount(targets, minlength=n_classes)
weights = (1.0 / np.maximum(class_counts, 1))
weights = weights / weights.sum() * n_classes
class_weights = torch.FloatTensor(weights).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

# 3-epoch diagnostic
n_epochs = 3
history = []
best_val_f1 = 0.0
for epoch in range(n_epochs):
    model.train()
    tr_loss, tr_correct, tr_n = 0, 0, 0
    t_ep = time.time()
    for batch_i, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        tr_loss += loss.item() * x.size(0)
        tr_correct += (out.argmax(1) == y).sum().item()
        tr_n += x.size(0)
        if batch_i % 50 == 0:
            print(f"  ep{epoch+1} batch{batch_i}/{len(train_loader)} loss={loss.item():.4f}")
    tr_loss /= tr_n
    tr_acc = tr_correct / tr_n

    # Validation
    model.eval()
    va_preds, va_labels = [], []
    va_loss = 0.0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            va_loss += criterion(out, y).item() * x.size(0)
            va_preds.extend(out.argmax(1).cpu().tolist())
            va_labels.extend(y.cpu().tolist())
    va_loss /= len(va_idx)
    va_f1 = f1_score(va_labels, va_preds, average="macro", zero_division=0)
    va_ba = balanced_accuracy_score(va_labels, va_preds)
    history.append({
        "epoch": epoch + 1,
        "tr_loss": round(tr_loss, 4), "tr_acc": round(tr_acc, 4),
        "va_loss": round(va_loss, 4), "va_f1_macro": round(va_f1, 4),
        "va_balanced_acc": round(va_ba, 4),
        "elapsed_sec": round(time.time() - t_ep, 2)
    })
    print(f"epoch {epoch+1}: tr_loss={tr_loss:.4f} tr_acc={tr_acc:.4f} va_f1={va_f1:.4f} va_ba={va_ba:.4f} ({history[-1]['elapsed_sec']}s)")
    if va_f1 > best_val_f1:
        best_val_f1 = va_f1
        torch.save(model.state_dict(), "/kaggle/working/best_resnet50_pv.pth")

# Test eval
model.load_state_dict(torch.load("/kaggle/working/best_resnet50_pv.pth"))
model.eval()
te_preds, te_labels = [], []
te_loss = 0.0
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        te_loss += criterion(out, y).item() * x.size(0)
        te_preds.extend(out.argmax(1).cpu().tolist())
        te_labels.extend(y.cpu().tolist())
te_loss /= len(te_idx)
te_f1 = f1_score(te_labels, te_preds, average="macro", zero_division=0)
te_ba = balanced_accuracy_score(te_labels, te_preds)

report["history"] = history
report["test_metrics"] = {
    "loss": round(te_loss, 4),
    "f1_macro": round(te_f1, 4),
    "balanced_accuracy": round(te_ba, 4),
    "n_test": len(te_idx)
}
report["n_train"] = len(tr_idx)
report["n_val"] = len(va_idx)
report["n_classes"] = n_classes
report["elapsed_sec"] = round(time.time() - t0, 2)
report_bytes = json.dumps(report, indent=2).encode()
report["sha256_self"] = hashlib.sha256(report_bytes).hexdigest()

with open("/kaggle/working/phase6_baseline_report.json", "w") as f:
    f.write(json.dumps(report, indent=2))
print("WROTE /kaggle/working/phase6_baseline_report.json")
print(json.dumps(report["test_metrics"], indent=2))
print(f"Total elapsed: {report['elapsed_sec']} sec, sha256={report['sha256_self']}")
