"""Phase 8 — Cross-dataset transfer.
Load ResNet-50 from Phase 6 (trained on PlantVillage 38 classes), test directly on:
  - PlantDoc (27 classes, field, no overlap with PlantVillage classes)
  - Cassava (5 classes, field, no overlap with PlantVillage classes)

For Cassava: since PV has no cassava classes, the trained ResNet-50 will misclassify
  all 5 cassava classes as the closest "leaf" class in PV (likely Tomato or Apple).
  This measures OOD behavior — what does the model do on completely unseen crops?
  We report:
    - top-1 / top-5 PV-class accuracy (formal)
    - OOD detection: AUROC using max-softmax (out-of-distribution = cassava)
    - entropy of predictions on cassava
    - For PlantDoc: same, but also top-5 for reference
    - The trained model has 38 classes; all 27 PD / 5 CV are OOD vs those 38

We do NOT train a label mapper; we measure the trained model's behavior.
"""
import os, json, sys, time, hashlib, random
import subprocess
print("Installing GPU torch...")
r = subprocess.run(
    ["pip", "install", "--quiet", "torch==2.5.0", "torchvision==0.20.0",
     "--index-url", "https://download.pytorch.org/whl/cu121"],
    check=False, capture_output=True, text=True, timeout=300)
print("pip install rc:", r.returncode)

import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

t0 = time.time()
report = {"phase": "8_cross_dataset_transfer", "captured_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Load PlantVillage model from Phase 6 — but we cannot use Kaggle's /kaggle/input for kernel output.
# We need to re-attach the model. Solution: re-train ResNet-50 inline on a small train split first.
# (We don't have the saved .pth from phase 6; the file is in /kaggle/working which is per-run.)
# So: this phase re-trains a quick ResNet-50 on PlantVillage (2 epochs, sufficient for cross-dataset test).
DATA_ROOT = "/kaggle/input/datasets/abdallahalidev/plantvillage-dataset/color"
if not os.path.isdir(DATA_ROOT):
    print("FATAL: PlantVillage not mounted at", DATA_ROOT)
    raise SystemExit(1)

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

print("Loading PlantVillage training subset (80%)...")
full_ds = datasets.ImageFolder(DATA_ROOT, transform=eval_tf)
n_total = len(full_ds)
classes = full_ds.classes
n_classes = len(classes)
print(f"PV total: {n_total}, classes: {n_classes}")
targets = np.array([s[1] for s in full_ds.samples])
idx = np.arange(n_total)
from sklearn.model_selection import train_test_split
tr_idx, tmp_idx, tr_y, tmp_y = train_test_split(idx, targets, test_size=0.2, random_state=42, stratify=targets)
print(f"train: {len(tr_idx)}")

train_ds = datasets.ImageFolder(DATA_ROOT, transform=train_tf)
train_loader = DataLoader(torch.utils.data.Subset(train_ds, tr_idx.tolist()),
                          batch_size=64, shuffle=True, num_workers=2, pin_memory=True)

model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
model.fc = nn.Linear(model.fc.in_features, n_classes)
model = model.to(device)
class_counts = np.bincount(targets, minlength=n_classes)
weights = (1.0 / np.maximum(class_counts, 1))
weights = weights / weights.sum() * n_classes
class_weights = torch.FloatTensor(weights).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

# Quick 2-epoch training for cross-dataset test (diagnostic; not publication-grade)
n_epochs_train = 2
for epoch in range(n_epochs_train):
    model.train()
    t_ep = time.time()
    correct, total = 0, 0
    for batch_i, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
        if batch_i % 100 == 0:
            print(f"  retrain ep{epoch+1} batch{batch_i}/{len(train_loader)} loss={loss.item():.4f} acc={correct/total:.4f}")
    print(f"retrain epoch {epoch+1}: acc={correct/total:.4f} ({time.time()-t_ep:.0f}s)")

# Save model
torch.save(model.state_dict(), "/kaggle/working/resnet50_pv_retrained.pth")
print("Saved /kaggle/working/resnet50_pv_retrained.pth")

# --- Now test on PlantDoc ---
print("\n=== Testing on PlantDoc ===")
PD_ROOT = "/kaggle/input/datasets/nirmalsankalana/plantdoc-dataset/test"
if not os.path.isdir(PD_ROOT):
    print("FATAL: PlantDoc not mounted at", PD_ROOT)
    raise SystemExit(1)
pd_full = datasets.ImageFolder(PD_ROOT, transform=eval_tf)
pd_classes = pd_full.classes
print(f"PD: {len(pd_full)} images, {len(pd_classes)} classes")
pd_loader = DataLoader(pd_full, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

model.eval()
pd_logits, pd_labels = [], []
with torch.no_grad():
    for x, y in pd_loader:
        x = x.to(device)
        out = model(x)
        pd_logits.append(out.cpu().numpy())
        pd_labels.append(y.numpy())
pd_logits = np.concatenate(pd_logits)
pd_labels = np.concatenate(pd_labels)
pd_probs = torch.softmax(torch.tensor(pd_logits), dim=1).numpy()
pd_preds = pd_probs.argmax(1)
pd_max_conf = pd_probs.max(1)
pd_entropy = -(pd_probs * np.log(pd_probs + 1e-10)).sum(1)

# PlantDoc has no overlap with PlantVillage classes — all PD labels are OOD
# Top-K accuracy on PV's 38-class space (formal)
top1 = (pd_preds == pd_labels).mean()  # Will be ~0% since classes don't overlap
top5 = 0
for i in range(len(pd_labels)):
    if pd_labels[i] in pd_preds[i:i+1]:
        pass  # no, top-5
# Actually we need top-5 accuracy properly
top5_preds = np.argsort(pd_probs, axis=1)[:, -5:]
top5 = sum(pd_labels[i] in top5_preds[i] for i in range(len(pd_labels))) / len(pd_labels)

# Confidence distribution
print(f"PD max-confidence mean: {pd_max_conf.mean():.4f}, std: {pd_max_conf.std():.4f}")
print(f"PD entropy mean: {pd_entropy.mean():.4f}, std: {pd_entropy.std():.4f}")
print(f"PD top-1 (label-match, will be 0 since no class overlap): {top1:.4f}")
print(f"PD top-5 (label-match, will be 0): {top5:.4f}")

# What does the model predict PD as?  Most common predicted PV class for PD images
from collections import Counter
pred_classes_count = Counter(pd_preds.tolist())
print(f"PD predicted as PV classes (top 5):")
for cls_idx, n in pred_classes_count.most_common(5):
    print(f"  {classes[cls_idx]}: {n} PD images")

# --- Now test on Cassava ---
print("\n=== Testing on Cassava ===")
CV_ROOT = "/kaggle/input/datasets/nirmalsankalana/cassava-leaf-disease-classification/data"
if not os.path.isdir(CV_ROOT):
    print("FATAL: Cassava not mounted at", CV_ROOT)
    raise SystemExit(1)
cv_full = datasets.ImageFolder(CV_ROOT, transform=eval_tf)
cv_classes = cv_full.classes
print(f"CV: {len(cv_full)} images, {len(cv_classes)} classes")
cv_loader = DataLoader(cv_full, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

cv_logits, cv_labels = [], []
with torch.no_grad():
    for x, y in cv_loader:
        x = x.to(device)
        out = model(x)
        cv_logits.append(out.cpu().numpy())
        cv_labels.append(y.numpy())
cv_logits = np.concatenate(cv_logits)
cv_labels = np.concatenate(cv_labels)
cv_probs = torch.softmax(torch.tensor(cv_logits), dim=1).numpy()
cv_preds = cv_probs.argmax(1)
cv_max_conf = cv_probs.max(1)
cv_entropy = -(cv_probs * np.log(cv_probs + 1e-10)).sum(1)

print(f"CV max-confidence mean: {cv_max_conf.mean():.4f}, std: {cv_max_conf.std():.4f}")
print(f"CV entropy mean: {cv_entropy.mean():.4f}, std: {cv_entropy.std():.4f}")
print(f"CV top-1 (label-match, no class overlap): {(cv_preds == cv_labels).mean():.4f}")

# --- OOD detection: AUROC for cassava-as-OOD using max-softmax ---
# Use PV in-domain (held-out 20%) as positive (in-distribution)
print("\n=== OOD detection: Cassava as OOD ===")
te_idx, va_idx, te_y, va_y = train_test_split(tmp_idx, tmp_y, test_size=0.5, random_state=42, stratify=tmp_y)
pv_te_loader = DataLoader(torch.utils.data.Subset(
    datasets.ImageFolder(DATA_ROOT, transform=eval_tf), te_idx.tolist()),
    batch_size=128, shuffle=False, num_workers=2, pin_memory=True)
pv_te_logits, pv_te_labels = [], []
with torch.no_grad():
    for x, y in pv_te_loader:
        x = x.to(device)
        out = model(x)
        pv_te_logits.append(out.cpu().numpy())
        pv_te_labels.append(y.numpy())
pv_te_logits = np.concatenate(pv_te_logits)
pv_te_probs = torch.softmax(torch.tensor(pv_te_logits), dim=1).numpy()
pv_te_max_conf = pv_te_probs.max(1)

# OOD label: 0 for PV (in-dist), 1 for CV (OOD)
y_true = np.concatenate([np.zeros(len(pv_te_max_conf)), np.ones(len(cv_max_conf))])
scores = np.concatenate([-pv_te_max_conf, -cv_max_conf])  # negative conf so higher = OOD
auroc_softmax = roc_auc_score(y_true, scores)
print(f"PV in-dist max conf: mean={pv_te_max_conf.mean():.4f}, std={pv_te_max_conf.std():.4f}")
print(f"CV OOD max conf:     mean={cv_max_conf.mean():.4f}, std={cv_max_conf.std():.4f}")
print(f"OOD AUROC (softmax): {auroc_softmax:.4f}")

# --- Save report ---
report["plantvillage_test"] = {
    "n": len(pv_te_max_conf),
    "max_conf_mean": round(float(pv_te_max_conf.mean()), 4),
    "max_conf_std": round(float(pv_te_max_conf.std()), 4),
}
report["plantdoc_results"] = {
    "n": len(pd_max_conf),
    "max_conf_mean": round(float(pd_max_conf.mean()), 4),
    "max_conf_std": round(float(pd_max_conf.std()), 4),
    "entropy_mean": round(float(pd_entropy.mean()), 4),
    "entropy_std": round(float(pd_entropy.std()), 4),
    "top1_label_match": round(float(top1), 4),
    "top5_label_match": round(float(top5), 4),
    "top_predicted_pv_classes": [
        {"pv_class": classes[int(cls_idx)], "n_pd_images": int(n)}
        for cls_idx, n in pred_classes_count.most_common(5)
    ],
    "note": "PlantDoc classes do not overlap PlantVillage; top-1=0 expected (OOD)."
}
report["cassava_results"] = {
    "n": len(cv_max_conf),
    "max_conf_mean": round(float(cv_max_conf.mean()), 4),
    "max_conf_std": round(float(cv_max_conf.std()), 4),
    "entropy_mean": round(float(cv_entropy.mean()), 4),
    "entropy_std": round(float(cv_entropy.std()), 4),
    "top1_label_match": round(float((cv_preds == cv_labels).mean()), 4),
    "note": "Cassava classes do not overlap PlantVillage; top-1=0 expected (OOD)."
}
report["ood_detection_cv_vs_pv"] = {
    "auroc_max_softmax": round(float(auroc_softmax), 4),
    "auroc_interpretation": "0.5 = random, 1.0 = perfect OOD detection",
    "decision_rule": "in-dist if max_softmax > threshold; OOD otherwise",
    "n_pv_in_dist": len(pv_te_max_conf),
    "n_cv_ood": len(cv_max_conf)
}
report["elapsed_sec"] = round(time.time() - t0, 2)
report_bytes = json.dumps(report, indent=2).encode()
report["sha256_self"] = hashlib.sha256(report_bytes).hexdigest()

with open("/kaggle/working/phase8_transfer_report.json", "w") as f:
    f.write(json.dumps(report, indent=2))
print("\nWROTE /kaggle/working/phase8_transfer_report.json")
print(json.dumps(report, indent=2))
