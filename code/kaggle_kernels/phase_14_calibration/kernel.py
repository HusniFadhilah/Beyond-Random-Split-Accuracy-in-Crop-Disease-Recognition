"""Phase 14 — Calibration: in-domain vs cross-dataset.
Compute ECE, Brier, NLL on:
  - PlantVillage test set (in-domain reference)
  - PlantDoc (cross-dataset field)
  - Cassava (cross-dataset field)
Method: temperature scaling fit on PV val set, then evaluate on test sets.
Reports: pre-calibration and post-calibration ECE / Brier / NLL.
"""
import os, json, sys, time, hashlib, random
import subprocess
print("Installing GPU torch...")
r = subprocess.run(
    ["pip", "install", "--quiet", "torch==2.5.0", "torchvision==0.20.0",
     "--index-url", "https://download.pytorch.org/whl/cu121"],
    check=False, capture_output=True, text=True, timeout=300)

import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss, log_loss

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

t0 = time.time()
report = {"phase": "14_calibration", "captured_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

# Load PlantVillage, train 2-epoch ResNet-50 (diagnostic)
DATA_ROOT = "/kaggle/input/datasets/abdallahalidev/plantvillage-dataset/color"
PD_ROOT = "/kaggle/input/datasets/nirmalsankalana/plantdoc-dataset/test"
CV_ROOT = "/kaggle/input/datasets/nirmalsankalana/cassava-leaf-disease-classification/data"

eval_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
train_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

full = datasets.ImageFolder(DATA_ROOT, transform=eval_tf)
classes = full.classes; n_classes = len(classes)
targets = np.array([s[1] for s in full.samples])
idx = np.arange(len(full))
tr_idx, tmp_idx, tr_y, tmp_y = train_test_split(idx, targets, test_size=0.2, random_state=42, stratify=targets)
va_idx, te_idx, va_y, te_y = train_test_split(tmp_idx, tmp_y, test_size=0.5, random_state=42, stratify=tmp_y)
print(f"train={len(tr_idx)}, val={len(va_idx)}, test={len(te_idx)}")

train_loader = DataLoader(torch.utils.data.Subset(
    datasets.ImageFolder(DATA_ROOT, transform=train_tf), tr_idx.tolist()),
    batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(torch.utils.data.Subset(
    datasets.ImageFolder(DATA_ROOT, transform=eval_tf), va_idx.tolist()),
    batch_size=128, shuffle=False, num_workers=2, pin_memory=True)
test_loader = DataLoader(torch.utils.data.Subset(
    datasets.ImageFolder(DATA_ROOT, transform=eval_tf), te_idx.tolist()),
    batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
model.fc = nn.Linear(model.fc.in_features, n_classes)
model = model.to(device)
class_counts = np.bincount(targets, minlength=n_classes)
weights = (1.0 / np.maximum(class_counts, 1)); weights = weights/weights.sum()*n_classes
class_weights = torch.FloatTensor(weights).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

n_ep = 2
for ep in range(n_ep):
    model.train()
    t_ep = time.time()
    correct, total = 0, 0
    for batch_i, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward(); optimizer.step()
        correct += (out.argmax(1) == y).sum().item(); total += x.size(0)
        if batch_i % 100 == 0:
            print(f"  ep{ep+1} batch{batch_i}/{len(train_loader)} loss={loss.item():.4f}")
    print(f"epoch {ep+1}: train acc={correct/total:.4f} ({time.time()-t_ep:.0f}s)")

# Function to extract logits + labels
@torch.no_grad()
def extract(loader, model):
    logits, labels = [], []
    for x, y in loader:
        x = x.to(device)
        out = model(x)
        logits.append(out.cpu().numpy()); labels.append(y.numpy())
    return np.concatenate(logits), np.concatenate(labels)

# Get val + test + PD + CV logits
print("Extracting val (for temp scaling)...")
val_logits, val_labels = extract(val_loader, model)
print("Extracting test (in-domain)...")
test_logits, test_labels = extract(test_loader, model)
print("Extracting PlantDoc (cross-dataset)...")
pd_full = datasets.ImageFolder(PD_ROOT, transform=eval_tf)
pd_loader = DataLoader(pd_full, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)
pd_logits, pd_labels = extract(pd_loader, model)
# PlantDoc has its own 27 classes (no overlap with PV 38). For calibration, we need to map
# PlantDoc labels to PV's 38-class space or treat as separate. Decision: for ECE we use the
# softmax of PV 38 classes, comparing top-1 prediction's confidence against actual correctness
# in PV's label space. Since PD labels don't map, all PD predictions are "incorrect" by definition.
# This is intentional: it tests whether confidence is well-calibrated when applied to OOD.
print("Extracting Cassava (cross-dataset)...")
cv_full = datasets.ImageFolder(CV_ROOT, transform=eval_tf)
cv_loader = DataLoader(cv_full, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)
cv_logits, cv_labels = extract(cv_loader, model)

def softmax(z, T=1.0):
    z = z / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z); return e / e.sum(axis=1, keepdims=True)

def ece(probs, labels, n_bins=15):
    """Expected Calibration Error."""
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(float)
    bins = np.linspace(0, 1, n_bins+1)
    ece_val = 0.0
    bin_data = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            bin_data.append({"bin": f"({lo:.2f},{hi:.2f}]", "n": 0, "acc": None, "conf": None})
            continue
        bin_acc = accuracies[mask].mean()
        bin_conf = confidences[mask].mean()
        ece_val += mask.sum() * abs(bin_acc - bin_conf)
        bin_data.append({"bin": f"({lo:.2f},{hi:.2f}]", "n": int(mask.sum()),
                         "acc": round(float(bin_acc), 4), "conf": round(float(bin_conf), 4)})
    ece_val /= len(probs)
    return float(ece_val), bin_data

def brier_multiclass(probs, labels, n_classes):
    """Mean Brier score (one-hot vs probs)."""
    onehot = np.eye(n_classes)[labels]
    return float(((probs - onehot) ** 2).sum(axis=1).mean())

def nll(probs, labels, eps=1e-12):
    return float(-np.log(np.clip(probs[np.arange(len(labels)), labels], eps, 1)).mean())

# Find best temperature on PV val
def temp_nll(T, logits, labels):
    p = softmax(logits, T)
    return nll(p, labels)

# Grid search
Ts = np.linspace(0.5, 3.0, 26)
val_nlls = [temp_nll(T, val_logits, val_labels) for T in Ts]
best_T = Ts[int(np.argmin(val_nlls))]
print(f"Best temperature on PV val: T={best_T:.3f} (NLL={min(val_nlls):.4f})")

# Evaluate
def evaluate(name, logits, labels, n_classes_used, with_T=True):
    T = best_T if with_T else 1.0
    probs = softmax(logits, T)
    e, bins = ece(probs, labels)
    brier = brier_multiclass(probs, labels, n_classes_used)
    nll_v = nll(probs, labels)
    acc = float((probs.argmax(1) == labels).mean())
    f1 = None
    from sklearn.metrics import f1_score
    f1 = f1_score(labels, probs.argmax(1), average="macro", zero_division=0)
    return {
        "ece": round(e, 4), "brier": round(brier, 4), "nll": round(nll_v, 4),
        "acc": round(acc, 4), "f1_macro": round(float(f1), 4),
        "n_samples": int(len(labels)), "n_classes": int(n_classes_used),
        "temperature": round(float(best_T if with_T else 1.0), 3),
        "bin_data_sample": bins[:5]
    }

# For PD and CV, classes are not the same as PV. We use PV's 38-class probs and assign each
# PD/CV image a "label" = argmax of PV prediction. This is synthetic but tests calibration:
# if model is well-calibrated, top-1 confidence should match actual top-1 accuracy.
# Since PD/CV have different class structure, acc on PV-class space is meaningless; we report
# only confidence-based metrics: ECE on confidence vs (predicted == pseudo-label).
results = {}
print("Evaluating in-domain test (with T scaling)...")
results["in_domain_test_T_scaled"] = evaluate("pv_test", test_logits, test_labels, n_classes, with_T=True)
print("Evaluating in-domain test (no T scaling)...")
results["in_domain_test_raw"] = evaluate("pv_test", test_logits, test_labels, n_classes, with_T=False)

# Cross-dataset: for OOD, we report confidence distribution
@torch.no_grad()
def conf_stats(logits, name):
    p = softmax(logits, 1.0)
    return {
        "n": int(len(p)),
        "max_conf_mean": round(float(p.max(1).mean()), 4),
        "max_conf_std": round(float(p.max(1).std()), 4),
        "top5_sum_mean": round(float(np.sort(p, axis=1)[:, -5:].sum(1).mean()), 4),
    }

print("PlantDoc confidence stats...")
results["plantdoc_OOD_conf"] = conf_stats(pd_logits, "pd")
print("Cassava confidence stats...")
results["cassava_OOD_conf"] = conf_stats(cv_logits, "cv")

# Calibration on pseudo-OOD: temperature-scaled probs, treat top-1 as pseudo-correct
def evaluate_ood(name, logits):
    p_T = softmax(logits, best_T)
    p_raw = softmax(logits, 1.0)
    pred_T = p_T.argmax(1); pred_raw = p_raw.argmax(1)
    conf_T = p_T.max(1); conf_raw = p_raw.max(1)
    # Pseudo-correctness = top-1 prediction's confidence vs top-1 acc
    # For OOD, the "true" label is unknown; calibration is undefined.
    # But we can measure: of those predicted with conf>0.9, what fraction stays the same prediction after T-scaling?
    return {
        "n": int(len(p_T)),
        "raw_max_conf_mean": round(float(conf_raw.mean()), 4),
        "T_max_conf_mean": round(float(conf_T.mean()), 4),
        "raw_pred_frac_changing_after_T": round(float((pred_raw != pred_T).mean()), 4),
        "raw_pred_top5_within_T_top5": round(float(np.mean([
            set(np.argsort(p_raw[i])[-5:]) == set(np.argsort(p_T[i])[-5:])
            for i in range(len(p_T))])), 4)
    }

results["plantdoc_T_vs_raw"] = evaluate_ood("pd", pd_logits)
results["cassava_T_vs_raw"] = evaluate_ood("cv", cv_logits)

# Final report
results["best_temperature"] = round(float(best_T), 3)
results["val_nll_per_T"] = {f"T={T:.2f}": round(float(v), 4) for T, v in zip(Ts, val_nlls)}
report["results"] = results
report["elapsed_sec"] = round(time.time() - t0, 2)
report_bytes = json.dumps(report, indent=2).encode()
report["sha256_self"] = hashlib.sha256(report_bytes).hexdigest()

with open("/kaggle/working/phase14_calibration_report.json", "w") as f:
    f.write(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
