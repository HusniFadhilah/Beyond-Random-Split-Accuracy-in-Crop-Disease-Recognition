"""Phase 9 — Corruption robustness.
Apply common image corruptions to PlantVillage test set, measure ResNet-50 degradation.
Corruptions: gaussian_noise, shot_noise, impulse_noise, defocus_blur, glass_blur,
  motion_blur, zoom_blur, snow, frost, fog, brightness, contrast, elastic_transform,
  pixelate, jpeg_compression, speckle_noise, gaussian_blur (17 types)
Severities: 1, 2, 3, 4, 5
Measure: top-1 accuracy, F1, Brier, ECE per (corruption, severity)
Plot: degradation curve.
"""
import os, json, sys, time, hashlib, random
import subprocess
print("Installing GPU torch + albumentations...")
subprocess.run(
    ["pip", "install", "--quiet", "torch==2.5.0", "torchvision==0.20.0",
     "--index-url", "https://download.pytorch.org/whl/cu121"],
    check=False, capture_output=True, text=True, timeout=300)
subprocess.run(["pip", "install", "--quiet", "albumentations==1.4.10"],
               check=False, capture_output=True, text=True, timeout=180)

import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, brier_score_loss
import cv2
from PIL import Image

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

t0 = time.time()
report = {"phase": "9_corruption_robustness", "captured_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

DATA_ROOT = "/kaggle/input/datasets/abdallahalidev/plantvillage-dataset/color"
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
print(f"train={len(tr_idx)}, test={len(te_idx)}")

train_loader = DataLoader(torch.utils.data.Subset(
    datasets.ImageFolder(DATA_ROOT, transform=train_tf), tr_idx.tolist()),
    batch_size=64, shuffle=True, num_workers=2, pin_memory=True)

# Train 2-epoch ResNet-50 (diagnostic)
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
model.fc = nn.Linear(model.fc.in_features, n_classes)
model = model.to(device)
class_counts = np.bincount(targets, minlength=n_classes)
weights = (1.0 / np.maximum(class_counts, 1)); weights = weights/weights.sum()*n_classes
class_weights = torch.FloatTensor(weights).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
for ep in range(2):
    model.train()
    t_ep = time.time()
    correct, total = 0, 0
    for batch_i, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x); loss = criterion(out, y)
        loss.backward(); optimizer.step()
        correct += (out.argmax(1) == y).sum().item(); total += x.size(0)
        if batch_i % 100 == 0:
            print(f"  ep{ep+1} batch{batch_i}/{len(train_loader)} loss={loss.item():.4f}")
    print(f"epoch {ep+1}: train acc={correct/total:.4f} ({time.time()-t_ep:.0f}s)")

# Load test images as raw PIL (no transforms), then apply corruptions
test_paths = [full.samples[i][0] for i in te_idx]
test_labels = [full.samples[i][1] for i in te_idx]
print(f"Test set: {len(test_paths)} images")

# Use a subset for tractability
MAX_TEST = 1000
if len(test_paths) > MAX_TEST:
    random.seed(123)
    sample = random.sample(range(len(test_paths)), MAX_TEST)
    test_paths = [test_paths[i] for i in sample]
    test_labels = [test_labels[i] for i in sample]
    print(f"Subsampled to {len(test_paths)} for corruption test")

# Define corruption functions (operate on numpy uint8 HxWxC, severity 1-5)
def gauss_noise(x, sev):
    s = sev * 4.0  # severity 1 = 4 std
    out = x.astype(np.float32) + np.random.normal(0, s, x.shape)
    return np.clip(out, 0, 255).astype(np.uint8)

def shot_noise(x, sev):
    rate = 60.0 / sev  # lower rate = more noise
    out = np.random.poisson(x.astype(np.float32) * rate / 255.0) / rate * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)

def impulse_noise(x, sev):
    p = sev * 0.02
    out = x.copy()
    mask_r = np.random.rand(*x.shape[:2]) < p
    out[mask_r] = np.random.choice([0, 255], mask_r.sum())
    return out

def defocus_blur(x, sev):
    k = sev * 2 + 1
    return cv2.GaussianBlur(x, (k, k), 0)

def motion_blur(x, sev):
    k = sev * 3
    kernel = np.zeros((k, k))
    kernel[k//2, :] = 1.0 / k
    return cv2.filter2D(x, -1, kernel)

def zoom_blur(x, sev):
    out = np.zeros_like(x, dtype=np.float32)
    h, w = x.shape[:2]
    for f in range(1, sev + 2):
        scale = 1.0 + 0.05 * f
        M = cv2.getRotationMatrix2D((w/2, h/2), 0, scale)
        zoomed = cv2.warpAffine(x, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        out += zoomed.astype(np.float32)
    return (out / (sev + 1)).astype(np.uint8)

def brightness(x, sev):
    factor = 1.0 - 0.15 * sev
    return np.clip(x.astype(np.float32) * factor, 0, 255).astype(np.uint8)

def contrast(x, sev):
    factor = 1.0 - 0.15 * sev
    mean = x.mean()
    out = (x.astype(np.float32) - mean) * factor + mean
    return np.clip(out, 0, 255).astype(np.uint8)

def fog(x, sev):
    # additive white noise scaled by severity
    overlay = np.ones_like(x) * 200
    alpha = 0.05 * sev
    out = cv2.addWeighted(x, 1 - alpha, overlay, alpha, 0)
    return out

def frost(x, sev):
    # darken + add slight blue tint
    alpha = 0.1 * sev
    blue = np.zeros_like(x); blue[:, :, 0] = 180; blue[:, :, 1] = 200; blue[:, :, 2] = 255
    out = cv2.addWeighted(x, 1 - alpha, blue, alpha, 0)
    return out

def snow(x, sev):
    # add white noise
    noise = np.random.normal(200, 30, x.shape).clip(0, 255).astype(np.uint8)
    alpha = 0.05 * sev
    out = cv2.addWeighted(x, 1 - alpha, noise, alpha, 0)
    return out

def pixelate(x, sev):
    h, w = x.shape[:2]
    factor = 1.0 / (1 + sev)
    new_w = max(int(w * factor), 8)
    new_h = max(int(h * factor), 8)
    small = cv2.resize(x, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

def jpeg_compress(x, sev):
    quality = max(95 - sev * 15, 5)
    enc = cv2.imencode('.jpg', x, [cv2.IMWRITE_JPEG_QUALITY, quality])[1]
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)

def gaussian_blur(x, sev):
    k = 2 * sev + 1
    return cv2.GaussianBlur(x, (k, k), 0)

def glass_blur(x, sev):
    out = x.copy()
    h, w = x.shape[:2]
    for _ in range(sev):
        dx = np.random.randint(-2, 3, (h, w))
        dy = np.random.randint(-2, 3, (h, w))
        for i in range(1, h-1):
            for j in range(1, w-1):
                ni = (i + dx[i, j]) % h
                nj = (j + dy[i, j]) % w
                out[i, j] = x[ni, nj]
        x = out
    return out

def elastic_transform(x, sev):
    h, w = x.shape[:2]
    alpha = sev * 10
    dx = cv2.GaussianBlur((np.random.rand(h, w) * 2 - 1).astype(np.float32), (15, 15), 0) * alpha
    dy = cv2.GaussianBlur((np.random.rand(h, w) * 2 - 1).astype(np.float32), (15, 15), 0) * alpha
    x_coord = np.arange(w); y_coord = np.arange(h)
    X, Y = np.meshgrid(x_coord, y_coord)
    map_x = (X + dx).astype(np.float32)
    map_y = (Y + dy).astype(np.float32)
    return cv2.remap(x, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT)

def speckle_noise(x, sev):
    s = sev * 0.05
    out = x.astype(np.float32) + x.astype(np.float32) * np.random.normal(0, s, x.shape)
    return np.clip(out, 0, 255).astype(np.uint8)

CORRUPTIONS = {
    "gauss_noise": gauss_noise,
    "shot_noise": shot_noise,
    "impulse_noise": impulse_noise,
    "defocus_blur": defocus_blur,
    "glass_blur": glass_blur,
    "motion_blur": motion_blur,
    "zoom_blur": zoom_blur,
    "snow": snow,
    "frost": frost,
    "fog": fog,
    "brightness": brightness,
    "contrast": contrast,
    "elastic_transform": elastic_transform,
    "pixelate": pixelate,
    "jpeg_compression": jpeg_compress,
    "speckle_noise": speckle_noise,
    "gaussian_blur": gaussian_blur,
}

# Load all test images once into memory (uint8 HxWxC)
print("Loading test images into memory...")
imgs = []
for fp in test_paths:
    img = cv2.imread(fp)
    if img is None:
        img = np.zeros((256, 256, 3), dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    imgs.append(img)
labels_arr = np.array(test_labels)
print(f"Loaded {len(imgs)} images, labels: {np.bincount(labels_arr, minlength=n_classes).sum()} total")

# Pre-process: normalize using ImageNet stats
def to_tensor(arr_uint8):
    """uint8 HxWxC -> normalized float32 3xHxW"""
    x = arr_uint8.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
    x = (x - mean) / std
    return torch.from_numpy(x.transpose(2, 0, 1)).float()

# Baseline (no corruption)
@torch.no_grad()
def evaluate_batch(imgs_arr, labels, model, batch_size=64):
    all_preds, all_probs = [], []
    for i in range(0, len(imgs_arr), batch_size):
        batch = imgs_arr[i:i+batch_size]
        x = torch.stack([to_tensor(im) for im in batch]).to(device)
        out = model(x)
        probs = torch.softmax(out, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_preds.append(probs.argmax(1))
    preds = np.concatenate(all_preds)
    probs = np.concatenate(all_probs)
    acc = float((preds == labels).mean())
    f1 = f1_score(labels, preds, average="macro", zero_division=0)
    return acc, f1, probs

print("\n=== Baseline (no corruption) ===")
base_acc, base_f1, base_probs = evaluate_batch(imgs, labels_arr, model)
print(f"baseline acc={base_acc:.4f}, f1={base_f1:.4f}")
base_max_conf = base_probs.max(1).mean()
print(f"baseline max_conf mean: {base_max_conf:.4f}")

# Corruption sweep
results = {"baseline": {"acc": round(base_acc, 4), "f1_macro": round(base_f1, 4),
                        "max_conf_mean": round(float(base_max_conf), 4), "n": len(imgs)},
           "corruptions": {}}

for cname, cfn in CORRUPTIONS.items():
    print(f"\n--- corruption: {cname} ---")
    sev_results = []
    for sev in [1, 2, 3, 4, 5]:
        t0c = time.time()
        corrupted = [cfn(im.copy(), sev) for im in imgs]
        acc, f1, probs = evaluate_batch(corrupted, labels_arr, model)
        max_conf = probs.max(1).mean()
        sev_results.append({
            "severity": sev,
            "acc": round(acc, 4),
            "f1_macro": round(f1, 4),
            "max_conf_mean": round(float(max_conf), 4),
            "elapsed_sec": round(time.time() - t0c, 2)
        })
        print(f"  sev={sev}: acc={acc:.4f}, f1={f1:.4f}, conf={max_conf:.4f}")
    # mean over severities
    mean_acc = float(np.mean([s["acc"] for s in sev_results]))
    mean_f1 = float(np.mean([s["f1_macro"] for s in sev_results]))
    results["corruptions"][cname] = {
        "per_severity": sev_results,
        "mean_acc": round(mean_acc, 4),
        "mean_f1_macro": round(mean_f1, 4),
    }
    print(f"  → mean acc: {mean_acc:.4f}, mean f1: {mean_f1:.4f}")

# Summary: rank by mean_acc
ranked = sorted(results["corruptions"].items(), key=lambda x: x[1]["mean_acc"])
print("\n=== Corruption robustness ranking (worst to best) ===")
for name, data in ranked:
    print(f"  {name:25s} mean_acc={data['mean_acc']:.4f}, mean_f1={data['mean_f1_macro']:.4f}")

report["results"] = results
report["n_test_sampled"] = len(imgs)
report["n_corruptions"] = len(CORRUPTIONS)
report["n_severities"] = 5
report["elapsed_sec"] = round(time.time() - t0, 2)
report_bytes = json.dumps(report, indent=2).encode()
report["sha256_self"] = hashlib.sha256(report_bytes).hexdigest()

with open("/kaggle/working/phase9_corruption_report.json", "w") as f:
    f.write(json.dumps(report, indent=2))
print("\nWROTE /kaggle/working/phase9_corruption_report.json")
