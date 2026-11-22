"""Phase 5 — EDA sample visualizations.
For each dataset: 1 grid PNG with 1 sample per class.
Output: /kaggle/working/eda_<dataset>.png + summary JSON
"""
import os, json, sys, time, hashlib
from pathlib import Path

t0 = time.time()
report = {
    "phase": "5_eda",
    "captured_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}

INPUT_ROOT = "/kaggle/input/datasets"
DATASETS = {
    "plantvillage": ("abdallahalidev", "plantvillage-dataset", "color"),
    "plantdoc":    ("nirmalsankalana", "plantdoc-dataset", "test"),
    "cassava":     ("nirmalsankalana", "cassava-leaf-disease-classification", "data"),
}

try:
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np
except Exception as e:
    report["error"] = f"deps missing: {e}"
    print(json.dumps(report, indent=2))
    raise

found = {}
for short, (owner, slug, sub) in DATASETS.items():
    p = os.path.join(INPUT_ROOT, owner, slug, sub)
    if os.path.isdir(p):
        found[short] = p
report["dataset_paths"] = found

THUMB = 128
COLS = 8

def make_grid(root, out_png, title):
    classes = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
    n = len(classes)
    rows = (n + COLS - 1) // COLS
    grid = Image.new("RGB", (THUMB * COLS, THUMB * rows + 30), "white")
    draw = ImageDraw.Draw(grid)
    for i, cls in enumerate(classes):
        cdir = os.path.join(root, cls)
        files = [f for f in os.listdir(cdir) if os.path.splitext(f)[1].lower() in {".jpg", ".jpeg", ".png"}]
        if not files:
            continue
        fp = os.path.join(cdir, files[0])
        try:
            img = Image.open(fp).convert("RGB")
            img.thumbnail((THUMB, THUMB))
            r, c = i // COLS, i % COLS
            grid.paste(img, (c * THUMB, r * THUMB + 30))
        except Exception:
            pass
    draw.text((4, 4), title, fill="black")
    grid.save(out_png, "PNG", optimize=True)
    return n, os.path.getsize(out_png)

os.makedirs("/kaggle/working", exist_ok=True)
for short, path in found.items():
    out_png = f"/kaggle/working/eda_{short}.png"
    n, sz = make_grid(path, out_png, f"{short} — 1 sample per class")
    report[short] = {"classes": n, "png_bytes": sz, "path": out_png}
    print(f"  {short}: {n} classes -> {out_png} ({sz} bytes)")

report["elapsed_sec"] = round(time.time() - t0, 2)
report_bytes = json.dumps(report, indent=2).encode()
report["sha256_self"] = hashlib.sha256(report_bytes).hexdigest()

with open("/kaggle/working/phase5_eda_report.json", "w") as f:
    f.write(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
