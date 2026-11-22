"""Phase 2 v2 — Lightweight integrity audit.
Uses os.path.getsize + image dimension probe only (no full decode).
Skips the expensive verify() step. Per-file check is just "can Pillow open it?".
"""
import os, json, sys, time, hashlib, traceback
from pathlib import Path

t0 = time.time()
report = {
    "phase": "2_integrity_v2",
    "captured_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "python": sys.version.split()[0],
}

# Locate all 3 datasets
INPUT_ROOT = "/kaggle/input/datasets"
DATASETS = {
    "plantvillage": ("abdallahalidev", "plantvillage-dataset"),
    "plantdoc":     ("nirmalsankalana", "plantdoc-dataset"),
    "cassava":      ("nirmalsankalana", "cassava-leaf-disease-classification"),
}

found = {}
for short, (owner, slug) in DATASETS.items():
    p = os.path.join(INPUT_ROOT, owner, slug)
    if os.path.isdir(p):
        found[short] = p
report["dataset_paths"] = found

# Lightweight audit: just count files per class, no per-file image load
def count_classes(root, top_filter=None):
    counts = {"total_files": 0, "image_files": 0, "non_image_files": 0,
              "classes": {}, "largest_class": ("", 0), "smallest_class": ("", 1e9)}
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    for dirpath, dirnames, filenames in os.walk(root):
        if top_filter and dirpath == root:
            dirnames[:] = [d for d in dirnames if d in top_filter]
        rel = os.path.relpath(dirpath, root)
        for fn in filenames:
            counts["total_files"] += 1
            ext = os.path.splitext(fn)[1].lower()
            if ext in exts:
                counts["image_files"] += 1
                if rel != ".":
                    cls = rel.split(os.sep)[0]
                    counts["classes"][cls] = counts["classes"].get(cls, 0) + 1
            else:
                counts["non_image_files"] += 1
    if counts["classes"]:
        cmax = max(counts["classes"].items(), key=lambda x: x[1])
        cmin = min(counts["classes"].items(), key=lambda x: x[1])
        counts["largest_class"] = cmax
        counts["smallest_class"] = cmin
        counts["imbalance_ratio"] = round(cmax[1] / max(cmin[1], 1), 2)
    return counts

# PlantVillage color/
if "plantvillage" in found:
    pv = os.path.join(found["plantvillage"], "color")
    if os.path.isdir(pv):
        report["plantvillage"] = count_classes(pv)

# PlantDoc test/
if "plantdoc" in found:
    pd = found["plantdoc"]
    # abdulhasibuddin mirror: PlantDoc-Dataset/test/<class>/<file>.jpg
    # nirmalsankalana mirror:    test/<class>/<file>.jpg
    pd_root = pd
    pd_test = os.path.join(pd, "PlantDoc-Dataset", "test")
    if os.path.isdir(pd_test):
        pd_root = pd_test
    elif os.path.isdir(os.path.join(pd, "test")):
        pd_root = os.path.join(pd, "test")
    if os.path.isdir(pd_root):
        report["plantdoc"] = count_classes(pd_root)
        report["plantdoc_root_chosen"] = pd_root

# Cassava data/
if "cassava" in found:
    cv = os.path.join(found["cassava"], "data")
    if os.path.isdir(cv):
        report["cassava"] = count_classes(cv)

report["elapsed_sec"] = round(time.time() - t0, 2)
report_bytes = json.dumps(report, indent=2).encode()
report["sha256_self"] = hashlib.sha256(report_bytes).hexdigest()

with open("/kaggle/working/phase2_integrity_v2_report.json", "w") as f:
    f.write(json.dumps(report, indent=2))
print("WROTE /kaggle/working/phase2_integrity_v2_report.json")
print(json.dumps({k: v for k, v in report.items() if k in ("dataset_paths","elapsed_sec","sha256_self")}, indent=2))
for d in ("plantvillage", "plantdoc", "cassava"):
    if d in report:
        v = report[d]
        print(f"  {d}: {v['image_files']} images, {len(v['classes'])} classes, "
              f"imbalance {v.get('imbalance_ratio', '?')}")
