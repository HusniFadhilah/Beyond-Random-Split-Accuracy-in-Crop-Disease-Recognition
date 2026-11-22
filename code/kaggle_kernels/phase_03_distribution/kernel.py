"""Phase 3 — Distribution audit.
For each dataset x class: count, color stats (R/G/B mean+std), image dim, edge density.
Output JSON + CSV-style summary.
"""
import os, json, sys, time, hashlib
import struct

t0 = time.time()
report = {
    "phase": "3_distribution",
    "captured_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "python": sys.version.split()[0],
}

INPUT_ROOT = "/kaggle/input/datasets"
DATASETS = {
    "plantvillage": ("abdallahalidev", "plantvillage-dataset", "color"),
    "plantdoc":    ("nirmalsankalana", "plantdoc-dataset", "test"),
    "cassava":     ("nirmalsankalana", "cassava-leaf-disease-classification", "data"),
}

found = {}
for short, (owner, slug, sub) in DATASETS.items():
    p = os.path.join(INPUT_ROOT, owner, slug, sub)
    if os.path.isdir(p):
        found[short] = p
report["dataset_paths"] = found

def per_class_stats(root, sample_per_class=50):
    """For each subdir (class), sample N images and compute stats.
    Returns dict[class] -> {count, sampled, mean_rgb, std_rgb, dim_median, edge_density_median}.
    """
    out = {}
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    try:
        from PIL import Image
        import numpy as np
    except Exception as e:
        return {"error": f"PIL/numpy not available: {e}"}
    for cls in sorted(os.listdir(root)):
        cdir = os.path.join(root, cls)
        if not os.path.isdir(cdir):
            continue
        files = [f for f in os.listdir(cdir) if os.path.splitext(f)[1].lower() in exts]
        # sample
        if len(files) > sample_per_class:
            import random
            random.seed(42)
            files = random.sample(files, sample_per_class)
        r_means, g_means, b_means = [], [], []
        widths, heights = [], []
        edge_densities = []
        for fn in files:
            full = os.path.join(cdir, fn)
            try:
                img = Image.open(full).convert("RGB")
                w, h = img.size
                widths.append(w); heights.append(h)
                arr = np.asarray(img, dtype=np.float32) / 255.0
                r_means.append(arr[:,:,0].mean())
                g_means.append(arr[:,:,1].mean())
                b_means.append(arr[:,:,2].mean())
                # edge density (Sobel proxy: gradient magnitude mean)
                gx = np.abs(np.diff(arr, axis=1)).mean()
                gy = np.abs(np.diff(arr, axis=0)).mean()
                edge_densities.append((gx + gy) / 2.0)
            except Exception:
                continue
        if not r_means:
            continue
        out[cls] = {
            "count_total": len(os.listdir(cdir)),
            "sampled": len(r_means),
            "mean_R": round(float(sum(r_means)/len(r_means)), 4),
            "mean_G": round(float(sum(g_means)/len(g_means)), 4),
            "mean_B": round(float(sum(b_means)/len(b_means)), 4),
            "median_width": int(sorted(widths)[len(widths)//2]),
            "median_height": int(sorted(heights)[len(heights)//2]),
            "edge_density_median": round(float(sorted(edge_densities)[len(edge_densities)//2]), 4),
        }
    return out

for short, path in found.items():
    report[short] = per_class_stats(path)

# Cross-dataset aggregate
for short in found:
    if short in report and isinstance(report[short], dict) and "error" not in report[short]:
        classes = report[short]
        if classes:
            all_r = [c["mean_R"] for c in classes.values()]
            all_g = [c["mean_G"] for c in classes.values()]
            all_b = [c["mean_B"] for c in classes.values()]
            all_e = [c["edge_density_median"] for c in classes.values()]
            report[f"{short}_aggregate"] = {
                "n_classes": len(classes),
                "mean_R_across_classes": round(float(sum(all_r)/len(all_r)), 4),
                "mean_G_across_classes": round(float(sum(all_g)/len(all_g)), 4),
                "mean_B_across_classes": round(float(sum(all_b)/len(all_b)), 4),
                "edge_density_median": round(float(sorted(all_e)[len(all_e)//2]), 4),
            }

report["elapsed_sec"] = round(time.time() - t0, 2)
report_bytes = json.dumps(report, indent=2).encode()
report["sha256_self"] = hashlib.sha256(report_bytes).hexdigest()

with open("/kaggle/working/phase3_distribution_report.json", "w") as f:
    f.write(json.dumps(report, indent=2))
print("WROTE /kaggle/working/phase3_distribution_report.json")
print(json.dumps({k: v for k, v in report.items()
                  if k in ("dataset_paths","elapsed_sec","sha256_self") or k.endswith("_aggregate")}, indent=2))
