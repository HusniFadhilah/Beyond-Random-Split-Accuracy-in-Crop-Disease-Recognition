
"""Phase 99 v2 — Dry-run with fixes: no GPU needed for verification.
Verifies: dataset mount, file structure, python+deps running in Kaggle env.
"""
import os, json, sys, time, hashlib, platform

t0 = time.time()
report = {
    "phase": "99_dryrun_v2",
    "captured_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "python": sys.version.split()[0],
    "platform": platform.platform(),
}

# 1. CPU torch info (no CUDA needed)
try:
    import torch
    report["torch"] = torch.__version__
    report["cuda_available"] = torch.cuda.is_available()
except Exception as e:
    report["torch_error"] = repr(e)

# 2. Datasets mounted at /kaggle/input/?
input_dir = "/kaggle/input"
if os.path.isdir(input_dir):
    entries = sorted(os.listdir(input_dir))
    report["kaggle_input_entries"] = entries
    for e in entries:
        p = os.path.join(input_dir, e)
        if os.path.isdir(p):
            top = sorted(os.listdir(p))[:5]
            report[f"first5_{e}"] = top
else:
    report["kaggle_input"] = "MISSING"

# 3. Workspace info
report["cwd"] = os.getcwd()
report["home_contents"] = sorted(os.listdir("/kaggle"))[:10]

# 4. Try to enumerate PlantVillage class folders from /kaggle/input
pv_root = "/kaggle/input/plantvillage-dataset/color"
if os.path.isdir(pv_root):
    classes = sorted(os.listdir(pv_root))
    report["pv_class_count"] = len(classes)
    report["pv_first_classes"] = classes[:5]
    report["pv_last_classes"] = classes[-5:]
    # count images in first 2 classes
    sample_counts = {}
    for c in classes[:2]:
        cdir = os.path.join(pv_root, c)
        if os.path.isdir(cdir):
            sample_counts[c] = len(os.listdir(cdir))
    report["pv_sample_counts"] = sample_counts
else:
    report["pv_root"] = "MISSING"
    # search for it
    for root, dirs, files in os.walk("/kaggle/input"):
        if "color" in dirs:
            report["pv_root_found"] = os.path.join(root, "color")
            break

# 5. PlantDoc
pd_root = "/kaggle/input/plantdoc-dataset"
if os.path.isdir(pd_root):
    pd_top = sorted(os.listdir(pd_root))
    report["pd_top"] = pd_top
    if "test" in pd_top:
        test_classes = sorted(os.listdir(os.path.join(pd_root, "test")))
        report["pd_test_classes_count"] = len(test_classes)
        report["pd_test_first5"] = test_classes[:5]
else:
    report["pd_root"] = "MISSING"

# 6. Cassava
cv_root = "/kaggle/input/cassava-leaf-disease-classification"
if os.path.isdir(cv_root):
    cv_top = sorted(os.listdir(cv_root))
    report["cv_top"] = cv_top
    if "data" in cv_top:
        cv_classes = sorted(os.listdir(os.path.join(cv_root, "data")))
        report["cv_class_count"] = len(cv_classes)
        report["cv_first_classes"] = cv_classes

report["elapsed_sec"] = round(time.time() - t0, 2)
report_bytes = json.dumps(report, indent=2).encode()
report["sha256_self"] = hashlib.sha256(report_bytes).hexdigest()

# Output (use str, not bytes — fix from v1)
os.makedirs("/kaggle/working", exist_ok=True)
with open("/kaggle/working/phase99_dryrun_v2_report.json", "w") as f:
    f.write(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
