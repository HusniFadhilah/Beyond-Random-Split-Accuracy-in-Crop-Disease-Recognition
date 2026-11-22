# Kaggle running notes for crop-disease-shift research

## Verified (Phase 99 dry-run)

- **Active Kaggle accounts**: 27/27 (full list in `data/metadata/kaggle_accounts.json`)
- **Default active account** when `kaggle` CLI is run without env override: `remahankecil`
- **Push success**: `datasetresearchhusni/crop-disease-phase99-dryrun` reached `KernelWorkerStatus.COMPLETE` on v2 push
- **Log fetch**: `kaggle kernels output <kernel_id>` writes report + log to current working directory
- **Output file**: `/kaggle/working/phase99_dryrun_v2_report.json` (sha256 in header)

## Critical findings (NOT assumptions)

1. **Path prefix**: dataset sources mount at `/kaggle/input/datasets/<owner>/<slug>/...`, NOT `/kaggle/input/<slug>/...`. This is the new Kaggle convention (it concatenates multiple sources under `datasets/`).
2. **PyTorch**: Kaggle container pre-installs `torch==2.10.0+cpu` only. To use T4 GPU, must `pip install torch==2.5.0 --index-url https://download.pytorch.org/whl/cu121` in kernel.
3. **GPU compatibility**: Kaggle auto-allocates P100 (sm_60) which is INCOMPATIBLE with current PyTorch. Sm_70+ only. Default for T4 is `sm_75` (Turing) which works.
4. **PlantDoc structure**: Only contains `test/` split — usable as external field test target, NOT as training source.
5. **PlantVillage structure**: Only `color/` subset (no grayscale/segmented). 38 classes arranged `color/Class_name/UUID___FREC_Scab_NNNN.JPG`.
6. **Cassava structure**: `data/<class>/IMG.jpg`. 5 classes.

## Bug from v1 → v2

- `f.write(report_bytes)` fails because `bytes` is not `str`. Use `f.write(json.dumps(report, indent=2))` (already str), or `f.write(report_bytes.decode())`.

## Push pattern

```bash
# First push:
KAGGLE_USERNAME=datasetresearchhusni \
KAGGLE_KEY=<key> \
kaggle kernels push -p <kernel_dir>

# Subsequent pushes use the same credentials; the CLI auto-bumps version.
```

## Active kernel inventory

| Kernel | Account | Status | Log |
|---|---|---|---|
| `crop-disease-phase99-dryrun` v2 | datasetresearchhusni | COMPLETE | `data/metadata/phase99_v3_kernel.log` |

## Next steps (Phase 2 prep)

1. Patch kernel script to handle the `/kaggle/input/datasets/<owner>/<slug>/` path prefix
2. Add multi-dataset support (PlantVillage + PlantDoc + Cassava)
3. Pin `torch==2.5.0+cu121` for GPU T4 support
4. Run integrity audit on all 3 datasets
