# Beyond Random-Split Accuracy in Crop Disease Recognition

Reproducibility materials for **Beyond Random-Split Accuracy: Distribution-Shift, Calibration, and Robustness Evaluation of Crop Disease Recognition**.

This repository contains the executed experiment code, the configuration used for the main run, raw run logs, result files, figures, and selected model/prediction artifacts. The manuscript includes duplicate-leakage sensitivity analysis, unseen-corruption evaluation, a calibration rerun, and complete training-history curves.

## Repository layout

- `code/kaggle_kernels/` — Kaggle scripts and kernel metadata used for the experiments.
- `artifacts/metadata/` — downloaded JSON results and execution logs.
- `artifacts/figures/` — figures used in the manuscript.
- `artifacts/metadata/phase21_paired_output/phase21_per_sample_predictions.json` — retained per-sample prediction output where available.
- `artifacts/metadata/*.pth` — retained model checkpoints.
- `manuscript/` — manuscript source and bibliography.
- `configs/` — training configuration for the primary training-history run.
- `environment/` — dependency specification.
- `provenance/` — execution-environment information recovered from the original logs.

## Data

The PlantVillage, PlantDoc, and Cassava datasets are not redistributed. They remain available from their original providers under their respective terms. The scripts, manifests, split sizes, and result files needed to identify the data used in the experiments are included here.

## Environment record

The Phase 58 log records execution on CUDA. The original log does not record the GPU model, GPU memory, CPU model, system memory, operating system, Python version, or CUDA runtime version. Those fields are explicitly marked as not captured in `provenance/execution_environment.json`; no hardware details are inferred.

The executed installation command records PyTorch 2.3.1 and torchvision 0.18.1. See `environment/requirements.txt` and `configs/phase58_training.yaml` for the primary run settings.

## Reproduction

1. Obtain the datasets from their original providers.
2. Install `environment/requirements.txt`.
3. Review the relevant kernel metadata and configuration.
4. Run the corresponding script in an authorized Kaggle environment with the required dataset inputs.
5. Compare the generated JSON reports and figures with the retained artifacts.

The repository does not claim to reproduce provider datasets or external services. It preserves what was actually executed and downloaded from the recorded runs.
