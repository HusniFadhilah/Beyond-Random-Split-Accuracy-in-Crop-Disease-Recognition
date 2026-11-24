# Beyond Random-Split Accuracy in Crop Disease Recognition

Reproducibility materials for **Beyond Random-Split Accuracy: Distribution-Shift, Calibration, and Robustness Evaluation of Crop Disease Recognition**.

This repository contains the executed experiment code, the configuration used for the main run, raw run logs, result files, figures, and selected model/prediction artifacts. The manuscript includes duplicate-leakage sensitivity analysis, unseen-corruption evaluation, a calibration rerun, and complete training-history curves.

## Repository layout

- `code/kaggle_kernels/` — Kaggle scripts and kernel metadata used for the experiments.
- `artifacts/metadata/` — downloaded JSON results and execution logs.
- `artifacts/figures/` — figures used in the manuscript.
- `artifacts/metadata/phase21_paired_output/phase21_per_sample_predictions.json` — retained per-sample prediction output where available.
- `artifacts/metadata/*.pth` — retained model checkpoints.
- `manuscript/` — manuscript source and bibliography for the earlier preflight source.
- `manuscript/jbcs/` — the latest JBCS manuscript source, compiled PDF, bibliography, and figures.
- `response/` — the current point-by-point response to Reviewers B and E.
- `configs/` — training configuration for the primary training-history run.
- `environment/` — dependency specification.
- `provenance/` — execution-environment information, including the distinction between archived run logs and separate system inspection.

## Data

The PlantVillage, PlantDoc, and Cassava datasets are not redistributed. They remain available from their original providers under their respective terms. The scripts, manifests, split sizes, and result files needed to identify the data used in the experiments are included here.

## Environment record

The archived Phase 58 log records execution on CUDA and the installation command records PyTorch 2.3.1 and torchvision 0.18.1. A separate end-to-end inference benchmark was completed on a current Kaggle Tesla T4 runtime with PyTorch 2.11.0+cu130; its latency and peak-memory results are retained in `artifacts/metadata/end_to_end_gpu_inference_report.json`. These values are not time-matched to Phase 58 and must not be used to reconstruct its historical hardware. A separate author-supplied environment inspection reports an Intel Xeon CPU, NVIDIA Tesla P100 with 16 GB VRAM, driver 560.35.03, and approximately 24.6 GB system memory. These hardware values also are not independently time-matched to the archived Phase 58 log. Operating-system, Python, and original PyTorch CUDA-runtime versions were not established from the available records. See `provenance/COMPUTE_ENVIRONMENT_PROVENANCE.md`.

## Reproduction

1. Obtain the datasets from their original providers.
2. Install `environment/requirements.txt`.
3. Review the relevant kernel metadata and configuration.
4. Run the corresponding script in an authorized Kaggle environment with the required dataset inputs.
5. Compare the generated JSON reports and figures with the retained artifacts.

The repository does not claim to reproduce provider datasets or external services. It preserves what was actually executed and downloaded from the recorded runs.
