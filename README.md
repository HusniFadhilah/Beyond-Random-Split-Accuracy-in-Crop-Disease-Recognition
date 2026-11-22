# Beyond Random-Split Accuracy in Crop Disease Recognition

Reproducibility release for **Beyond Random-Split Accuracy: Distribution-Shift, Calibration, and Robustness Evaluation of Crop Disease Recognition**.

## Contents

- `code/kaggle_kernels/`: original executed Kaggle kernel sources and metadata for the experiment phases.
- `artifacts/metadata/`: downloaded reports and raw execution logs.
- `artifacts/figures/`: generated figures used by the manuscript.
- `manuscript/`: manuscript source and expanded bibliography.
- `configs/`: training configuration used for the Phase 58 primary training-history run.
- `environment/`: dependency specification.
- `provenance/execution_environment.json`: environment fields recovered from the original logs.

## Data

The original PlantVillage, PlantDoc, and Cassava datasets are not redistributed. The reports and scripts identify the source datasets and expected split sizes. Dataset manifests and derived artifacts are included where available. Provider terms remain applicable.

## Prediction outputs and model checkpoints

Prediction arrays and model checkpoints were not present in the downloaded original artifacts and are therefore not fabricated or represented as deposited. The retained reports contain aggregate metrics and diagnostic summaries. Where prediction outputs are permissible and available from the original run, they can be added under `artifacts/predictions/` with provenance and hashes.

## Execution environment boundary

Phase 58's original log explicitly recorded `device: cuda`, but did not print the GPU model, GPU memory, CPU model, system memory, operating system, Python version, or CUDA runtime version. These fields are marked `NOT_CAPTURED_IN_ORIGINAL_LOG` in `provenance/execution_environment.json`. The executed install command recorded `torch==2.3.1` and `torchvision==0.18.1`; this is not a substitute for a full runtime inventory.

## Reproduction

1. Obtain the datasets from their original providers.
2. Install `environment/requirements.txt`.
3. Review `configs/phase58_training.yaml`.
4. Run the relevant kernel source from `code/kaggle_kernels/` in an authorized Kaggle environment with the corresponding dataset inputs.
5. Compare generated reports with the retained reports using the recorded split sizes, protocol, and metrics.

## License

No repository license is asserted by this release. Dataset and third-party code terms remain applicable.
