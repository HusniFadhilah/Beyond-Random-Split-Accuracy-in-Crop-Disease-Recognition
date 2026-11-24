# Computational environment evidence and provenance

This file distinguishes independently archived execution records from system-inspection text provided in the conversation. It is not a replacement for a complete, timestamped runtime capture.

## Confirmed in the public GitHub repository
- The Phase 58 primary log prints `device: cuda`.
- The executed kernel explicitly requires a CUDA device and uses EfficientNet-B0.
- The kernel installs `torch==2.3.1` and `torchvision==0.18.1`. These version values come from the installation command.
- The kernel code contains `batch_size=128`, seed 42, AdamW, a CosineAnnealingLR scheduler, 224-pixel inputs, six epochs, label smoothing 0.1, a learning rate of 0.0003, and weight decay of 0.0001.
- The primary split is 43,444 train, 5,430 validation, and 5,431 test images.

## Additional author-supplied system-inspection details
- Execution platform: Kaggle Kernel
- CPU: Intel(R) Xeon(R) CPU @ 2.00GHz, reported as 4 logical CPU(s) and 2 cores
- GPU: 1 x NVIDIA Tesla P100-PCIE, 16 GB VRAM
- NVIDIA driver: 560.35.03
- CUDA reported by `nvidia-smi`: 12.6 (driver-supported version, not a PyTorch runtime measurement)
- Memory reported from `/proc/meminfo`: 24,605,328 kB (about 24.6 GB decimal)
- User provided disk context: capacity ~1.8 TB, per-account allowance ~214.75 GB, per-session usable ~20.5 GB. These values were not needed in the manuscript because they do not establish memory/storage footprint of the experiment.

## Separate current-runtime inference benchmark
- Execution platform: Kaggle Kernel
- GPU: NVIDIA Tesla T4
- PyTorch: 2.11.0+cu130
- PyTorch-reported CUDA: 13.0
- cuDNN: 92700
- Protocol: random 224x224 RGB inputs, batch size 32, 20 warm-up iterations, and 100 CUDA-event timed iterations
- Report: `artifacts/metadata/end_to_end_gpu_inference_report.json`

This benchmark is a separate current-runtime measurement. It is not time-matched to the archived Phase 58 training run and does not identify the historical Phase 58 GPU.

The original public run log does NOT identify the specific Tesla P100, CPU, memory, driver or `nvidia-smi` version. The separate environment-inspection text is included as supplied by the authors and is not independently time-matched to the archived run. The system-inspection excerpt contains no independently established operating system, Python version, or PyTorch CUDA runtime version.

No claim is made that the GPU has 32 CUDA cores: this ambiguous value in the supplied notes is not used.
