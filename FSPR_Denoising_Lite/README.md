# Sequential Frequency–Spatial Decoupling for High-Frequency Preserving Image Denoising (FSPR)

> **Notice.** This repository accompanies a manuscript under review / submitted to *The Visual Computer* (update as appropriate). If you use this code, please cite the paper once it is officially published (DOI will be updated here).

## Overview

This **lite** release provides:

- **Core network definition** — `AdaReNet.py` (includes the **phase-preserving frequency module M1** and the **multi-scale spatial attention block M2** in the FSPR path) and `FCNN_plus.py` (steerable / PCA convolution kernels used by the frequency-branch UNet).
- **Inference-only script** — `demo_test.py`: reads RGB images from a folder, runs a single forward pass (with reflective padding to multiples of 32), writes denoised PNGs.

It intentionally **does not** ship training loops, full data pipelines, loss implementations, or high-performance checkpoints.

## Dependencies

- Python 3.8+
- PyTorch 1.10+
- See `requirements.txt` (Pillow, NumPy, torchvision).

```bash
pip install -r requirements.txt
```

## Usage

Place your own `state_dict` checkpoint (same architecture as `AdaReNet` with matching `--ada-config`) and run:

```bash
python demo_test.py --ckpt /path/to/weights.pt --input_dir ./sample_data --output_dir ./results --cuda
```

`--ada-config` defaults to `4` (paper FSPR+ setting). Use the value that matches how the weights were trained.

## Sample data

The `sample_data/` directory contains **three small synthetic noisy crops** (no clean GT files) for pipeline checks only. Replace with your own inputs as needed.

## Statement on full code, data, and weights

Due to ongoing **institutional patent filings** and **commercial / IP restrictions** on the complete frequency–spatial training pipeline, we currently release **only** the core architectural implementation and this **inference-only** driver. **Full training scripts, complete datasets, and pre-trained weights** are not included in this archive; we plan to extend the release after the patent process and formal publication reach an appropriate stage. For urgent academic questions, please contact the corresponding author.

## Obtaining a DOI (e.g. Zenodo)

1. Create a **zip** or **GitHub release** of this `FSPR_Denoising_Lite` folder (excluding any private checkpoints you do not wish to upload).
2. Upload to [Zenodo](https://zenodo.org/) (linked GitHub integration or manual upload).
3. Zenodo issues a **version DOI** and a **concept DOI**; add the DOI badge and citation snippet to this README after registration.

## What is **not** in this lite folder

- `train.py`, `datasets.py`, Noise2Noise trainers, visualization pipelines, checkpoints, full Urban100/SIDD trees, etc. — remain in the internal / full project repository only.

## License

Add your license file (`LICENSE`) before public release. Until then, default copyright applies.
