# Automatic Detection of Hyperreflective Points in OCT Images
BELAGHIT Tassnim Alla · Dec 2025 – Jan 2026

## Overview

This project develops a deep learning pipeline for the **automatic detection
and counting of hyperreflective points (HRPs)** in retinal OCT B-scans.
HRPs are small bright structures (~15–25 px) associated with AMD and DME
progression. Their manual counting is impractical in clinical workflows,
motivating a fully automated approach.

---

## Architectures Explored

| Model | Task | Status |
|---|---|---|
| YOLOv8 | Object detection | ❌ Rejected (grid crowding, domain gap) |
| Faster R-CNN v1 | Object detection | ⚠️ Baseline |
| Faster R-CNN v2 | Object detection | ⚠️ Improved |
| U-Net v1 (ResNet-34) | Segmentation | ⚠️ Baseline |
| **U-Net v2 (ResNet-34 + scSE)** | **Segmentation** | ✅ **Final model** |

---

## Final Model

- **Architecture:** U-Net with ResNet-34 encoder + scSE decoder attention
- **Loss:** Tversky-Focal (α=0.3, β=0.7)
- **Training:** Two-stage (encoder frozen for 5 epochs, then full fine-tune)
- **Augmentation:** Flips, rotations, brightness/contrast jitter, Gaussian noise/blur, elastic deformation
- **Post-processing:** Morphological close+open → connected components → area + circularity filter
- **Inference:** Test-Time Augmentation (8 geometric transforms)
- **Best val Dice:** `0.3646` @ epoch 29

---

## Repository Structure
```
.
├── notebooks/
│   ├── faster_rcnn_v1.ipynb       # Baseline Faster R-CNN
│   ├── faster_rcnn_v2.ipynb       # Improved Faster R-CNN
│   ├── unet_v1.ipynb              # Baseline U-Net
│   └── u-net__1_.ipynb            # Final Kaggle-ready U-Net
├── report/
│   └── main_fixed.tex             # Full technical report (LaTeX)
├── .gitignore
└── README.md
```

---

## Dataset

Data was collected under a clinical study using a
Spectralis SD-OCT system. Annotations are in Pascal VOC XML format.

| Split | Images |
|---|---|
| Train | 2,781 |
| Val | 251 |
| Test | 46 |

>  The dataset contains medical imaging data and is **not included**
> in this repository.

---

## Installation
```bash
# Core deep learning
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Segmentation models
pip install segmentation-models-pytorch timm

# Image processing
pip install opencv-python-headless pillow scikit-image scipy

# Utilities
pip install matplotlib numpy scikit-learn pandas
```

---

## Quick Inference
```python
import torch
import segmentation_models_pytorch as smp

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = smp.Unet(
    encoder_name           = 'resnet34',
    encoder_weights        = None,
    in_channels            = 3,
    classes                = 1,
    decoder_attention_type = 'scse',
).to(device)

checkpoint = torch.load('unet_best.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

Then call `predict_dots(model, pil_image)` — see the notebook for the full
inference and TTA pipeline.

---
