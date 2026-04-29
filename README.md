# MultiDentNetV3

**A unified deep learning framework for multi-class dental condition screening and preliminary risk stratification of cancer-suspicious oral lesions**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Aliyar4061/MultiDentNetV3/blob/main/colab_setup.ipynb)

---

## 📝 Abstract

**Objectives:** To develop and evaluate MultiDentNet, a unified deep learning framework for multi-class dental condition screening and preliminary risk stratification of cancer-suspicious oral lesions, utilizing backbone-diverse ensembling and inter-class relational modeling.

**Methods:** Four complementary CNN backbones (DenseNet-121, EfficientNetV2-S, ResNet-50, Inception-V3) integrated with Squeeze-and-Excitation (SE) attention, graph convolutional networks (GCNs), and multi-task learning were fused via validation-optimized weighting. The framework was evaluated on 10,235 clinical intraoral images (five conditions) and 940 clinically labeled oral lesion images (cancer-suspicious vs. non-cancer-suspicious; histopathological confirmation unavailable).

**Results:**  
- **Dental Classification:** 99.70% accuracy (95% CI: 99.32–100.00%), κ = 0.996, MCC = 0.996  
- **Oral Lesion Risk Stratification:** 95.71% accuracy (95% CI: 92.20–98.58%), κ = 0.913, MCC = 0.914  
- **Cancer-suspicious Recall:** 97.31% (FNR: 2.69%, 95% CI: 0.00–7.15%)  
- **Class Imbalance Mitigation:** Reduced Hypodontia FNR to 1.65% (95% CI: 0.00–4.24%)

---

## 🚀 Key Features

- ✅ **Backbone Ensemble** – Fuses 4 architectures with SE attention to suppress backbone-specific variance  
- ✅ **GCN Relational Modeling** – Captures inter-class dependencies via learnable adjacency matrices  
- ✅ **Multi-Task Learning** – Simultaneous dental screening and auxiliary lesion risk assessment  
- ✅ **Statistical Rigor** – 1000-iteration bootstrapping for robust 95% CIs and Label Smoothing (0.1)  
- ✅ **XAI Integration** – Clinical transparency through Grad-CAM and affinity visualizations  

---

## 📊 Dataset Summary

| Task | Samples | Classes / Conditions |
|------|---------|----------------------|
| Dental Conditions | 10,235 | 5 (Caries, Gingivitis, Discoloration, Ulcers, Hypodontia) |
| Oral Lesion Risk | 940 | 2 (Cancer-Suspicious vs. Non-Cancer-Suspicious) |

---

## 📈 Performance Highlights

| Metric | Dental Classification | Lesion Risk Stratification |
|--------|----------------------|----------------------------|
| Accuracy | 99.70% | 95.71% |
| Cohen's κ | 0.996 | 0.913 |
| MCC | 0.996 | 0.914 |
| FNR (Minority/Critical) | 1.65% (Hypodontia) | 2.69% (Suspicious) |

---

## 🛠 Implementation Details

All experiments are implemented in **PyTorch 2.0+** with the `timm` library for pretrained backbones. Training employs batch size 128 (adjusted for GPU memory constraints), seed 42 for reproducibility, and deterministic algorithms where supported. Data loading utilizes pinned memory and two worker processes for efficient I/O. The complete source code, trained weights, and evaluation scripts are publicly available at [https://github.com/Aliyar4061/MultiDentNetV3/tree/main](https://github.com/Aliyar4061/MultiDentNetV3/tree/main) to ensure full reproducibility and community adoption.

### Training Hyperparameters & Software Stack

| Component | Specification |
|-----------|---------------|
| **Framework** | PyTorch 2.0+, torchvision 0.15+ |
| **Backbones** | DenseNet-121, EfficientNetV2-S, ResNet-50, Inception-V3 (via `timm`) |
| **Input Size** | 224×224 pixels |
| **Batch Size** | 128 (reduced to 64 or 32 for smaller GPUs) |
| **Optimizer** | AdamW (lr = 5e-4, weight decay = 1e-4) |
| **Scheduler** | CosineAnnealingLR (T_max = epochs) |
| **Loss Function** | Focal Loss (γ = 2.0) + Label Smoothing (ε = 0.1) |
| **Training Epochs** | 12 (early stopping patience = 3) |
| **Data Augmentation** | Horizontal/Vertical flip, rotation (±20°), color jitter, CoarseDropout, CLAHE, elastic transform |
| **Test Time Augmentation** | 5 steps (flips, rotation) |
| **Ensemble Weighting** | Validation-accuracy-based exponential weighting (temperature τ = 10) |
| **Bootstrapping** | 1000 iterations, 95% confidence intervals |
| **Hardware** | NVIDIA RTX 3060 (12GB) / RTX 3090 / A100 |
| **CUDA Version** | 11.7 or 11.8 |
| **Reproducibility** | Random seed 42, deterministic CuDNN, `torch.use_deterministic_algorithms(True)` |

---

## 💻 Environment & Hardware

### System Requirements
- **OS:** Linux (Ubuntu 20.04+), Windows 10/11, or macOS (CPU only)  
- **Python:** 3.8 – 3.10  
- **CUDA:** 11.7 or 11.8 (for GPU training)  
- **GPU:** NVIDIA RTX 3060 (12GB) minimum; RTX 3090/A100 recommended for batch size 128  
- **RAM:** 32 GB+ recommended  

### Software Dependencies
All required packages are listed in `requirements.txt`. Core libraries:
- `torch>=2.0.0`, `torchvision>=0.15.0`
- `timm>=0.9.0` (ImageNet pretrained backbones)
- `albumentations>=1.3.0` (augmentations)
- `scikit-learn`, `pandas`, `numpy`
- `matplotlib`, `seaborn`, `tqdm`
- `opencv-python-headless`, `pillow`
- `thop` (optional, for FLOPs calculation)

---
## 🔧 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/Aliyar4061/MultiDentNet.git
cd MultiDentNet
pip install -r requirements.txt




dataset_cleaned/
├── train/
│   ├── Data caries/
│   ├── Gingivitis/
│   ├── hypodontia/
│   ├── Tooth Discoloration/
│   └── Mouth Ulcer/
├── val/
│   └── (same classes)
└── test/
    └── (same classes)


MultiDentNet/
├── colab_training.py          # Complete training & evaluation script (Colab/local)
├── colab_setup.ipynb          # Step‑by‑step Colab notebook
├── train.py                   # Minimal training example
├── evaluate.py                # Bootstrapped evaluation script
├── requirements.txt           # Python dependencies
├── .gitignore
├── LICENSE (MIT)
├── configs/
│   └── dental_config.yaml     # Hyperparameters (image size, batch, LR, etc.)
├── models/                    # Custom modules (SEBlock, GCN, CustomModel)
├── preprocessing/             # Data loaders, transforms
├── notebooks/                 # Demo notebook (same as colab_setup)
├── docs/                      # Placeholder for generated figures
└── weights/                   # Saved model checkpoints


@article{abdian2026multidentnet,
  title={MultiDentNet: A Unified Deep Learning Framework for Multi-Class Dental Condition Screening and Preliminary Oral Lesion Triage},
  author={Abdian, Ali Zeydi and Javidi, Mohammad Masoud and Mansouri, Najme and Mehranfar, Farzaneh and Cheperli, Sahar},
  journal={Scientific Reports},
  year={2026},
  publisher={Nature Publishing Group}
}
