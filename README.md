# 🧠 DCFormer: Efficient 3D Vision Encoder for Medical Vision-Language Models

**[[arXiv Paper]](https://arxiv.org/abs/2502.05091)** | **[[Project Page]](https://mirthai.github.io/DCFormer)** | **[[Dataset: CT-RATE]](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE)**

---
## 📌 Overview

**DCFormer** is an efficient 3D vision encoder designed to scale Vision-Language Models (VLMs) to high-resolution volumetric data. Unlike computationally expensive ViTs or heavy 3D CNNs, DCFormer factorizes 3D convolutions into **three parallel 1D convolutions** along depth, height, and width — reducing FLOPs and parameter count while preserving spatial context.

> 💡 Integrated into a CLIP-based vision-language framework, DCFormer achieves **state-of-the-art performance** in zero-shot and fine-tuned pathology detection and image-text retrieval on the large-scale **CT-RATE** dataset.

---

## 🏗️ Model Architecture

- ✅ **Factorized 3D Convolutions**: 3 × 1D convs along D, H, W
- ✅ **Low compute & memory footprint**
- ✅ **Compatible with VLMs like CLIP**
- ✅ **Trained on 50k+ chest CT scans**

---

## 📊 Results Summary

| Task                       | Metric    | DCFormer | CT-ViT | ConvNeXt | TransUNet |
|----------------------------|-----------|----------|--------|-----------|------------|
| Zero-shot Pathology Det.  | AUC       | **0.843**| 0.804  | 0.792     | 0.785      |
| Image-Text Retrieval       | Recall@1  | **38.7** | 32.4   | 28.6      | 29.8       |

> 📄 Full results and ablations in the [paper](https://arxiv.org/abs/2502.05091).

---

## 📂 Dataset: CT-RATE

- 50,188 reconstructed 3D CT volumes
- Paired with radiology reports
- 18 expert-annotated pathology labels
- Available upon request or through institutional agreement

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/mirthAI/DCFormer.git
cd DCFormer
