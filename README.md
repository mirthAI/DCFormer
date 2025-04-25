# 🧠 DCFormer: Efficient 3D Vision Encoder for Medical Vision-Language Models

Official PyTorch implementation of: 
[DCFormer: Efficient 3D Vision Encoder for Medical Vision-Language Models](https://arxiv.org/abs/2502.05091)

---
## 📌 Abstract

Vision-language models (VLMs) have been widely applied to 2D medical image analysis due to their ability to align visual and textual representations. However, extending VLMs to 3D imaging remains computationally challenging. Existing 3D VLMs often rely on Vision Transformers (ViTs), which are computationally expensive due to the quadratic complexity of self-attention, or on 3D convolutions, which require large numbers of parameters and FLOPs as kernel size increases. We introduce DCFormer, an efficient 3D image encoder that factorizes 3D convolutions into three parallel 1D convolutions along the depth, height, and width dimensions. This design preserves spatial information while significantly reducing computational cost. Integrated into a CLIP-based vision-language framework, DCFormer is trained and evaluated on CT-RATE, a dataset of 50,188 paired 3D chest CT volumes and radiology reports. In zero-shot and fine-tuned detection of 18 pathologies, as well as in image–text retrieval tasks, DCFormer consistently outperforms state-of-the-art 3D vision encoders, including CT-ViT, ViT, ConvNeXt, PoolFormer, and TransUNet. These results highlight DCFormer’s potential for scalable, clinically deployable 3D medical VLMs.

---

## 🏗️ Model Architecture

- ✅ **Decomposed 3D Convolutions**: 3 × 1D convs along D, H, W
- ✅ **Low computation & memory efficient**
- ✅ **Compatible with VLMs like CLIP**
- ✅ **Trained on 50k+ chest CT scans**

---

## 📊 Results Summary

Model | Variant | Params (M) | GFLOPS | Accuracy (%) | F1 Score (%) | Precision (%) | Recall (%)
DCFormer | nano | 0.92 | 34.21 | 60.4 | 41.9 | 27.2 | 62.8
ConvNeXt [¹] | nano | 3.19 | 31.92 | 62.2 | 39.4 | 26.7 | 55.1
PoolFormer [²] | nano | 2.79 | 27.14 | 60.2 | 37.0 | 24.8 | 52.3
DCFormer | naïve | 5.85 | 49.48 | 63.1 | 44.5 | 29.5 | 65.5
ViT [³] | naïve | 11.10 | 39.05 | 55.0 | 42.5 | 25.8 | 71.5
ConvNeXt [¹] | naïve | 15.63 | 96.84 | 60.7 | 42.4 | 27.7 | 63.8
PoolFormer [²] | naïve | 11.31 | 63.75 | 60.1 | 39.1 | 25.7 | 56.8
TransUNet [⁴] | naïve | 12.48 | 118.9 | 58.6 | 41.4 | 26.5 | 56.0
DCFormer | tiny | 15.1 | 168.2 | 62.0 | 46.3 | 29.7 | 70.1
ViT [³] | tiny | 26.34 | 86.43 | 61.0 | 43.2 | 28.0 | 64.8
ConvNeXt [¹] | tiny | 31.59 | 156.31 | 62.5 | 42.1 | 28.2 | 60.1
TransUNet [⁴] | tiny | 23.93 | 207.5 | 61.5 | 35.8 | 24.7 | 48.7
PoolFormer [²] | tiny | 20.68 | 117.46 | 61.8 | 38.3 | 26.0 | 53.5
CTViT [⁵] | - | 101.1 | 160.5 | 62.9 | 44.3 | 29.3 | 65.7

> 📄 Full results in the [paper](https://arxiv.org/abs/2502.05091).

---

## 📂 **[[Dataset: CT-RATE]](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE)**

- 50,188 reconstructed 3D CT volumes
- Paired with radiology reports
- 18 expert-annotated pathology labels

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/mirthAI/DCFormer.git
cd DCFormer
```
## References
The code is mainly adapted from [CT-CLIP](https://github.com/ibrahimethemhamamci/CT-CLIP/tree/main).


## Citations and Acknowledgements
The code is only for research purposes. If you have any questions regarding how to use this code, feel free to contact Gorkem Can Ates at g.canates@gmail.com.

Kindly cite the following papers if you use our code.

```bibtex
@article{ates2025dcformer,
  title={DCFormer: Efficient 3D Vision-Language Modeling with Decomposed Convolutions},
  author={Ates, Gorkem Can and Gong, Kuang and Shao, Wei},
  journal={arXiv preprint arXiv:2502.05091},
  year={2025}
}
@article{hamamci2024developing,
  title={Developing Generalist Foundation Models from a Multimodal Dataset for 3D Computed Tomography},
  author={Hamamci, Ibrahim Ethem and Er, Sezgin and Almas, Furkan and Simsek, Ayse Gulnihan and Esirgun, Sevval Nil and Dogan, Irem and Dasdelen, Muhammed Furkan and Durugol, Omer Faruk and Wittmann, Bastian and Amiranashvili, Tamaz and others},
  journal={arXiv preprint arXiv:2403.17834},
  year={2024}
}

```

