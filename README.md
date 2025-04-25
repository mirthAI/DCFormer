# 🧠 DCFormer: Efficient 3D Vision Encoder for Medical Vision-Language Models

Official PyTorch implementation of: 
[DCFormer: Efficient 3D Vision Encoder for Medical Vision-Language Models](https://arxiv.org/abs/2502.05091)

---
## 📌 Abstract

Vision-language models (VLMs) have been widely applied to 2D medical image analysis due to their ability to align visual and textual representations. However, extending VLMs to 3D imaging remains computationally challenging. Existing 3D VLMs often rely on Vision Transformers (ViTs), which are computationally expensive due to the quadratic complexity of self-attention, or on 3D convolutions, which require large numbers of parameters and FLOPs as kernel size increases. We introduce DCFormer, an efficient 3D image encoder that factorizes 3D convolutions into three parallel 1D convolutions along the depth, height, and width dimensions. This design preserves spatial information while significantly reducing computational cost. Integrated into a CLIP-based vision-language framework, DCFormer is trained and evaluated on CT-RATE, a dataset of 50,188 paired 3D chest CT volumes and radiology reports. In zero-shot and fine-tuned detection of 18 pathologies, as well as in image–text retrieval tasks, DCFormer consistently outperforms state-of-the-art 3D vision encoders, including CT-ViT, ViT, ConvNeXt, PoolFormer, and TransUNet. These results highlight DCFormer’s potential for scalable, clinically deployable 3D medical VLMs.

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

## 📂 **[[Dataset: CT-RATE]](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE)**

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

