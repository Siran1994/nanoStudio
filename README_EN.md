<div align="center">

# ⚡ nanoStudio

### An Interactive, Zero-Dependency Multi-Modal AI Studio in Pure PyTorch
**Train, Monitor & Play with Custom LLMs and Vision Models in Your Browser**

[![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue.svg?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Hardware Auto-detect](https://img.shields.io/badge/Hardware-CPU%20%2F%20CUDA%20Auto-brightgreen.svg)](#)

[English](./README_EN.md) · [简体中文](./README.md) · [Original nanoGPT](./README_ORIGINAL.md)

</div>

---

## 📖 Overview

**nanoStudio** is an all-in-one, beginner-friendly deep learning laboratory and interactive web cockpit built on pure PyTorch.

Originating from Andrej Karpathy's renowned [nanoGPT](https://github.com/karpathy/nanoGPT), **nanoStudio** upgrades the architecture into an extensible, multi-modal playground:
* **Train Custom LLMs from scratch**: Support character-level GPT models with both English (Shakespeare) and Chinese literature (*YuYe TanJing*).
* **Computer Vision & Image Classification**: Built-in `SimpleCNN` and industrial `ResNet-18` on CIFAR-10 and custom image datasets.
* **Real-time Dual-Metric Monitoring**: Canvas charts plotting live Loss and Top-1 Accuracy % curves side-by-side.
* **Interactive Playgrounds**: Auto-regressive text sampling and Drag-and-Drop image prediction with Top-5 probability bars.
* **Hardware Adaptive**: Runs out-of-the-box on standard laptop CPUs; automatically enables CUDA acceleration when a GPU is detected.

---

## ⚡ Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/your-username/nanoStudio.git
cd nanoStudio

# 2. Install minimal dependencies
pip install torch pillow numpy

# 3. Launch the studio
python server.py
```

Open your browser at:
👉 **`http://127.0.0.1:8000`**

---

## ✨ Key Highlights

| Feature | Description |
| :--- | :--- |
| **📝 LLM Studio** | 6 scaling tiers (0.8M Baby to 124M GPT-2), temperature/top-k sliders, custom Chinese & English datasets |
| **👁️ Vision Studio** | SimpleCNN & ResNet-18, dual-axis Loss & Accuracy % monitoring, drag-and-drop instant image classifier |
| **🗂️ Dataset Asset Manager** | Full CRUD asset control for text and vision datasets, supporting folder-per-class custom structures |
| **🎨 Diffusion Ready** | Extensible architecture with reserved slots for DDPM/GAN generative models |
| **💻 CPU & GPU Friendly** | Zero heavy dependencies (no external torchvision required), cross-platform compatibility |

---

## 📜 License & Credits

* Licensed under the [MIT License](LICENSE).
* Special thanks to **[Andrej Karpathy](https://github.com/karpathy)** for [nanoGPT](https://github.com/karpathy/nanoGPT).
