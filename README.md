<div align="center">

# ⚡ nanoStudio

### 极简通用多模态深度学习实验台 · LLM / Vision / Diffusion
**An Interactive, Lightweight Multi-Modal Deep Learning Studio in Pure PyTorch**

[![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue.svg?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Zero Setup](https://img.shields.io/badge/Hardware-CPU%20%2F%20CUDA%20Auto-brightgreen.svg)](#-硬件支持)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey.svg)](#)

[简体中文](./README.md) · [English](./README_EN.md) · [原版 nanoGPT](./README_ORIGINAL.md)

</div>

---

## 📖 项目简介 (Introduction)

**nanoStudio** 是一个专为深度学习初学者、教育工作者和 AI 爱好者打造的**全栈、交互式多模态实验工作台**。

项目源自 Andrej Karpathy 著名的开源项目 [nanoGPT](https://github.com/karpathy/nanoGPT)，并在其基础上进行了**全架构的多模态升级与产品化重构**：
* 无论是 **从零训练属于自己的中文语言模型（LLM）**；
* 还是 **训练卷积神经网络识别各类图片（Vision）**；
* 或是 **查看实时 Loss 与准确率双指标收敛轨迹、拖拽图片即时预测、自由增删数据集**；

用户都无需在黑黢黢的终端里痛苦配置，只需在浏览器中打开优雅现代的 **Dark Glassmorphism** 控制台，即可沉浸式探索深度学习的奥秘。**普通笔记本 CPU 即可秒级训练，有独立显卡自动起飞！**

---

## ✨ 核心特性 (Features)

### 1. 📝 语言大模型工坊 (LLM Studio)
* **自定义语料训练**：支持从零训练字符级 Transformer，开箱自带 **🇨🇳 中文书籍《雨夜谈经》** 与 **🇬🇧 莎士比亚戏剧** 双模态语料。
* **6 级参数规模一键切换**：从 `0.8M Baby`（CPU 几秒见效）到 `124M 完整 GPT-2`，涵盖层数、头数、通道维度动态整除自检。
* **自回归对话与生成测试台 (Playground)**：内置精美中文经义与莎翁台词引导词预设，支持采样温度（Temperature）与 Top-K 截断实时微调。

### 2. 👁️ 计算机视觉与图像识别 (Vision Studio)
* **纯原生 PyTorch 视觉引擎**：内置 `SimpleCNN (0.6M)` 与工业级残差网络 `ResNet-18 (11.2M)`，零额外复杂环境依赖。
* **双指标实时收敛图表**：Canvas 毫秒级绘制 **Train/Val Loss** 损失轨迹与 **Top-1 准确率 (Accuracy %)** 曲线，直观见证识别率从 10% 飙升至 90%+。
* **交互式识别测试台**：
  * 支持拖拽上传任意本地 JPG/PNG 图片；
  * 提供 10 类典型目标（飞机、汽车、鸟、猫、鹿、狗、青蛙、马、船、卡车）一键测试样张；
  * 实时输出最高置信度预测、毫秒级推理耗时与 **Top-5 概率分布水平条形图**。

### 3. 🗂️ 全模态数据集资产管理 (Dataset Manager)
* **自由增删查**：集中打理文本与图像数据集，实时统计样本规模、词表大小与磁盘占用。
* **私有数据集扩展规范**：支持标准的**文件夹分类格式（Folder-per-class）**，把自己的照片按文件夹分类放置，即可自动识别为训练标签。
* **安全清理**：一键安全清理废弃检查点与数据集，释放磁盘空间。

### 4. 💻 智能硬件自适应 (Auto Hardware Detection)
* 系统启动时自动探测宿主机算力环境：
  * **有显卡时**：自动点亮 CUDA 绿色标牌，调用 GPU 高并发与显存加速；
  * **无显卡时**：自动启用多线程 CPU 并发优化，保证绝大多数小模型在本地流畅秒级运行。

---

## ⚡ 极速上手 (Quick Start)

### 1. 克隆仓库与准备环境

```bash
# 1. 克隆本项目
git clone https://github.com/your-username/nanoStudio.git
cd nanoStudio

# 2. 安装基础依赖 (仅需 PyTorch、Pillow 与 NumPy)
pip install torch pillow numpy
```

### 2. 一键启动交互式控制台

* **Windows 用户（推荐极速启动）**：
  直接双击根目录的 **`start.bat`**（或在终端运行 `.\start.bat`），系统将自动校验依赖并弹窗打开浏览器控制台！

* **跨平台命令行启动**：
  ```bash
  python server.py
  ```

在终端看到就绪提示后，打开浏览器访问：
👉 **`http://127.0.0.1:8000`**

立即开启您的多模态深度学习实验之旅！

---

## 📁 目录架构 (Directory Structure)

```text
nanoStudio/
├── server.py              # 高并发多线程 Web 后端调度器 (API + 硬件检测 + 训练管理)
├── tasks/                 # 核心多模态任务模块体系
│   ├── llm/               # 语言大模型工坊 (GPT-2 解码器模型、训练、采样与基准测试)
│   │   ├── model.py       # GPT-2 Transformer 架构定义
│   │   ├── train.py       # 自回归语言模型训练引擎
│   │   ├── sample.py      # 文本生成采样推理引擎
│   │   └── configurator.py# 超参数命令行解析器
│   └── vision/            # 计算机视觉专属任务包 (纯 PyTorch 原生实现)
│       ├── model.py       # SimpleCNN 与 ResNet-18 骨干模型
│       ├── dataset.py     # 视觉数据集加载、标准化与 CIFAR-10 流水线
│       ├── train.py       # 视觉训练循环 (统计 Loss 与 Accuracy)
│       └── infer.py       # 单图前向推理与 Top-K 概率解析
├── checkpoints/           # 集中收纳所有模型权重检查点
│   ├── llm_yuye_tanjing/  # 《雨夜谈经》中文模型检查点
│   ├── llm_shakespeare_char/# 莎翁英文字符级模型检查点
│   └── vision/            # CIFAR-10 视觉分类模型检查点
├── data/                  # 全模态数据集资产目录
│   ├── yuye_tanjing/      # 中文书籍《雨夜谈经》分词与二进制训练集 (含 raw/ 原始 PDF)
│   ├── shakespeare_char/  # 莎士比亚经典字符级数据集
│   └── vision/            # 视觉数据集存储目录 (如 cifar10、私有图片集)
├── notebooks/             # 深度学习理论与 Scaling Law 探索笔记本
│   ├── scaling_laws.ipynb # 计算量与 Loss 缩放律实证分析
│   └── transformer_sizing.ipynb # Transformer 尺寸与显存测算
├── web/                   # 纯原生现代玻璃拟态 (Glassmorphism) 前端
│   ├── index.html         # 交互式单页控制台结构
│   ├── style.css          # 高科技暗黑美学样式体系
│   └── app.js             # 实时 Canvas 折线图、拖拽预测与数据流转
├── config/                # 训练超参数预设配置
├── docs/                  # 技术设计规范与实现计划
└── train.py / sample.py   # 根目录向下兼容命令行代理快捷入口
```

---

## 🎯 路线图 (Roadmap)

- [x] nanoGPT 文本训练与 Windows/跨平台 UTF-8 兼容性优化
- [x] 中文书籍（PDF/TXT）字符级打包与微型中文模型实战
- [x] 6 档模型尺寸预设与实时参数量/显存理论计算器
- [x] 计算机视觉模块上线（SimpleCNN / ResNet-18 + CIFAR-10）
- [x] 拖拽图片即时预测与 Top-5 概率柱状条可视化
- [x] 全模态数据集资产管理中心（支持自定义文件夹格式扩展）
- [ ] 🎨 生成式扩散模型（DDPM / DCGAN）去噪过程逐步动画化
- [ ] 🤖 问答指令微调（SFT）对话模式支持
- [ ] 📦 针对纯新手的一键桌面可执行文件（Desktop App 打包）

---

## 📜 许可证与鸣谢 (License & Credits)

* 本项目基于 **MIT 许可证** 开源，免费供学术研究、个人学习与商业参考。
* 特别鸣谢：
  * **[Andrej Karpathy](https://github.com/karpathy)**：致敬经典原作 [nanoGPT](https://github.com/karpathy/nanoGPT) 对全球深度学习教育的卓越启发！
  * **PyTorch Team**：提供优雅简洁的张量计算与自动微分引擎。

---

<div align="center">
  <sub>如果这个项目对您的深度学习与大模型学习有所启发，欢迎点一个 ⭐️ <b>Star</b> 支持作者！</sub>
</div>
