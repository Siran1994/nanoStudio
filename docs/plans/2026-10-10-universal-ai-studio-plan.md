# Universal Multi-Modal AI Studio Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 nanoGPT 控制台重构为支持 LLM、Vision 图像识别与分类、Diffusion 生图的通用多模态 AI 实验台，实现即时图片上传推理、准确率/Loss双指标监控与数据集自由增删改查。

**Architecture:** 
- 前端：顶部导航中枢（LLM / Vision / Diffusion 模式无缝切换），各模态自适应专属配置与体验台。
- 后端：`tasks/vision/` 纯 PyTorch 实现无额外重依赖的视觉网络（SimpleCNN、ResNet-18）与 CIFAR-10/自定义数据集流；`server.py` 提供统一硬件探测（CUDA/CPU）、数据集 CRUD API 与视觉训练/推理调度。

**Tech Stack:** Python 3, PyTorch, Pillow, Vanilla HTML5/CSS3/JavaScript (Canvas 图表, Glassmorphism 现代暗黑风).

**Spec:** [Universal Multi-Modal AI Studio Design Spec](file:///d:/Desktop/nanoGPT/docs/specs/2026-10-10-universal-ai-studio-design.md)

---

## File Structure

```
d:/Desktop/nanoGPT/
├── docs/
│   ├── specs/2026-10-10-universal-ai-studio-design.md
│   └── plans/2026-10-10-universal-ai-studio-plan.md
├── tasks/
│   ├── vision/
│   │   ├── __init__.py
│   │   ├── model.py         # SimpleCNN, ResNet-18, ViT 基础架构
│   │   ├── dataset.py       # CIFAR-10, MNIST 与自定义文件夹格式加载器
│   │   ├── train.py         # 视觉训练循环，记录 Loss 与 Accuracy
│   │   └── infer.py         # 单张图像前向预测与 Top-K 概率解析
├── data/
│   └── vision/              # 视觉数据集存储目录 (cifar10, custom 等)
├── server.py                # 升级：硬件探测、多模态路由、数据集 CRUD API
└── web/
    ├── index.html           # 升级：顶级多任务中枢导航、Vision 控制台、数据集管理面板
    ├── style.css            # 升级：Vision 上传画板、Top-5 概率条、模式切换器样式
    └── app.js               # 升级：多任务控制器、Dual-Metric Canvas 图表、图片上传推理
```

---

## Tasks

### Task 1: 视觉核心模型与数据集加载器 (`tasks/vision/`)
- [ ] 创建 `tasks/vision/model.py`，实现 `SimpleCNN`（3层卷积+池化，~0.2M 参数，CPU 极速）和 `ResNet18`（残差模块，适配图像特征提取）。
- [ ] 创建 `tasks/vision/dataset.py`，实现 CIFAR-10 10类标签映射、自动下载/解压工具、以及支持本地自定义文件夹分类的 `FolderDataset`，自带轻量数据增广（标准化、随机翻转）。
- [ ] 编写测试脚本验证 `model.py` 与 `dataset.py` 前向传播正常。

### Task 2: 视觉训练与推理脚本 (`tasks/vision/train.py`, `infer.py`)
- [ ] 编写 `tasks/vision/train.py`，支持 CLI 参数解析（`--dataset`, `--model`, `--batch_size`, `--epochs`, `--lr`, `--device`），每个 epoch/step 评估并输出 `train_loss`, `val_loss`, `val_acc`，保存最佳模型检查点至 `out-vision/ckpt.pt`。
- [ ] 编写 `tasks/vision/infer.py`，支持读取单张图片文件或 Base64 流，缩放并标准化，输出 JSON 格式的 Top-5 预测标签与百分比概率。
- [ ] 测试单次 10-step 训练和单图推理验证。

### Task 3: 后端引擎多模态与数据集 CRUD API (`server.py`)
- [ ] 在 `server.py` 中增加硬件探测功能（检测 CUDA 设备型号、显存大小、CPU 核心数），开放 `GET /api/hardware`。
- [ ] 增加数据集资产管理 API：
  - `GET /api/datasets`：返回文本数据集与视觉数据集列表及元信息（样本数、类别数）。
  - `POST /api/datasets/manage`：支持删除指定数据集或一键触发下载。
- [ ] 增加视觉任务调度路由：
  - `POST /api/vision/train/start` 与 `POST /api/vision/train/stop`；
  - `GET /api/vision/state`：同步视觉训练日志、实时 Loss 与 Accuracy 历史数据；
  - `POST /api/vision/infer`：接收前端上传的图片并返回 Top-K 结果。

### Task 4: 前端界面升级：顶级任务切换中枢与视觉工作台 (`web/index.html`, `style.css`)
- [ ] 在 `web/index.html` 顶部增加顶级模态切换药丸（`📝 语言大模型 LLM` / `👁️ 图像识别与分类 Vision` / `🎨 图像生成 Diffusion`）。
- [ ] 新建 `section#vision-task-container` 视觉专属视图，包含：
  - 超参数区：骨干网络切换（SimpleCNN / ResNet-18）、数据集选择（CIFAR-10 / MNIST / 自定义）、Batch Size、Epochs、学习率；
  - 监控区：实时 Loss 曲线与 **分类准确率 (Accuracy %)** 曲线绘制；
  - 视觉实战测试台：拖拽上传图片区域、预置测试样张点击按钮、预测类别大卡片与 Top-5 概率柱状条。
- [ ] 在 `web/style.css` 中增加 Vision 预测条、上传拖拽边框高亮、双指标图表等现代暗黑 Glassmorphism 样式。

### Task 5: 前端交互逻辑与图像即时预测 (`web/app.js`)
- [ ] 在 `web/app.js` 中接入任务切换逻辑（LLM vs Vision 视图无缝切换与状态保持）。
- [ ] 接入视觉训练控制（启动/停止、实时轮询 `/api/vision/state`，在 Canvas 上双轴渲染 Loss 与 Accuracy）。
- [ ] 接入视觉推理交互：支持从本地选择图片文件即时上传并调用 `/api/vision/infer`，动态展示 Top-5 概率动画与预测结果。
- [ ] 预置 10 张经典 CIFAR-10 样例图片供一键点击测试。

### Task 6: 联调与端到端系统验证
- [ ] 重启 `server.py` 服务；
- [ ] 验证 LLM 任务与 Vision 任务切换流畅；
- [ ] 运行一次 CIFAR-10 快速训练，验证 Loss 下降与 Accuracy 上升曲线正常呈现在图表上；
- [ ] 上传测试图片验证 Top-5 预测输出正确。
