# Universal Multi-Modal AI Studio: 架构与数据集管理设计规范

- **状态**: 规划与设计审核 (Pending User Approval)
- **版本**: v1.0
- **创建时间**: 2026-10-10
- **目标**: 将单一的字符级 nanoGPT 拓展升级为一个高扩展性、可插拔的通用多模态 AI 实验工作台（Universal Multi-Modal Deep Learning Studio），支持大语言模型 (LLM)、计算机视觉与图像识别 (Vision)、图像生成与扩散 (Diffusion)，并提供全模态数据集资产管理 (CRUD)。

---

## 一、系统架构概览

```
[Web 前端界面: web/index.html + app.js + style.css]
                     │
                     ▼ (RESTful API / JSON)
        [通用服务端引擎: server.py]
  ├── 硬件自适应管理器 (CUDA GPU / CPU Auto-detect)
  ├── 任务调度中枢 (Task Runner & Process Manager)
  └── 数据集资产管理器 (Dataset Asset Manager - CRUD)
                     │
       ┌─────────────┼─────────────┐
       ▼             ▼             ▼
  tasks/llm/   tasks/vision/   tasks/diffusion/
 (nanoGPT)     (CNN/ResNet)      (DDPM/GAN)
```

---

## 二、数据集资产管理中枢 (Dataset CRUD Manager)

为了满足用户随时自由扩充、管理各类模型训练数据的需求，建立统一的本地数据集管理协议：

### 1. 图像数据集 (Vision Datasets)
- **存储路径**: `data/vision/<dataset_name>/`
- **支持格式**: 
  - 目录分类结构（Folder-per-class 规范）：
    ```
    data/vision/my_custom_dataset/
    ├── train/
    │   ├── cat/
    │   └── dog/
    ├── val/
    │   ├── cat/
    │   └── dog/
    └── meta.json (类别列表、样本数、图片尺寸等)
    ```
- **CRUD 功能**:
  - **Create (增)**: 
    - 方式 A: 一键下载内置常用基准集（CIFAR-10 彩色图像、MNIST 手写数字）；
    - 方式 B: 上传本地 `.zip` 压缩包并自动解压构建类别映射。
  - **Read (查)**: 
    - 列出所有已就绪的视觉数据集；
    - 展示类别总数（Classes）、图片数量、尺寸及示例缩略图抽样预览。
  - **Update (改)**: 
    - 允许配置训练/验证集划分比例（如 8:2 或 9:1）；
    - 允许修改类别别名。
  - **Delete (删)**: 
    - 一键安全删除指定数据集目录及其缓存。

### 2. 文本数据集 (LLM Datasets)
- **存储路径**: `data/<dataset_name>/`
- **CRUD 功能**:
  - **增**: 上传/选择 `.pdf`、`.txt`，自动调用预处理脚本提取词表、生成 `train.bin` 与 `val.bin`；
  - **查**: 查看字符集与词表大小（如《雨夜谈经》1,782 字，莎士比亚 65 字符），预览文本片段；
  - **删**: 安全清理指定文本库及对应 `.bin` 缓存。

---

## 三、图像识别与分类模块 (Vision Module) 设计

### 1. 骨干模型与算力支持
- **模型实现** (`tasks/vision/model.py`):
  - **SimpleCNN**: 3层卷积 + MaxPooling + 全连接层（~0.2M 参数，CPU 秒级训练）；
  - **ResNet-18**: 经典残差网络（~11.2M 参数，GPU 高效并发，具备强悍特征提取能力）；
  - **ViT-Tiny**: 轻量级视觉 Transformer（基于 Patch Embedding 与自注意力）。
- **硬件自适应**:
  - 自动探测 `torch.cuda.is_available()`；
  - 显卡可用时启用 CUDA 加速并展示显卡型号；无显卡时自动启用多线程 CPU 训练。

### 2. 实时训练与双指标监控
- **训练指标**:
  - **Loss 折线**: Train Loss 与 Val 交叉熵损失；
  - **准确率折线 (Accuracy %)**: Top-1 验证集准确率，直观呈现模型从 10% 提升至 80%+ 的全过程。

### 3. 图像识别实战测试台 (Vision Playground)
- **图片上传与拖拽**: 用户可拖入任意 JPG/PNG 图像；
- **样例库一键测试**: 预置 CIFAR-10 各类别精选测试样张；
- **即时推理反馈**:
  - 展示最高置信度预测（如 `🐱 猫 (Cat): 95.4%`）；
  - 动态渲染 Top-5 类别概率条形图；
  - 标记端到端推理耗时（毫秒级）。

---

## 四、API 接口定义

| 端点 | 方法 | 描述 |
| :--- | :--- | :--- |
| `/api/hardware` | GET | 获取当前硬件状态（CPU 核心数 / CUDA 设备名称、显存） |
| `/api/task/switch` | POST | 切换当前工作模式 (`llm` / `vision` / `diffusion`) |
| `/api/datasets` | GET | 查询所有模态已安装的数据集列表 |
| `/api/datasets/download` | POST | 一键下载官方基准数据集（如 cifar10 / mnist） |
| `/api/datasets/delete` | POST | 删除指定的数据集 |
| `/api/vision/train/start`| POST | 启动图像分类训练任务 |
| `/api/vision/train/stop` | POST | 终止当前视觉训练任务 |
| `/api/vision/infer` | POST | 上传图像文件或 Base64，返回类别预测与 Top-K 概率分布 |

---

## 五、实施里程碑

1. **里程碑 1 (架构与视觉模块骨架)**: 创建 `tasks/vision/` 目录体系，实现数据集加载、SimpleCNN 与 ResNet 模型架构，完善 `server.py` 硬件检测与多模态路由。
2. **里程碑 2 (视觉训练与推理引擎)**: 实现标准的 CIFAR-10 自动化加载与训练流水线，支持准确率指标统计与保存。
3. **里程碑 3 (前端多模态界面与测试台)**: 升级前端 UI，实现顶栏模态切换、视觉训练实时双指标绘制（Loss+Accuracy）、以及拖拽上传图像即时预测。
4. **里程碑 4 (数据集管理与增删改查)**: 提供数据集卡片管理与一键增删改查界面。
