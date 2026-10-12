# nanoStudio 项目结构归类与规范化重构计划

- **状态**: 待执行 (Ready for Execution)
- **目标**: 将 nanoStudio 项目中散落的文件与文件夹进行工业级标准化归类整理：
  1. 统一收纳所有模型检查点（`out-*`）至 `checkpoints/`；
  2. 收纳所有 Jupyter Notebook 探究脚本至 `notebooks/`；
  3. 将孤立的 `PDF/` 移入对应数据集 `data/yuye_tanjing/raw/`；
  4. 将根目录的 LLM 核心代码收纳至 `tasks/llm/`，与 `tasks/vision/` 形成统一对称的多模态任务架构；并在根目录提供透明向下兼容转发，确保旧命令不报错；
  5. 更新 `server.py`、`web/` 前端、`prepare.py` 及相关配置，保证所有功能零 regression。

---

## 一、目标目录结构映射 (Target Structure)

```text
nanoStudio/
├── checkpoints/                  # 【集中收纳所有模型权重检查点】
│   ├── llm_yuye_tanjing/         # 迁移自 out-yuye-tanjing/ (ckpt.pt)
│   ├── llm_shakespeare_char/     # 迁移自 out-shakespeare-char/ (ckpt.pt)
│   └── vision/                   # 迁移自 out-vision/ (ckpt.pt)
├── notebooks/                    # 【集中收纳 Jupyter 实验探索笔记】
│   ├── scaling_laws.ipynb        # 迁移自根目录
│   └── transformer_sizing.ipynb  # 迁移自根目录
├── data/                         # 【数据集资产】
│   ├── yuye_tanjing/
│   │   ├── raw/
│   │   │   └── 113-雨夜谈经.pdf   # 迁移自根目录 PDF/
│   │   ├── input.txt
│   │   ├── meta.pkl
│   │   ├── prepare.py            # 更新 PDF 寻找路径
│   │   ├── train.bin
│   │   └── val.bin
│   ├── shakespeare_char/
│   ├── vision/
│   └── ...
├── tasks/                        # 【多模态核心任务体系】
│   ├── llm/                      # 【LLM 任务引擎：迁移自根目录】
│   │   ├── __init__.py
│   │   ├── model.py              # GPT-2 核心架构
│   │   ├── train.py              # 训练主循环
│   │   ├── sample.py             # 自回归文本生成
│   │   ├── bench.py              # 吞吐基准测算
│   │   └── configurator.py       # 参数解析器
│   └── vision/                   # 【计算机视觉引擎】
│       ├── __init__.py
│       ├── model.py
│       ├── dataset.py
│       ├── train.py
│       └── infer.py
├── config/                       # LLM 预设训练配置（同步更新默认 out_dir）
├── web/                          # 前端单页系统 (index.html, style.css, app.js)
├── docs/                         # 技术规范与规划文档
├── assets/                       # 静态资源与图片
├── server.py                     # 【根目录唯一核心服务调度入口】
├── train.py                      # 【根目录向下兼容转发代理】
├── sample.py                     # 【根目录向下兼容转发代理】
├── model.py                      # 【根目录向下兼容导出代理】
├── .gitignore                    # 【规范化忽略规则】
└── README.md                     # 【更新项目架构树说明】
```

---

## 二、实施任务分解 (Tasks Breakdown)

### 任务 1: 创建目标目录并迁移权重与笔记
- [x] 创建 `checkpoints/llm_yuye_tanjing`，将 `out-yuye-tanjing/ckpt.pt` 移入；
- [x] 创建 `checkpoints/llm_shakespeare_char`，将 `out-shakespeare-char/ckpt.pt` 移入；
- [x] 创建 `checkpoints/vision`，将 `out-vision/ckpt.pt` 移入；
- [x] 删除原根目录已空的 `out-yuye-tanjing/`、`out-shakespeare-char/`、`out-vision/`、`out/`；
- [x] 创建 `notebooks/`，将根目录的 `scaling_laws.ipynb` 与 `transformer_sizing.ipynb` 移动至该目录。

### 任务 2: 整理原始语料与数据预处理脚本
- [x] 创建 `data/yuye_tanjing/raw/`，将 `PDF/113-雨夜谈经.pdf` 移入；
- [x] 移除根目录空文件夹 `PDF/`；
- [x] 修改 `data/yuye_tanjing/prepare.py`，优先读取 `data/yuye_tanjing/raw/113-雨夜谈经.pdf`，保证兼容性；
- [x] 测试运行 `prepare.py` 验证路径有效性。

### 任务 3: 构建 `tasks/llm/` 模块与根目录向后兼容代理
- [x] 创建 `tasks/llm/` 目录；
- [x] 将 `model.py`、`train.py`、`sample.py`、`bench.py`、`configurator.py` 移动至 `tasks/llm/`；
- [x] 在 `tasks/llm/` 内部调整相对导入与运行路径（使 `tasks/llm/train.py` 和 `sample.py` 无论作为脚本运行还是模块导入均能精准解析 `configurator.py` 与 `model.py`）；
- [x] 在根目录创建向后兼容包装器：
  - `model.py`：导出 `from tasks.llm.model import *`；
  - `train.py`：委托调用 `tasks/llm/train.py`；
  - `sample.py`：委托调用 `tasks/llm/sample.py`；
- [x] 更新 `config/` 中的配置文件（`train_yuye_tanjing.py`, `train_shakespeare_char.py` 等），将其默认 `out_dir` 更新为 `checkpoints/llm_*`。

### 任务 4: 更新后端调度中枢 (`server.py`)
- [x] 更新 `server.py` 中的 LLM 训练命令路径与默认 checkpoint 路径：
  - 支持 `checkpoints/llm_yuye_tanjing`、`checkpoints/llm_shakespeare_char`、`checkpoints/vision`；
  - 增加向下兼容逻辑（如果检测到老路径也能平滑 fallback）；
  - 视觉模型训练参数 `--out_dir=checkpoints/vision`；
- [x] 更新 `get_checkpoint_info` 与 `checkpoints` 接口，准确监控 `checkpoints/` 下的新路径。

### 任务 5: 更新前端交互与展示 (`web/index.html` & `web/app.js`)
- [x] 更新 `web/index.html` 中的 `data-out` 属性为新 `checkpoints/` 路径；
- [x] 更新 `web/app.js` 中的默认检查点路径变量与检查点管理表格；
- [x] 检查所有 UI 标签与文案，确保显示整洁一致。

### 任务 6: 更新 `.gitignore` 与 `README.md`
- [x] 更新 `.gitignore`，规范忽略 `checkpoints/` 下的动态权重（保留目录占位 `.gitkeep`）以及 `__pycache__`、`notebooks/.ipynb_checkpoints` 等；
- [x] 更新 `README.md` 中的目录架构图与运行命令说明。

### 任务 7: 完整端到端验证 (End-to-End Verification)
- [x] 验证文本采样命令：`python sample.py --out_dir=checkpoints/llm_yuye_tanjing --device=cpu --start="夜深"`；
- [x] 验证视觉推理命令：`python -c "from tasks.vision.infer import predict; ..."`；
- [x] 启动 `server.py`，发起 API 探测测试 `/api/state`、`/api/checkpoints`、`/api/hardware`、`/api/datasets`；
- [x] 确认全流程运行正常，根目录整洁清爽。
