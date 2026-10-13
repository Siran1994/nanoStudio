"""
Vision Inference Pipeline
Load checkpoint from out-vision/ckpt.pt and predict top-k classes for an image.
Supports local image path or base64 encoded image string.
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import time
import json
import base64
import io
import argparse
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from tasks.vision.model import get_vision_model
from tasks.vision.dataset import CIFAR10_CLASSES

# Global cache for loaded model to make repeated inference super fast
_cached_model = None
_cached_ckpt_path = None
_cached_class_names = None
_cached_device = None


def load_inference_model(ckpt_path=None, device="cpu"):
    global _cached_model, _cached_ckpt_path, _cached_class_names, _cached_device

    if ckpt_path is None:
        new_path = os.path.join(BASE_DIR, "checkpoints", "vision", "ckpt.pt")
        legacy_path = os.path.join(BASE_DIR, "out-vision", "ckpt.pt")
        ckpt_path = new_path if os.path.exists(new_path) else legacy_path

    if not os.path.exists(ckpt_path):
        return None, None, f"权重文件不存在: {ckpt_path}，请先训练视觉模型保存检查点"

    # Reuse cached model if path hasn't changed
    mtime = os.path.getmtime(ckpt_path)
    if _cached_model is not None and _cached_ckpt_path == (ckpt_path, mtime):
        return _cached_model, _cached_class_names, None

    try:
        ckpt = torch.load(ckpt_path, map_location=device)
        model_name = ckpt.get("model_name", "simple_cnn")
        num_classes = ckpt.get("num_classes", 10)
        class_names = ckpt.get("class_names", CIFAR10_CLASSES)

        model = get_vision_model(model_name, num_classes=num_classes)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        model.eval()

        _cached_model = model
        _cached_ckpt_path = (ckpt_path, mtime)
        _cached_class_names = class_names
        _cached_device = device

        return model, class_names, None
    except Exception as e:
        return None, None, f"加载权重失败: {e}"


def preprocess_image(img_input):
    """
    Accepts: PIL Image, file path, or base64 string.
    Returns: Tensor of shape (1, 3, 32, 32)
    """
    if isinstance(img_input, str):
        if img_input.startswith("data:image") or len(img_input) > 256:
            # Base64 string
            if "base64," in img_input:
                img_input = img_input.split("base64,")[1]
            img_bytes = base64.b64decode(img_input)
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        else:
            # File path
            img = Image.open(img_input).convert("RGB")
    elif isinstance(img_input, Image.Image):
        img = img_input.convert("RGB")
    else:
        raise ValueError("不支持的图像输入格式")

    # Resize to 32x32 standard benchmark size
    img = img.resize((32, 32), Image.Resampling.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0  # (32, 32, 3)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1, 3, 32, 32)
    return tensor


def predict(img_input, ckpt_path=None, top_k=5, device="cpu"):
    t0 = time.time()
    model, class_names, err = load_inference_model(ckpt_path, device=device)
    if err:
        return {"success": False, "error": err}

    try:
        tensor = preprocess_image(img_input).to(device)
    except Exception as e:
        return {"success": False, "error": f"图像预处理失败: {e}"}

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()

    # Get top-k indices
    top_indices = np.argsort(probs)[::-1][:top_k]
    top_list = []
    for idx in top_indices:
        cls_name = class_names[idx] if idx < len(class_names) else f"Class_{idx}"
        prob_pct = round(float(probs[idx]) * 100, 2)
        top_list.append({
            "class_id": int(idx),
            "class_name": cls_name,
            "probability": prob_pct
        })

    elapsed_ms = round((time.time() - t0) * 1000, 2)
    best = top_list[0]

    return {
        "success": True,
        "top_prediction": best,
        "top_k": top_list,
        "inference_time_ms": elapsed_ms
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, help="Image file path or base64")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    if args.image:
        res = predict(args.image, ckpt_path=args.ckpt, top_k=args.top_k)
        print(json.dumps(res, ensure_ascii=False, indent=2))
    else:
        print("Usage: python tasks/vision/infer.py --image <image_path>")
