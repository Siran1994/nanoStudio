"""
nanoGPT Studio Web Server
High-concurrency ThreadingHTTPServer with full CORS support:
- Non-blocking multi-threaded request processing
- Robust prompt handling via FILE: temp-prompt to prevent Windows CLI argument breaks
- Automatic vocabulary validation & character sanitization for Shakespeare char model
- Hyperparameter tuning, real-time training monitor & loss metrics
- Interactive text generation & model playground
"""

import os
import sys
import json
import time
import pickle
import threading
import subprocess

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse

PORT = 8000
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(BASE_DIR, 'web')

# Training state holder
training_lock = threading.Lock()
training_proc = None
training_state = {
    "is_training": False,
    "status": "idle",       # idle, training, finished, stopped, error
    "current_iter": 0,
    "max_iters": 1000,
    "train_loss": None,
    "val_loss": None,
    "latest_loss": None,
    "latest_time_ms": None,
    "latest_mfu": None,
    "start_time": None,
    "elapsed_seconds": 0,
    "tokens_per_sec": 0,
    "loss_history": [],     # list of {iter, train_loss, val_loss, time_ms}
    "logs": [],             # list of string lines
    "params": {},
    "error_message": None
}

# Cache dataset vocab if available
vocab_cache = {}
def get_dataset_vocab(dataset="shakespeare_char"):
    if dataset in vocab_cache:
        return vocab_cache[dataset]
    meta_path = os.path.join(BASE_DIR, "data", dataset, "meta.pkl")
    if not os.path.exists(meta_path) and "shakespeare" in dataset:
        meta_path = os.path.join(BASE_DIR, "data", "shakespeare_char", "meta.pkl")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
                vocab_cache[dataset] = set(meta["stoi"].keys())
                return vocab_cache[dataset]
        except Exception:
            pass
    return None

def get_checkpoint_info(out_dir="checkpoints/llm_shakespeare_char", dataset=None):
    candidates = [
        os.path.join(BASE_DIR, out_dir, "ckpt.pt"),
        os.path.join(BASE_DIR, out_dir) if out_dir.endswith(".pt") else ""
    ]
    check_str = (out_dir + " " + (dataset or "")).lower()
    if "yuye" in check_str:
        candidates.extend([
            os.path.join(BASE_DIR, "checkpoints", "llm_yuye_tanjing", "ckpt.pt"),
            os.path.join(BASE_DIR, "out-yuye-tanjing", "ckpt.pt")
        ])
    elif "shakespeare" in check_str:
        candidates.extend([
            os.path.join(BASE_DIR, "checkpoints", "llm_shakespeare_char", "ckpt.pt"),
            os.path.join(BASE_DIR, "checkpoints", "llm_shakespeare", "ckpt.pt"),
            os.path.join(BASE_DIR, "out-shakespeare-char", "ckpt.pt")
        ])
    elif "vision" in check_str:
        candidates.extend([
            os.path.join(BASE_DIR, "checkpoints", "vision", "ckpt.pt"),
            os.path.join(BASE_DIR, "out-vision", "ckpt.pt")
        ])
    for p in candidates:
        if p and os.path.exists(p):
            size_mb = os.path.getsize(p) / (1024 * 1024)
            mtime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(p)))
            return {
                "exists": True,
                "path": p,
                "size_mb": round(size_mb, 2),
                "updated_at": mtime
            }
    default_p = os.path.join(BASE_DIR, out_dir, "ckpt.pt")
    return {"exists": False, "path": default_p}

def calculate_model_params(n_layer, n_head, n_embd, block_size, vocab_size=65):
    token_emb = vocab_size * n_embd
    pos_emb = block_size * n_embd
    attn_params = (n_embd * 3 * n_embd) + (n_embd * n_embd)
    mlp_params = (n_embd * 4 * n_embd) + (4 * n_embd * n_embd)
    ln_params = 2 * n_embd * 2
    block_params = attn_params + mlp_params + ln_params
    total_transformer = n_layer * block_params
    final_ln = 2 * n_embd
    total_params = token_emb + pos_emb + total_transformer + final_ln
    return {
        "total_params": total_params,
        "total_params_m": round(total_params / 1e6, 3),
        "est_memory_mb": round((total_params * 4 * 3) / (1024 * 1024), 2)
    }

def training_worker(cmd, params):
    global training_proc, training_state
    
    with training_lock:
        training_state["is_training"] = True
        training_state["status"] = "training"
        training_state["current_iter"] = 0
        training_state["max_iters"] = int(params.get("max_iters", 1000))
        training_state["train_loss"] = None
        training_state["val_loss"] = None
        training_state["latest_loss"] = None
        training_state["start_time"] = time.time()
        training_state["elapsed_seconds"] = 0
        training_state["loss_history"] = []
        training_state["logs"] = []
        training_state["params"] = params
        training_state["error_message"] = None

    try:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        
        training_proc = subprocess.Popen(
            cmd,
            cwd=BASE_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
            encoding='utf-8',
            errors='replace'
        )

        for line in iter(training_proc.stdout.readline, ''):
            clean_line = line.strip()
            if not clean_line:
                continue
            
            with training_lock:
                training_state["logs"].append(clean_line)
                if len(training_state["logs"]) > 500:
                    training_state["logs"].pop(0)
                
                training_state["elapsed_seconds"] = round(time.time() - training_state["start_time"], 1)

                if "train loss" in clean_line and "val loss" in clean_line:
                    try:
                        parts = clean_line.split(":")
                        step_num = int(parts[0].replace("step", "").strip())
                        subparts = parts[1].split(",")
                        train_l = float(subparts[0].replace("train loss", "").strip())
                        val_l = float(subparts[1].replace("val loss", "").strip())
                        training_state["current_iter"] = step_num
                        training_state["train_loss"] = train_l
                        training_state["val_loss"] = val_l
                        training_state["loss_history"].append({
                            "iter": step_num,
                            "train_loss": train_l,
                            "val_loss": val_l,
                            "type": "eval"
                        })
                    except Exception:
                        pass

                elif clean_line.startswith("iter ") and "loss" in clean_line:
                    try:
                        parts = clean_line.split(":")
                        iter_num = int(parts[0].replace("iter", "").strip())
                        info = parts[1].split(",")
                        loss_val = float(info[0].replace("loss", "").strip())
                        time_str = info[1].replace("time", "").replace("ms", "").strip()
                        time_ms = float(time_str)
                        training_state["current_iter"] = iter_num
                        training_state["latest_loss"] = loss_val
                        training_state["latest_time_ms"] = time_ms
                        
                        training_state["loss_history"].append({
                            "iter": iter_num,
                            "train_loss": loss_val,
                            "val_loss": None,
                            "time_ms": time_ms,
                            "type": "train_step"
                        })
                    except Exception:
                        pass

        training_proc.wait()
        return_code = training_proc.returncode

        with training_lock:
            if return_code == 0:
                training_state["status"] = "finished"
                training_state["logs"].append("🎉 训练已成功完成！模型检查点已保存。")
            else:
                if training_state["status"] != "stopped":
                    training_state["status"] = "error"
                    training_state["error_message"] = f"进程退出，返回码: {return_code}"
                    training_state["logs"].append(f"❌ 训练异常退出，代码: {return_code}")
            training_state["is_training"] = False

    except Exception as e:
        with training_lock:
            training_state["status"] = "error"
            training_state["error_message"] = str(e)
            training_state["is_training"] = False
            training_state["logs"].append(f"❌ 启动异常: {e}")
    finally:
        with training_lock:
            training_proc = None
            training_state["is_training"] = False


# =====================================================================
# Universal Vision Module & Dataset Asset Management
# =====================================================================
vision_lock = threading.Lock()
vision_proc = None
vision_state = {
    "is_training": False,
    "status": "idle",       # idle, training, finished, stopped, error
    "epoch": 0,
    "max_epochs": 10,
    "step": 0,
    "train_loss": None,
    "val_loss": None,
    "val_acc": None,
    "latest_loss": None,
    "start_time": None,
    "elapsed_seconds": 0,
    "loss_history": [],     # list of {step, epoch, train_loss, val_loss, val_acc}
    "logs": [],
    "params": {},
    "error_message": None
}

def get_hardware_info():
    import torch
    info = {
        "cpu_count": os.cpu_count() or 4,
        "cuda_available": torch.cuda.is_available(),
        "device_name": f"CPU ({os.cpu_count() or 4} 核心并发优化)",
        "vram_total_gb": 0,
        "device_type": "cpu",
        "physical_gpu_detected": False,
        "physical_gpu_name": None,
        "physical_gpu_vram_mb": 0,
        "torch_version": torch.__version__,
        "hint": ""
    }
    if torch.cuda.is_available():
        info["device_name"] = torch.cuda.get_device_name(0)
        info["device_type"] = "cuda"
        info["device_count"] = torch.cuda.device_count()
        props = torch.cuda.get_device_properties(0)
        info["vram_total_gb"] = round(props.total_memory / (1024**3), 2)
        info["physical_gpu_detected"] = True
        info["physical_gpu_name"] = info["device_name"]
        info["physical_gpu_vram_mb"] = round(props.total_memory / (1024**2))
    else:
        # Check physical GPU via nvidia-smi
        try:
            res = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2
            )
            if res.returncode == 0 and res.stdout.strip():
                parts = [p.strip() for p in res.stdout.strip().splitlines()[0].split(",")]
                if len(parts) >= 2:
                    gpu_name = parts[0]
                    vram_mb = int(parts[1])
                    info["physical_gpu_detected"] = True
                    info["physical_gpu_name"] = gpu_name
                    info["physical_gpu_vram_mb"] = vram_mb
                    info["hint"] = f"已检测到物理显卡 {gpu_name} ({round(vram_mb/1024, 1)}GB)，但当前 Python 环境中的 PyTorch ({torch.__version__}) 为 CPU 编译版本，尚未启用 CUDA 驱动库。"
        except Exception:
            pass
    return info

def get_dataset_inventory():
    items = []
    # 1. Text datasets
    data_dir = os.path.join(BASE_DIR, "data")
    if os.path.exists(data_dir):
        for entry in os.listdir(data_dir):
            p = os.path.join(data_dir, entry)
            if os.path.isdir(p) and entry != "vision":
                train_b = os.path.join(p, "train.bin")
                meta_p = os.path.join(p, "meta.pkl")
                if not os.path.exists(train_b) and not os.path.exists(meta_p):
                    continue
                sz_mb = os.path.getsize(train_b) / (1024*1024) if os.path.exists(train_b) else 0
                vocab_sz = 0
                custom_name = None
                is_cjk = False

                if os.path.exists(meta_p):
                    try:
                        with open(meta_p, "rb") as mf:
                            meta = pickle.load(mf)
                            vocab_sz = meta.get("vocab_size", len(meta.get("stoi", {})))
                            custom_name = meta.get("name")
                            stoi = meta.get("stoi", {})
                            if stoi:
                                is_cjk = any('\u4e00' <= ch <= '\u9fff' for ch in stoi.keys())
                    except Exception:
                        pass

                if entry in ("shakespeare_char", "shakespeare", "openwebtext"):
                    is_cjk = False
                elif "yuye" in entry or "daode" in entry:
                    is_cjk = True

                name_map = {
                    "yuye_tanjing": "🇨🇳 《雨夜谈经》中文书籍",
                    "shakespeare_char": "🇬🇧 Shakespeare 英文戏剧",
                    "shakespeare": "🇬🇧 Shakespeare BPE 语料",
                    "openwebtext": "🌐 OpenWebText 英文预训练集"
                }
                disp_name = custom_name or name_map.get(entry, entry)
                items.append({
                    "id": entry,
                    "name": disp_name,
                    "type": "text",
                    "lang": "zh" if is_cjk else "en",
                    "vocab_size": vocab_sz,
                    "size_mb": round(sz_mb, 2),
                    "path": p
                })

    # 2. Vision datasets
    v_dir = os.path.join(BASE_DIR, "data", "vision")
    if os.path.exists(v_dir):
        for entry in os.listdir(v_dir):
            p = os.path.join(v_dir, entry)
            if os.path.isdir(p):
                sz_mb = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(p) for f in fs) / (1024*1024)
                items.append({
                    "id": entry,
                    "name": "CIFAR-10 彩色图像 (10类)" if entry == "cifar10" else entry,
                    "type": "vision",
                    "lang": "en",
                    "classes": 10 if entry == "cifar10" else 0,
                    "size_mb": round(sz_mb, 2),
                    "path": p
                })
    return items

def scan_checkpoints():
    ckpts = []
    ckpt_base = os.path.join(BASE_DIR, "checkpoints")
    if not os.path.exists(ckpt_base):
        return ckpts

    datasets = {d["id"]: d for d in get_dataset_inventory()}

    for entry in sorted(os.listdir(ckpt_base)):
        entry_path = os.path.join(ckpt_base, entry)
        if not os.path.isdir(entry_path):
            continue
        pt_file = os.path.join(entry_path, "ckpt.pt")
        exists = os.path.exists(pt_file)
        size_mb = round(os.path.getsize(pt_file) / (1024 * 1024), 2) if exists else 0
        mtime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(pt_file))) if exists else "-"

        if entry.startswith("llm_"):
            ds_id = entry[4:]
            ds_info = datasets.get(ds_id)
            if ds_info:
                name = f"{ds_info['name']} 模型权重"
                linked_ds = ds_info['name']
                has_dataset = True
            else:
                legacy_names = {
                    "yuye_tanjing": "🇨🇳 《雨夜谈经》中文模型 (Tiny 4.7M)",
                    "shakespeare_char": "🇬🇧 莎士比亚字符级 (Baby 0.8M)",
                    "shakespeare": "🇬🇧 莎士比亚 BPE 微调"
                }
                name = legacy_names.get(ds_id, f"📝 LLM [{ds_id}] 权重")
                linked_ds = f"{ds_id} (数据源已删除)"
                has_dataset = False
            modality = "LLM"
        elif entry == "vision":
            name = "👁️ CIFAR-10 计算机视觉权重 (SimpleCNN)"
            linked_ds = "CIFAR-10 (vision)"
            has_dataset = True
            modality = "Vision"
        else:
            name = f"自定义模型权重 [{entry}]"
            linked_ds = entry
            has_dataset = False
            modality = "Other"

        ckpts.append({
            "dir": f"checkpoints/{entry}",
            "id": entry,
            "name": name,
            "modality": modality,
            "linked_ds": linked_ds,
            "has_dataset": has_dataset,
            "size_mb": size_mb,
            "updated_at": mtime,
            "exists": exists
        })
    return ckpts

def create_text_dataset(dataset_id, name, content):
    import re
    import numpy as np
    clean_id = re.sub(r'[^a-zA-Z0-9_\-]', '', dataset_id).lower()
    if not clean_id:
        return False, "数据集英文标识 (ID) 必须由字母、数字或下划线组成"
    if not content or len(content.strip()) < 10:
        return False, "文本内容过短，请至少提供 10 个字符"

    target_dir = os.path.join(BASE_DIR, "data", clean_id)
    os.makedirs(target_dir, exist_ok=True)

    # 1. 保存 input.txt
    txt_path = os.path.join(target_dir, "input.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(content)

    # 2. 提取唯一字符并建立词表映射
    chars = sorted(list(set(content)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    # 3. 划分 90% 训练集与 10% 验证集并打包二进制
    n = len(content)
    split_idx = max(int(n * 0.9), 1)
    train_data = content[:split_idx]
    val_data = content[split_idx:] if split_idx < n else content

    encode = lambda s: [stoi[c] for c in s]
    train_ids = np.array(encode(train_data), dtype=np.uint16)
    val_ids = np.array(encode(val_data), dtype=np.uint16)

    train_ids.tofile(os.path.join(target_dir, "train.bin"))
    val_ids.tofile(os.path.join(target_dir, "val.bin"))

    # 4. 保存 meta.pkl
    meta = {
        'vocab_size': vocab_size,
        'itos': itos,
        'stoi': stoi,
        'name': name.strip() if name and name.strip() else clean_id
    }
    with open(os.path.join(target_dir, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)

    # 同步更新词表缓存
    vocab_cache[clean_id] = set(stoi.keys())

    return True, {
        "id": clean_id,
        "name": meta["name"],
        "vocab_size": vocab_size,
        "train_tokens": len(train_ids),
        "val_tokens": len(val_ids)
    }

def vision_training_worker(cmd, params):
    global vision_proc, vision_state
    with vision_lock:
        vision_state["is_training"] = True
        vision_state["status"] = "training"
        vision_state["epoch"] = 0
        vision_state["max_epochs"] = int(params.get("epochs", 10))
        vision_state["step"] = 0
        vision_state["train_loss"] = None
        vision_state["val_loss"] = None
        vision_state["val_acc"] = None
        vision_state["start_time"] = time.time()
        vision_state["elapsed_seconds"] = 0
        vision_state["loss_history"] = []
        vision_state["logs"] = []
        vision_state["params"] = params
        vision_state["error_message"] = None

    try:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        vision_proc = subprocess.Popen(
            cmd,
            cwd=BASE_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
            encoding='utf-8',
            errors='replace'
        )

        import re
        for line in iter(vision_proc.stdout.readline, ''):
            clean_line = line.strip()
            if not clean_line:
                continue

            with vision_lock:
                vision_state["logs"].append(clean_line)
                if len(vision_state["logs"]) > 500:
                    vision_state["logs"].pop(0)
                vision_state["elapsed_seconds"] = round(time.time() - vision_state["start_time"], 1)

                if "val loss" in clean_line and "val acc" in clean_line:
                    try:
                        m = re.search(r"epoch (\d+)/\d+ \(step (\d+)\): train loss ([\d\.]+), val loss ([\d\.]+), val acc ([\d\.]+)%", clean_line)
                        if m:
                            ep = int(m.group(1))
                            st = int(m.group(2))
                            tr_l = float(m.group(3))
                            v_l = float(m.group(4))
                            v_acc = float(m.group(5))
                            vision_state["epoch"] = ep
                            vision_state["step"] = st
                            vision_state["train_loss"] = tr_l
                            vision_state["val_loss"] = v_l
                            vision_state["val_acc"] = v_acc
                            vision_state["loss_history"].append({
                                "step": st,
                                "epoch": ep,
                                "train_loss": tr_l,
                                "val_loss": v_l,
                                "val_acc": v_acc
                            })
                    except Exception:
                        pass
                elif clean_line.startswith("iter ") and "loss" in clean_line:
                    try:
                        parts = clean_line.split(":")
                        it = int(parts[0].replace("iter", "").strip())
                        sub = parts[1].split(",")
                        l_val = float(sub[0].replace("loss", "").strip())
                        vision_state["step"] = it
                        vision_state["latest_loss"] = l_val
                    except Exception:
                        pass

        vision_proc.wait()
        code = vision_proc.returncode
        with vision_lock:
            if code == 0:
                vision_state["status"] = "finished"
                vision_state["logs"].append("🎉 视觉模型训练顺利完成！最优检查点已保存至 checkpoints/vision/ckpt.pt")
            else:
                if vision_state["status"] != "stopped":
                    vision_state["status"] = "error"
                    vision_state["error_message"] = f"训练进程异常退出，代码: {code}"
            vision_state["is_training"] = False
    except Exception as e:
        with vision_lock:
            vision_state["status"] = "error"
            vision_state["error_message"] = str(e)
            vision_state["is_training"] = False
    finally:
        with vision_lock:
            vision_proc = None
            vision_state["is_training"] = False


class NanoGPTHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WEB_DIR, **kwargs)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_cors_headers()
        self.end_headers()

    def send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Access-Control-Allow-Private-Network", "true")

    def do_GET(self):
        url = urlparse(self.path)
        
        if url.path == "/api/state":
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_cors_headers()
            self.end_headers()
            
            with training_lock:
                state_copy = dict(training_state)
                out_dir = state_copy["params"].get("out_dir", "checkpoints/llm_shakespeare_char")
                state_copy["checkpoint"] = get_checkpoint_info(out_dir)
                ckpts = {}
                ckpt_base = os.path.join(BASE_DIR, "checkpoints")
                if os.path.exists(ckpt_base):
                    for d in os.listdir(ckpt_base):
                        if d.startswith("llm_"):
                            ds_key = d[4:]
                            ckpts[ds_key] = get_checkpoint_info(os.path.join("checkpoints", d))
                if "shakespeare_char" in ckpts and "shakespeare" not in ckpts:
                    ckpts["shakespeare"] = ckpts["shakespeare_char"]
                state_copy["checkpoints"] = ckpts
            
            self.wfile.write(json.dumps(state_copy).encode('utf-8'))
            return

        elif url.path == "/api/hardware":
            self.send_json_response(get_hardware_info())
            return

        elif url.path == "/api/datasets":
            self.send_json_response(get_dataset_inventory())
            return

        elif url.path == "/api/vision/state":
            with vision_lock:
                st = dict(vision_state)
                st["checkpoint"] = get_checkpoint_info("checkpoints/vision")
            self.send_json_response(st)
            return

        elif url.path == "/api/checkpoints":
            self.send_json_response(scan_checkpoints())
            return

        return super().do_GET()

    def do_POST(self):
        url = urlparse(self.path)
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length) if content_length > 0 else b'{}'
        
        try:
            body = json.loads(post_data.decode('utf-8')) if post_data else {}
        except Exception:
            body = {}

        if url.path == "/api/train/start":
            global training_proc
            with training_lock:
                if training_state["is_training"]:
                    self.send_json_response({"success": False, "error": "训练已在进行中"})
                    return

            block_size = int(body.get("block_size", 64))
            batch_size = int(body.get("batch_size", 12))
            n_layer = int(body.get("n_layer", 4))
            n_head = int(body.get("n_head", 4))
            n_embd = int(body.get("n_embd", 128))
            max_iters = int(body.get("max_iters", 1000))
            lr = float(body.get("learning_rate", 1e-3))
            dropout = float(body.get("dropout", 0.0))
            eval_interval = int(body.get("eval_interval", 100))
            eval_iters = int(body.get("eval_iters", 20))
            log_interval = int(body.get("log_interval", 10))
            dataset = body.get("dataset", "shakespeare_char")
            req_device = body.get("device", "auto")
            import torch
            if req_device in ("auto", "cuda"):
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = "cpu"
            default_out = f"checkpoints/llm_{dataset}"
            out_dir = body.get("out_dir", default_out)

            if n_embd % n_head != 0:
                self.send_json_response({
                    "success": False,
                    "error": f"参数错误：通道维度 n_embd ({n_embd}) 必须能被注意力头数 n_head ({n_head}) 整除！推荐头数：{[h for h in [1,2,4,6,8,12,16] if n_embd % h == 0]}"
                })
                return

            config_file = f"config/train_{dataset}.py"
            if not os.path.exists(os.path.join(BASE_DIR, config_file)):
                config_file = "config/train_shakespeare_char.py"

            cmd = [
                sys.executable, "tasks/llm/train.py", config_file,
                f"--dataset={dataset}",
                f"--device={device}",
                f"--compile=False",
                f"--out_dir={out_dir}",
                f"--block_size={block_size}",
                f"--batch_size={batch_size}",
                f"--n_layer={n_layer}",
                f"--n_head={n_head}",
                f"--n_embd={n_embd}",
                f"--max_iters={max_iters}",
                f"--lr_decay_iters={max_iters}",
                f"--eval_interval={eval_interval}",
                f"--eval_iters={eval_iters}",
                f"--log_interval={log_interval}",
                f"--learning_rate={lr}",
                f"--dropout={dropout}"
            ]

            t = threading.Thread(target=training_worker, args=(cmd, body), daemon=True)
            t.start()
            self.send_json_response({"success": True, "message": "训练任务已成功启动"})
            return

        elif url.path == "/api/train/stop":
            global training_proc
            stopped = False
            with training_lock:
                if training_proc and training_state["is_training"]:
                    try:
                        training_state["status"] = "stopped"
                        training_state["logs"].append("⚠️ 收到用户停止指令，正在终止训练进程...")
                        training_proc.terminate()
                        stopped = True
                    except Exception:
                        pass
            self.send_json_response({"success": stopped, "message": "训练已终止" if stopped else "没有正在运行的任务"})
            return

        elif url.path == "/api/calculate":
            n_layer = int(body.get("n_layer", 4))
            n_head = int(body.get("n_head", 4))
            n_embd = int(body.get("n_embd", 128))
            block_size = int(body.get("block_size", 64))
            vocab_size = int(body.get("vocab_size", 65))
            stats = calculate_model_params(n_layer, n_head, n_embd, block_size, vocab_size)
            self.send_json_response({"success": True, "stats": stats})
            return

        elif url.path == "/api/generate":
            dataset = body.get("dataset", "shakespeare_char")
            default_out = f"checkpoints/llm_{dataset}"
            out_dir = body.get("out_dir", default_out)
            req_device = body.get("device", "auto")
            import torch
            if req_device in ("auto", "cuda"):
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = "cpu"
            raw_prompt = body.get("prompt", "\n")
            max_new_tokens = int(body.get("max_new_tokens", 200))
            temperature = float(body.get("temperature", 0.8))
            top_k = int(body.get("top_k", 200))

            ckpt_info = get_checkpoint_info(out_dir, dataset=dataset)
            if not ckpt_info["exists"]:
                self.send_json_response({
                    "success": False,
                    "error": f"找不到权重文件 {ckpt_info['path']}，请先在控制台启动对应数据集的训练保存权重后再进行测试"
                })
                return
            out_dir_for_sample = os.path.dirname(ckpt_info["path"])

            # Sanitize prompt against dataset vocabulary
            valid_chars = get_dataset_vocab(dataset)
            filtered_prompt = ""
            if valid_chars:
                for ch in raw_prompt:
                    if ch in valid_chars:
                        filtered_prompt += ch
                    elif ch in ('\r', '\t'):
                        filtered_prompt += ' '
                if not filtered_prompt:
                    filtered_prompt = "\n"
            else:
                filtered_prompt = raw_prompt or "\n"

            # Write prompt to temp file to avoid Windows command line argument issues
            prompt_file = os.path.join(BASE_DIR, "_web_prompt_temp.txt")
            try:
                with open(prompt_file, "w", encoding="utf-8") as pf:
                    pf.write(filtered_prompt)
            except Exception as e:
                self.send_json_response({"success": False, "error": f"写入提示词缓存失败: {e}"})
                return

            cmd = [
                sys.executable, "tasks/llm/sample.py",
                f"--out_dir={out_dir_for_sample}",
                f"--device={device}",
                f"--start=FILE:{prompt_file}",
                f"--max_new_tokens={max_new_tokens}",
                f"--temperature={temperature}",
                f"--top_k={top_k}",
                "--num_samples=1",
                "--compile=False"
            ]

            t0 = time.time()
            try:
                res = subprocess.run(
                    cmd,
                    cwd=BASE_DIR,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    encoding='utf-8',
                    errors='replace'
                )
                dt = round(time.time() - t0, 2)
                
                # Cleanup temp file
                if os.path.exists(prompt_file):
                    try:
                        os.remove(prompt_file)
                    except Exception:
                        pass

                if res.returncode == 0:
                    raw_out = res.stdout
                    lines = raw_out.splitlines()
                    clean_lines = []
                    record = False
                    for l in lines:
                        if "Loading meta from" in l or "No meta.pkl found" in l or "number of parameters" in l:
                            record = True
                            continue
                        if record and not l.startswith("Overriding:"):
                            clean_lines.append(l)
                    
                    generated = "\n".join(clean_lines).strip()
                    if not generated:
                        generated = raw_out.strip()
                    if generated.endswith("---------------"):
                        generated = generated[:-15].strip()

                    warn_msg = None
                    if raw_prompt and raw_prompt != filtered_prompt:
                        warn_msg = f"（提示：输入提示词中部分未在当前数据集 [{dataset}] 字符词表中的字符已被自动忽略）"

                    self.send_json_response({
                        "success": True,
                        "generated_text": generated,
                        "elapsed_seconds": dt,
                        "tokens_requested": max_new_tokens,
                        "warning": warn_msg
                    })
                else:
                    self.send_json_response({
                        "success": False,
                        "error": res.stderr or res.stdout
                    })
            except subprocess.TimeoutExpired:
                self.send_json_response({"success": False, "error": "文本采样生成超时"})
            except Exception as e:
                self.send_json_response({"success": False, "error": str(e)})
            return

        elif url.path == "/api/vision/train/start":
            global vision_proc
            with vision_lock:
                if vision_state["is_training"]:
                    self.send_json_response({"success": False, "error": "视觉模型训练已在运行中"})
                    return

            model = body.get("model", "simple_cnn")
            dataset = body.get("dataset", "cifar10")
            batch_size = int(body.get("batch_size", 32))
            epochs = int(body.get("epochs", 5))
            lr = float(body.get("learning_rate", 1e-3))
            device = body.get("device", "auto")

            cmd = [
                sys.executable, "tasks/vision/train.py",
                f"--model={model}",
                f"--dataset={dataset}",
                f"--batch_size={batch_size}",
                f"--epochs={epochs}",
                f"--lr={lr}",
                f"--device={device}",
                "--out_dir=checkpoints/vision"
            ]

            t = threading.Thread(target=vision_training_worker, args=(cmd, body), daemon=True)
            t.start()
            self.send_json_response({"success": True, "message": "视觉模型训练任务已成功启动"})
            return

        elif url.path == "/api/vision/train/stop":
            stopped = False
            with vision_lock:
                if vision_proc and vision_state["is_training"]:
                    try:
                        vision_state["status"] = "stopped"
                        vision_state["logs"].append("⚠️ 收到用户停止指令，正在终止视觉训练进程...")
                        vision_proc.terminate()
                        stopped = True
                    except Exception:
                        pass
            self.send_json_response({"success": stopped, "message": "视觉训练已终止" if stopped else "没有正在运行的任务"})
            return

        elif url.path == "/api/vision/infer":
            from tasks.vision.infer import predict
            image_input = body.get("image")
            top_k = int(body.get("top_k", 5))
            if not image_input:
                self.send_json_response({"success": False, "error": "请提供需要识别的图片数据"})
                return

            res = predict(image_input, top_k=top_k)
            self.send_json_response(res)
            return

        elif url.path == "/api/datasets/create":
            ds_id = body.get("id", "").strip()
            ds_name = body.get("name", "").strip()
            ds_content = body.get("content", "")
            success, res = create_text_dataset(ds_id, ds_name, ds_content)
            if success:
                self.send_json_response({"success": True, "dataset": res, "message": f"数据集 [{ds_id}] 创建成功！已完成字符词表提取与二进制打包。"})
            else:
                self.send_json_response({"success": False, "error": res})
            return

        elif url.path == "/api/datasets/delete":
            ds_type = body.get("type", "vision")
            ds_id = body.get("id")
            if not ds_id or "/" in ds_id or "\\" in ds_id:
                self.send_json_response({"success": False, "error": "非法数据集名称"})
                return
            target_p = os.path.join(BASE_DIR, "data", "vision" if ds_type == "vision" else "", ds_id)
            if os.path.exists(target_p):
                try:
                    import shutil
                    shutil.rmtree(target_p)
                    if ds_id in vocab_cache:
                        del vocab_cache[ds_id]
                    self.send_json_response({"success": True, "message": f"数据集 {ds_id} 已删除"})
                except Exception as e:
                    self.send_json_response({"success": False, "error": f"删除失败: {e}"})
            else:
                self.send_json_response({"success": False, "error": "数据集不存在"})
            return

        elif url.path == "/api/checkpoints/delete":
            ckpt_id = body.get("id", "").strip()
            if not ckpt_id or "/" in ckpt_id or "\\" in ckpt_id:
                self.send_json_response({"success": False, "error": "非法检查点标识"})
                return
            target_p = os.path.join(BASE_DIR, "checkpoints", ckpt_id)
            if os.path.exists(target_p):
                try:
                    import shutil
                    shutil.rmtree(target_p)
                    self.send_json_response({"success": True, "message": f"检查点 {ckpt_id} 已成功清理"})
                except Exception as e:
                    self.send_json_response({"success": False, "error": f"清理失败: {e}"})
            else:
                self.send_json_response({"success": False, "error": "检查点目录不存在"})
            return

        self.send_error(404, "Not Found")

    def send_json_response(self, data):
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_cors_headers()
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))


def run_server():
    os.makedirs(WEB_DIR, exist_ok=True)
    server_address = ('', PORT)
    httpd = ThreadingHTTPServer(server_address, NanoGPTHandler)
    print("==================================================")
    print(f">> nanoStudio Multi-threaded Server Ready!")
    print(f">> Open in browser: http://127.0.0.1:{PORT}")
    print("==================================================")

    def auto_open_browser():
        time.sleep(1.0)
        try:
            import webbrowser
            webbrowser.open(f"http://127.0.0.1:{PORT}")
        except Exception:
            pass

    threading.Thread(target=auto_open_browser, daemon=True).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        httpd.server_close()

if __name__ == '__main__':
    run_server()
