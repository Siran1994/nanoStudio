"""
Vision Training Script
Train SimpleCNN or ResNet18 on CIFAR-10 / Custom Vision Datasets.
Emits structured output for real-time frontend monitoring:
- Train Loss, Val Loss, Top-1 Accuracy %
- Automatic checkpoint saving to out-vision/ckpt.pt
"""

import os
import sys
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from tasks.vision.model import get_vision_model, count_parameters
from tasks.vision.dataset import get_vision_dataloaders


def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * len(labels)
            _, predicted = torch.max(outputs, 1)
            total += len(labels)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / max(total, 1)
    accuracy = (correct / max(total, 1)) * 100.0
    return avg_loss, accuracy


def train():
    parser = argparse.ArgumentParser(description="Train vision classification model")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--model", type=str, default="simple_cnn", choices=["simple_cnn", "resnet18"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--out_dir", type=str, default="checkpoints/vision")
    args = parser.parse_args()

    # Device auto-detection
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print("==================================================")
    print(f"🚀 Vision Model Training Started")
    print(f"   Model:      {args.model.upper()}")
    print(f"   Dataset:    {args.dataset}")
    print(f"   Device:     {device.upper()}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))
    print(f"   Batch Size: {args.batch_size}")
    print(f"   Epochs:     {args.epochs}")
    print(f"   LearningRate: {args.lr}")
    print("==================================================")

    out_path = os.path.join(BASE_DIR, args.out_dir)
    os.makedirs(out_path, exist_ok=True)

    # 1. Load Data
    train_loader, val_loader, class_names, num_classes = get_vision_dataloaders(
        dataset_name=args.dataset,
        data_root=os.path.join(BASE_DIR, "data", "vision"),
        batch_size=args.batch_size
    )

    # 2. Build Model
    model = get_vision_model(args.model, num_classes=num_classes)
    total_p, _ = count_parameters(model)
    print(f"Model parameters: {total_p:,} ({round(total_p/1e6, 2)}M)")
    model.to(device)

    # 3. Criterion & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Initial zero-shot evaluation
    init_loss, init_acc = evaluate(model, val_loader, criterion, device)
    print(f"step 0: train loss {init_loss:.4f}, val loss {init_loss:.4f}, val acc {init_acc:.2f}%")

    global_step = 0
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        total_samples = 0
        t0 = time.time()

        for step, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(labels)
            total_samples += len(labels)
            global_step += 1

            # Log intermediate mini-batch every 25 steps
            if (step + 1) % 25 == 0 or (step + 1) == len(train_loader):
                cur_train_loss = running_loss / max(total_samples, 1)
                step_time_ms = ((time.time() - t0) / (step + 1)) * 1000
                print(f"iter {global_step}: loss {cur_train_loss:.4f}, time {step_time_ms:.1f}ms (epoch {epoch}/{args.epochs})")

        # End of epoch full evaluation
        epoch_train_loss = running_loss / max(total_samples, 1)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        epoch_time_s = round(time.time() - t0, 2)

        # Standard regex parse target line
        print(f"epoch {epoch}/{args.epochs} (step {global_step}): train loss {epoch_train_loss:.4f}, val loss {val_loss:.4f}, val acc {val_acc:.2f}%, speed {epoch_time_s}s")

        # Save checkpoint if best or at final epoch
        if val_acc >= best_acc or epoch == args.epochs:
            best_acc = val_acc
            ckpt = {
                "model_state_dict": model.state_dict(),
                "model_name": args.model,
                "num_classes": num_classes,
                "class_names": class_names,
                "best_acc": round(best_acc, 2),
                "val_loss": round(val_loss, 4),
                "epoch": epoch,
                "global_step": global_step,
                "dataset": args.dataset
            }
            ckpt_file = os.path.join(out_path, "ckpt.pt")
            torch.save(ckpt, ckpt_file)
            print(f"saving checkpoint to {args.out_dir} (Acc: {best_acc:.2f}%)")

    print(f"🎉 训练完成！最高验证集准确率: {best_acc:.2f}%，模型保存在: {os.path.join(out_path, 'ckpt.pt')}")


if __name__ == "__main__":
    train()
