import argparse
import csv
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from evaluate_phase_quantization_v2 import (
    save_confusion_matrix_csv,
    save_confusion_matrix_png,
)
from run_vcsel_near_light_v2_train import (
    DNN,
    N_pixels,
    build_detector_pos,
    phase_to_etch_depth,
    save_curves,
    save_history_csv,
)


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

DEFAULT_DATA_ROOT = "data/vcsel_near_synth"
DEFAULT_OUTPUT_ROOT = Path("outputs/vcsel_near_v2_qat")
DEFAULT_REPORT_PATH = Path("reports/v2_qat_training_result_cn.md")
DEFAULT_PTQ_CSV = Path("outputs/vcsel_near_v2_quantization/quantization_results.csv")
DEFAULT_BASELINE_CSV = Path("outputs/vcsel_near_v2_train/training_history.csv")


def parse_args():
    parser = argparse.ArgumentParser(description="QAT training for VCSEL near-light V2.")
    parser.add_argument("--quant-levels", type=int, nargs="+", default=[4, 8, 16], choices=[2, 4, 8, 16])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--report-path", type=str, default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_dataloaders(data_root, batch_size, num_workers):
    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ]
    )
    train_dataset = torchvision.datasets.ImageFolder(
        os.path.join(data_root, "train"), transform=transform
    )
    val_dataset = torchvision.datasets.ImageFolder(
        os.path.join(data_root, "val"), transform=transform
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_dataset, val_dataset, train_loader, val_loader


def initialize_qat_model(device, detector_pos, quant_levels):
    num_layers = 3
    phase = [
        torch.nn.Parameter(torch.zeros((N_pixels, N_pixels), dtype=torch.float32))
        for _ in range(num_layers)
    ]
    distance_between_layers = [0.004, 0.004, 0.004, 0.004, 0.005]
    distance = [
        torch.nn.Parameter(torch.tensor([distance_between_layers[i]], dtype=torch.float32))
        for i in range(num_layers + 2)
    ]
    model = DNN(
        phase=phase,
        num_layers=num_layers,
        wl=830e-9,
        pixel_size=2e-6,
        distance=distance,
        detector_pos=detector_pos,
        phase_mapping="sigmoid",
        quantize_during_train=True,
        quant_levels=quant_levels,
        device=device,
    ).to(device)
    return model


def set_trainable_phase_only(model):
    for p in model.parameters():
        p.requires_grad = False
    for name, p in model.named_parameters():
        if name.startswith("phase_"):
            p.requires_grad_(True)
    params_to_update = [p for p in model.parameters() if p.requires_grad]
    assert params_to_update, "No trainable phase parameters found."
    return params_to_update


def evaluate(model, val_loader, criterion, device, num_classes):
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    pred_hist = torch.zeros(num_classes, dtype=torch.int64)
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Val"):
            images = torch.sqrt(images.to(device).squeeze(1))
            labels = labels.to(device).long()
            logits, _ = model(images)
            logits = logits.to(dtype=torch.float32)
            loss = criterion(logits, labels)
            val_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)

            pred_cpu = predicted.detach().cpu()
            label_cpu = labels.detach().cpu()
            for c in range(num_classes):
                pred_hist[c] += (pred_cpu == c).sum()
            for p, t in zip(pred_cpu.tolist(), label_cpu.tolist()):
                confusion[p, t] += 1

    return {
        "val_loss": val_loss / max(1, len(val_loader)),
        "val_acc": val_correct / max(1, val_total),
        "val_pred_hist": pred_hist.tolist(),
        "confusion_matrix": confusion.numpy(),
    }


def train_one_level(quant_levels, args, train_loader, val_loader, num_classes, device):
    output_dir = Path(args.output_root) / f"{quant_levels}level"
    output_dir.mkdir(parents=True, exist_ok=True)

    detector_pos = build_detector_pos()
    model = initialize_qat_model(device, detector_pos, quant_levels)
    params_to_update = set_trainable_phase_only(model)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(params_to_update, lr=args.learning_rate)

    history = []
    best_acc = -1.0
    best_state = None
    best_optimizer = None
    best_epoch = 0

    print(f"\n========== QAT {quant_levels}-level ==========")
    print("Output dir:", output_dir)
    print("Trainable params:", [name for name, p in model.named_parameters() if p.requires_grad])

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for images, labels in tqdm(train_loader, desc=f"Train {epoch + 1}/{args.epochs} [{quant_levels}level]"):
            images = torch.sqrt(images.to(device).squeeze(1))
            labels = labels.to(device).long()

            optimizer.zero_grad()
            logits, _ = model(images)
            logits = logits.to(dtype=torch.float32)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            running_correct += (predicted == labels).sum().item()
            running_total += labels.size(0)

        train_loss = running_loss / max(1, len(train_loader))
        train_acc = running_correct / max(1, running_total)
        val_metrics = evaluate(model, val_loader, criterion, device, num_classes)

        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_metrics["val_loss"],
            "train_acc": train_acc,
            "val_acc": val_metrics["val_acc"],
            "val_pred_hist": val_metrics["val_pred_hist"],
        }
        history.append(row)

        if val_metrics["val_acc"] > best_acc:
            best_acc = val_metrics["val_acc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_optimizer = optimizer.state_dict()
            best_epoch = epoch + 1
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "best_acc": best_acc,
                    "optimizer": optimizer.state_dict(),
                    "epoch": best_epoch,
                    "history": history,
                    "quant_levels": quant_levels,
                },
                output_dir / "best.pt",
            )

        print(
            f"Epoch={epoch + 1}/{args.epochs} "
            f"train_loss={train_loss:.6g}, val_loss={val_metrics['val_loss']:.6g}"
        )
        print(f"train_acc={train_acc:.6f}, val_acc={val_metrics['val_acc']:.6f}")
        print("val pred hist:", val_metrics["val_pred_hist"])
        print("-----------------------")

    if best_state is not None:
        model.load_state_dict(best_state)
    final_metrics = evaluate(model, val_loader, criterion, device, num_classes)

    save_history_csv(history, output_dir / "training_history.csv")
    save_curves(history, output_dir / "training_curves.png")
    export_maps(model, output_dir, quant_levels)
    save_confusion_matrix_csv(final_metrics["confusion_matrix"], output_dir / "confusion_matrix.csv")
    save_confusion_matrix_png(
        final_metrics["confusion_matrix"],
        f"QAT {quant_levels}-level confusion matrix",
        output_dir / "confusion_matrix.png",
    )

    return {
        "quant_levels": quant_levels,
        "best_epoch": best_epoch,
        "best_acc": best_acc,
        "final_val_loss": final_metrics["val_loss"],
        "final_val_acc": final_metrics["val_acc"],
        "final_val_pred_hist": final_metrics["val_pred_hist"],
        "output_dir": str(output_dir),
    }


def collect_phase_map(model):
    params = dict(model.named_parameters())
    keys = sorted(
        [k for k in params.keys() if k.startswith("phase_")],
        key=lambda x: int(x.split("_", 1)[1]),
    )
    phase_maps = []
    for key in keys:
        raw = params[key].detach().cpu()
        phi = np.pi * torch.sigmoid(raw)
        phase_maps.append(phi.numpy().astype(np.float32))
    return np.stack(phase_maps, axis=0)


def quantize_phase_map(phase_map, quant_levels):
    phi_max = np.pi
    levels = np.linspace(0.0, phi_max * (quant_levels - 1) / quant_levels, quant_levels, dtype=np.float32)
    idx = np.argmin(np.abs(phase_map[..., None] - levels), axis=-1)
    return levels[idx].astype(np.float32)


def export_maps(model, output_dir, quant_levels):
    phase_map = collect_phase_map(model)
    quantized_phase_map = quantize_phase_map(phase_map, quant_levels)
    height_map = phase_to_etch_depth(quantized_phase_map).astype(np.float32)
    np.save(output_dir / "phase_map.npy", phase_map.astype(np.float32))
    np.save(output_dir / "quantized_phase_map.npy", quantized_phase_map.astype(np.float32))
    np.save(output_dir / "height_map.npy", height_map)


def read_last_history(csv_path):
    if not csv_path.exists():
        return None
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else None


def read_ptq_results(csv_path):
    if not csv_path.exists():
        return {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return {row["mode"]: row for row in rows}


def write_summary_csv(results, output_root):
    path = Path(output_root) / "qat_summary.csv"
    merged = {}
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if not row.get("quant_levels"):
                    continue
                level = int(row["quant_levels"])
                merged[level] = {
                    "quant_levels": level,
                    "best_epoch": int(row["best_epoch"]),
                    "best_acc": float(row["best_acc"]),
                    "final_val_loss": float(row["final_val_loss"]),
                    "final_val_acc": float(row["final_val_acc"]),
                    "final_val_pred_hist": [int(x) for x in row["final_val_pred_hist"].split()],
                    "output_dir": row["output_dir"],
                }
    for row in results:
        merged[int(row["quant_levels"])] = row
    merged_results = [merged[level] for level in sorted(merged)]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "quant_levels",
                "best_epoch",
                "best_acc",
                "final_val_loss",
                "final_val_acc",
                "final_val_pred_hist",
                "output_dir",
            ],
        )
        writer.writeheader()
        for row in merged_results:
            out = dict(row)
            out["final_val_pred_hist"] = " ".join(str(x) for x in row["final_val_pred_hist"])
            writer.writerow(out)
    return path, merged_results


def write_report(results, report_path):
    baseline = read_last_history(DEFAULT_BASELINE_CSV)
    ptq = read_ptq_results(DEFAULT_PTQ_CSV)
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# V2 QAT 训练结果\n\n")
    lines.append("生成时间：2026-05-12  \n")
    lines.append("脚本：`run_vcsel_near_light_v2_qat.py`  \n")
    lines.append("输出目录：`outputs/vcsel_near_v2_qat/`\n\n")
    lines.append("## 1. 目标\n\n")
    lines.append("本脚本用于比较 4-level、8-level、16-level 的 QAT 训练效果，并与连续相位 baseline 和 PTQ 相位量化结果对比。QAT 前向传播使用量化相位，反向传播使用 STE 近似传递梯度。\n\n")
    lines.append("## 2. 核心物理逻辑\n\n")
    lines.append("- 输入图像使用 `sqrt` 转为电场振幅。\n")
    lines.append("- 相位调制使用 `exp(1j * phase)`。\n")
    lines.append("- 传播使用 FFT / 角谱法。\n")
    lines.append("- 输出强度使用 `abs(E)**2`。\n")
    lines.append("- detector 区域求和得到分类 logits。\n\n")
    lines.append("## 3. 对比结果\n\n")
    lines.append("| 方法 | levels | val_loss | val_acc | val_pred_hist | 备注 |\n")
    lines.append("|---|---:|---:|---:|---|---|\n")
    if baseline is not None:
        lines.append(
            f"| continuous baseline | continuous | {baseline['val_loss']} | {baseline['val_acc']} | {baseline['val_pred_hist']} | 第 {baseline['epoch']} epoch |\n"
        )
    if ptq:
        for mode in ["2level", "4level", "8level", "16level"]:
            row = ptq.get(mode)
            if row:
                lines.append(
                    f"| PTQ | {row['levels']} | {row['val_loss']} | {row['val_acc']} | {row['val_pred_hist']} | 训练后量化 |\n"
                )
    for row in results:
        lines.append(
            f"| QAT | {row['quant_levels']} | {row['final_val_loss']:.8g} | {row['final_val_acc']:.6f} | {row['final_val_pred_hist']} | best_acc={row['best_acc']:.6f}, best_epoch={row['best_epoch']} |\n"
        )
    lines.append("\n## 4. 输出文件\n\n")
    lines.append("每个量化级数会输出到独立目录：\n\n")
    for row in results:
        lines.append(f"- `outputs/vcsel_near_v2_qat/{row['quant_levels']}level/`\n")
    lines.append("\n")
    lines.append("每个目录包含：`best.pt`、`training_history.csv`、`training_curves.png`、`phase_map.npy`、`quantized_phase_map.npy`、`height_map.npy`、`confusion_matrix.png`。\n\n")
    lines.append("## 5. 下一步\n\n")
    lines.append("如果 QAT 4-level 明显优于 PTQ 4-level，可将 QAT 作为后续硬件感知训练主线；如果 8/16-level 已接近 continuous baseline，可优先把这些级数接到 height map、误差扰动和 COMSOL FEM 纳米柱映射流程。\n")
    report_path.write_text("".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using Device:", device)
    print("Data root:", args.data_root)
    print("Quant levels:", args.quant_levels)
    print("Batch size:", args.batch_size)
    print("Epochs:", args.epochs)
    print("Learning rate:", args.learning_rate)

    train_dataset, val_dataset, train_loader, val_loader = build_dataloaders(
        args.data_root, args.batch_size, args.num_workers
    )
    num_classes = len(train_dataset.classes)
    print("train classes:", train_dataset.class_to_idx)
    print("val classes:", val_dataset.class_to_idx)
    print("train samples:", len(train_dataset))
    print("val samples:", len(val_dataset))

    results = []
    for level in args.quant_levels:
        results.append(train_one_level(level, args, train_loader, val_loader, num_classes, device))

    summary_path, merged_results = write_summary_csv(results, output_root)
    write_report(merged_results, args.report_path)
    print("Saved:", summary_path)
    print("Saved:", args.report_path)
    print("QAT training finished")


if __name__ == "__main__":
    main()
