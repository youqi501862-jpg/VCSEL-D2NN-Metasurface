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

from run_vcsel_near_light_v2_train import (
    build_detector_pos,
    collect_phase_map,
    initialize_model,
)


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

DEFAULT_CHECKPOINT = "outputs/vcsel_near_v2_train/best.pt"
DEFAULT_VAL_DIR = "data/vcsel_near_synth/val"
DEFAULT_OUTPUT_DIR = "outputs/vcsel_near_v2_quantization"
DEFAULT_REPORT_PATH = "reports/v2_phase_quantization_result_cn.md"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate V2 phase quantization.")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--val-dir", type=str, default=DEFAULT_VAL_DIR)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-path", type=str, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def ensure_dirs(output_dir, report_path):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "confusion_matrices").mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)


def build_val_loader(val_dir, batch_size, num_workers):
    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ]
    )
    val_dataset = torchvision.datasets.ImageFolder(val_dir, transform=transform)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return val_dataset, val_loader


def load_best_model(checkpoint_path, device):
    detector_pos = build_detector_pos()
    model = initialize_model(device, detector_pos)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def new_model_from_state(state_dict, device):
    detector_pos = build_detector_pos()
    model = initialize_model(device, detector_pos)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def quantize_phase_tensor(phi, levels):
    diff = torch.abs(phi.unsqueeze(-1) - levels)
    idx = torch.argmin(diff, dim=-1)
    return levels[idx]


def raw_param_from_phase(phi, phi_max=np.pi, eps=1e-6):
    ratio = torch.clamp(phi / phi_max, eps, 1.0 - eps)
    return torch.log(ratio / (1.0 - ratio))


def make_quantized_model(state_dict, device, n_levels):
    model_q = new_model_from_state(state_dict, device)
    phi_max = float(getattr(model_q, "phi_max", np.pi))
    device = next(model_q.parameters()).device
    levels = torch.linspace(
        0.0,
        phi_max * (n_levels - 1) / n_levels,
        steps=n_levels,
        device=device,
        dtype=torch.float32,
    )

    with torch.no_grad():
        for name, param in model_q.named_parameters():
            if not name.startswith("phase_"):
                continue
            raw = param.detach().float()
            phi = phi_max * torch.sigmoid(raw)
            phi_q = quantize_phase_tensor(phi, levels)
            param.copy_(raw_param_from_phase(phi_q, phi_max=phi_max).to(param.dtype))

    model_q.eval()
    return model_q


def evaluate_model(model, val_loader, criterion, device, num_classes):
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    pred_hist = torch.zeros(num_classes, dtype=torch.int64)
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Eval"):
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


def save_confusion_matrix_csv(matrix, path):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["pred\\true"] + [str(i) for i in range(matrix.shape[1])])
        for i, row in enumerate(matrix):
            writer.writerow([str(i)] + [int(x) for x in row])


def save_confusion_matrix_png(matrix, title, path):
    fig, ax = plt.subplots(figsize=(5.5, 4.8), constrained_layout=True)
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("True label")
    ax.set_ylabel("Predicted label")
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_yticks(np.arange(matrix.shape[0]))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    thresh = matrix.max() / 2 if matrix.max() > 0 else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                int(matrix[i, j]),
                ha="center",
                va="center",
                color="white" if matrix[i, j] > thresh else "black",
            )
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_results_csv(results, path):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["mode", "levels", "val_loss", "val_acc", "val_pred_hist"],
        )
        writer.writeheader()
        for row in results:
            writer.writerow(
                {
                    "mode": row["mode"],
                    "levels": row["levels"],
                    "val_loss": row["val_loss"],
                    "val_acc": row["val_acc"],
                    "val_pred_hist": " ".join(str(x) for x in row["val_pred_hist"]),
                }
            )


def save_accuracy_curve(results, path):
    labels = [row["mode"] for row in results]
    acc = [row["val_acc"] for row in results]
    fig, ax = plt.subplots(figsize=(7, 4.2), constrained_layout=True)
    ax.plot(labels, acc, marker="o")
    ax.set_xlabel("phase mode")
    ax.set_ylabel("val accuracy")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    for x, y in zip(labels, acc):
        ax.text(x, y + 0.02, f"{y:.4f}", ha="center", fontsize=8)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_phase_maps(model, mode_name, output_dir):
    phase_map = collect_phase_map(model)
    np.save(output_dir / f"{mode_name}_phase_map.npy", phase_map.astype(np.float32))
    return phase_map


def write_report(results, checkpoint, val_dataset, output_dir, report_path):
    final_lines = []
    final_lines.append("# V2 相位量化评估结果\n")
    final_lines.append("生成时间：2026-05-12  \n")
    final_lines.append(f"输入 checkpoint：`{DEFAULT_CHECKPOINT}`  \n")
    final_lines.append(f"验证集：`{DEFAULT_VAL_DIR}`  \n")
    final_lines.append(f"输出目录：`{output_dir}`\n")
    final_lines.append("## 1. 评估目的\n")
    final_lines.append(
        "本次评估在不重新训练模型的前提下，加载已经训练完成的 V2 baseline `best.pt`，比较连续相位和不同离散相位级数下的验证集分类性能。\n"
    )
    final_lines.append("## 2. 数据集与模型\n")
    final_lines.append(f"- 验证集样本数：`{len(val_dataset)}`\n")
    final_lines.append(f"- 类别顺序：`{val_dataset.classes}`\n")
    final_lines.append(f"- checkpoint best_acc：`{checkpoint.get('best_acc')}`\n")
    final_lines.append(f"- checkpoint epoch：`{checkpoint.get('epoch')}`\n")
    final_lines.append("## 3. 量化结果\n")
    final_lines.append("| 模式 | levels | val_loss | val_acc | val_pred_hist |\n")
    final_lines.append("|---|---:|---:|---:|---|\n")
    for row in results:
        final_lines.append(
            f"| {row['mode']} | {row['levels']} | {row['val_loss']:.8g} | {row['val_acc']:.6f} | {row['val_pred_hist']} |\n"
        )
    final_lines.append("\n## 4. 输出文件\n")
    final_lines.append("- `quantization_results.csv`：各相位模式的 loss、accuracy、预测直方图。\n")
    final_lines.append("- `quantization_accuracy_curve.png`：不同量化级数下的验证准确率曲线。\n")
    final_lines.append("- `continuous_phase_map.npy`：连续相位图。\n")
    final_lines.append("- `2level_phase_map.npy`、`4level_phase_map.npy`、`8level_phase_map.npy`、`16level_phase_map.npy`：不同级数量化后重新写回模型参数并导出的相位图。\n")
    final_lines.append("- `confusion_matrices/`：每种模式的混淆矩阵 CSV 和 PNG。\n")
    final_lines.append("\n## 5. 结论与下一步\n")
    best_row = max(results, key=lambda r: r["val_acc"])
    final_lines.append(
        f"当前评估中最高验证准确率来自 `{best_row['mode']}`，val_acc=`{best_row['val_acc']:.6f}`。如果低级数量化精度明显下降，下一步应进入 QAT，让模型在训练阶段适应离散相位级别。\n"
    )
    final_lines.append(
        "后续建议继续评估相位量化后的 `height_map` 映射、加工误差扰动、层间对准误差，以及 COMSOL FEM 纳米柱查找表映射后的性能。\n"
    )
    report_path.write_text("".join(final_lines), encoding="utf-8")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    report_path = Path(args.report_path)
    ensure_dirs(output_dir, report_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using Device:", device)
    print("Checkpoint:", args.checkpoint)
    print("Val dir:", args.val_dir)
    print("Output dir:", output_dir)

    val_dataset, val_loader = build_val_loader(args.val_dir, args.batch_size, args.num_workers)
    num_classes = len(val_dataset.classes)
    print("val classes:", val_dataset.classes)
    print("val class_to_idx:", val_dataset.class_to_idx)
    print("val samples:", len(val_dataset))

    base_model, checkpoint = load_best_model(args.checkpoint, device)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    modes = [
        ("continuous", None),
        ("2level", 2),
        ("4level", 4),
        ("8level", 8),
        ("16level", 16),
    ]

    results = []
    for mode_name, levels in modes:
        print(f"\n=== Evaluating {mode_name} ===")
        if levels is None:
            model_eval = new_model_from_state(checkpoint["state_dict"], device)
            level_value = "continuous"
        else:
            model_eval = make_quantized_model(checkpoint["state_dict"], device, levels)
            level_value = str(levels)

        phase_map = save_phase_maps(model_eval, mode_name, output_dir)
        print(
            f"phase_map {mode_name}: shape={phase_map.shape}, "
            f"min={float(np.min(phase_map)):.6f}, max={float(np.max(phase_map)):.6f}"
        )

        metrics = evaluate_model(model_eval, val_loader, criterion, device, num_classes)
        matrix = metrics["confusion_matrix"]
        cm_csv = output_dir / "confusion_matrices" / f"{mode_name}_confusion_matrix.csv"
        cm_png = output_dir / "confusion_matrices" / f"{mode_name}_confusion_matrix.png"
        save_confusion_matrix_csv(matrix, cm_csv)
        save_confusion_matrix_png(matrix, f"{mode_name} confusion matrix", cm_png)

        row = {
            "mode": mode_name,
            "levels": level_value,
            "val_loss": metrics["val_loss"],
            "val_acc": metrics["val_acc"],
            "val_pred_hist": metrics["val_pred_hist"],
            "confusion_matrix": matrix,
        }
        results.append(row)
        print(
            f"{mode_name}: val_loss={metrics['val_loss']:.8g}, "
            f"val_acc={metrics['val_acc']:.6f}, "
            f"val_pred_hist={metrics['val_pred_hist']}"
        )
        print("confusion_matrix rows=pred, cols=true:")
        print(matrix)

    save_results_csv(results, output_dir / "quantization_results.csv")
    save_accuracy_curve(results, output_dir / "quantization_accuracy_curve.png")
    write_report(results, checkpoint, val_dataset, output_dir, report_path)

    print("\nSaved:", output_dir / "quantization_results.csv")
    print("Saved:", output_dir / "quantization_accuracy_curve.png")
    print("Saved:", report_path)
    print("phase quantization evaluation finished")


if __name__ == "__main__":
    main()
