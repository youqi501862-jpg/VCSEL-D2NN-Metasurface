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

from run_micro_device_d2nn_train import EXPECTED_CLASSES
from run_vcsel_near_light_v2_train import build_detector_pos, collect_phase_map, initialize_model


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

DEFAULT_CHECKPOINT = "outputs/micro_device_d2nn_train/best.pt"
DEFAULT_VAL_DIR = "data/micro_devices/val"
DEFAULT_OUTPUT_DIR = "outputs/micro_device_phase_quantization"
DEFAULT_REPORT_PATH = "reports/micro_device_phase_quantization_result_cn.md"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate micro-device D2NN phase quantization.")
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
    (output_dir / "classification_metrics").mkdir(parents=True, exist_ok=True)
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
    param_device = next(model_q.parameters()).device
    levels = torch.linspace(
        0.0,
        phi_max * (n_levels - 1) / n_levels,
        steps=n_levels,
        device=param_device,
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

            predicted = torch.argmax(logits, dim=1)
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


def compute_class_metrics(confusion):
    metrics = []
    eps = 1e-12
    for idx in range(confusion.shape[0]):
        tp = float(confusion[idx, idx])
        fp = float(confusion[idx, :].sum() - confusion[idx, idx])
        fn = float(confusion[:, idx].sum() - confusion[idx, idx])
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2.0 * precision * recall / (precision + recall + eps)
        metrics.append(
            {
                "class_idx": idx,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": int(confusion[:, idx].sum()),
            }
        )
    return metrics


def save_confusion_matrix_csv(matrix, class_names, path):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["pred\\true"] + class_names)
        for i, row in enumerate(matrix):
            writer.writerow([class_names[i]] + [int(x) for x in row])


def save_confusion_matrix_png(matrix, class_names, title, path):
    fig, ax = plt.subplots(figsize=(6.3, 5.4), constrained_layout=True)
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("True label")
    ax.set_ylabel("Predicted label")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticklabels(class_names)
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
                fontsize=8,
            )
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_metrics_csv(metrics, class_names, path):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["class_idx", "class_name", "precision", "recall", "f1", "support"],
        )
        writer.writeheader()
        for row in metrics:
            out = dict(row)
            out["class_name"] = class_names[row["class_idx"]]
            writer.writerow(out)


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
    fig, ax = plt.subplots(figsize=(7.2, 4.3), constrained_layout=True)
    ax.plot(labels, acc, marker="o")
    ax.set_xlabel("phase mode")
    ax.set_ylabel("val accuracy")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    for x, y in zip(labels, acc):
        ax.text(x, y + 0.02, f"{y:.4f}", ha="center", fontsize=8)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_report(report_path, args, checkpoint, val_dataset, results, metrics_by_mode):
    lines = []
    lines.append("# 微电子器件 D2NN PTQ 相位量化评估结果\n\n")
    lines.append("## 1. 评估目标\n\n")
    lines.append(
        "本评估不重新训练模型，只加载微电子器件 D2NN baseline checkpoint，比较连续相位和不同离散相位级数下的验证集分类性能。\n\n"
    )
    lines.append("## 2. 输入\n\n")
    lines.append(f"- checkpoint：`{args.checkpoint}`\n")
    lines.append(f"- checkpoint epoch：`{checkpoint.get('epoch')}`\n")
    lines.append(f"- checkpoint best_acc：`{checkpoint.get('best_acc')}`\n")
    lines.append(f"- checkpoint history_len：`{len(checkpoint.get('history', []))}`\n")
    lines.append(f"- 验证集：`{args.val_dir}`\n")
    lines.append(f"- 验证集样本数：`{len(val_dataset)}`\n")
    lines.append(f"- 类别顺序：`{val_dataset.classes}`\n\n")
    lines.append("## 3. 保留的核心物理逻辑\n\n")
    lines.append("- 输入图像通过 `sqrt(I)` 转为电场振幅。\n")
    lines.append("- 相位调制使用 `exp(1j * phase)`。\n")
    lines.append("- 传播使用 FFT / 角谱法。\n")
    lines.append("- 输出强度使用 `abs(E) ** 2`。\n")
    lines.append("- detector 区域能量求和得到五分类 logits。\n\n")
    lines.append("## 4. 量化评估结果\n\n")
    lines.append("| mode | levels | val_loss | val_acc | val_pred_hist |\n")
    lines.append("|---|---:|---:|---:|---|\n")
    for row in results:
        lines.append(
            f"| {row['mode']} | {row['levels']} | {row['val_loss']:.8g} | "
            f"{row['val_acc']:.6f} | `{row['val_pred_hist']}` |\n"
        )
    lines.append("\n## 5. 每类 precision / recall / F1\n\n")
    for row in results:
        mode = row["mode"]
        lines.append(f"### {mode}\n\n")
        lines.append("| class | precision | recall | F1 | support |\n")
        lines.append("|---|---:|---:|---:|---:|\n")
        for metric in metrics_by_mode[mode]:
            class_name = val_dataset.classes[metric["class_idx"]]
            lines.append(
                f"| {class_name} | {metric['precision']:.6f} | "
                f"{metric['recall']:.6f} | {metric['f1']:.6f} | {metric['support']} |\n"
            )
        lines.append("\n")
    lines.append("## 6. 输出文件\n\n")
    lines.append(f"- `quantization_results.csv`：`{Path(args.output_dir) / 'quantization_results.csv'}`\n")
    lines.append(f"- `quantization_accuracy_curve.png`：`{Path(args.output_dir) / 'quantization_accuracy_curve.png'}`\n")
    lines.append(f"- `confusion_matrices/`：每种相位模式的混淆矩阵 CSV 和 PNG。\n")
    lines.append(f"- `classification_metrics/`：每种相位模式的 precision、recall、F1。\n")
    lines.append(f"- `continuous_phase_map.npy`、`2level_phase_map.npy`、`4level_phase_map.npy`、`8level_phase_map.npy`、`16level_phase_map.npy`。\n\n")
    lines.append("## 7. 说明\n\n")
    lines.append(
        "PTQ 是训练后量化，因此低级数相位如果性能明显下降，下一步应为 micro device 任务增加 QAT 训练，而不是直接使用低位 PTQ 结果。\n"
    )
    report_path.write_text("".join(lines), encoding="utf-8")


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
    if val_dataset.classes != EXPECTED_CLASSES:
        raise RuntimeError(f"Unexpected classes: {val_dataset.classes}")
    print("classes:", val_dataset.classes)
    print("class_to_idx:", val_dataset.class_to_idx)
    print("val samples:", len(val_dataset))

    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint["state_dict"]
    criterion = torch.nn.CrossEntropyLoss().to(device)
    modes = [
        ("continuous", "continuous", None),
        ("2level", "2", 2),
        ("4level", "4", 4),
        ("8level", "8", 8),
        ("16level", "16", 16),
    ]

    results = []
    metrics_by_mode = {}
    for mode_name, levels_label, levels in modes:
        print(f"\n========== {mode_name} ==========")
        if levels is None:
            model_eval = new_model_from_state(state_dict, device)
        else:
            model_eval = make_quantized_model(state_dict, device, levels)
        metrics = evaluate_model(model_eval, val_loader, criterion, device, len(val_dataset.classes))
        phase_map = collect_phase_map(model_eval)
        np.save(output_dir / f"{mode_name}_phase_map.npy", phase_map.astype(np.float32))

        class_metrics = compute_class_metrics(metrics["confusion_matrix"])
        metrics_by_mode[mode_name] = class_metrics
        save_confusion_matrix_csv(
            metrics["confusion_matrix"],
            val_dataset.classes,
            output_dir / "confusion_matrices" / f"{mode_name}_confusion_matrix.csv",
        )
        save_confusion_matrix_png(
            metrics["confusion_matrix"],
            val_dataset.classes,
            f"{mode_name} confusion matrix",
            output_dir / "confusion_matrices" / f"{mode_name}_confusion_matrix.png",
        )
        save_metrics_csv(
            class_metrics,
            val_dataset.classes,
            output_dir / "classification_metrics" / f"{mode_name}_classification_metrics.csv",
        )

        row = {
            "mode": mode_name,
            "levels": levels_label,
            "val_loss": metrics["val_loss"],
            "val_acc": metrics["val_acc"],
            "val_pred_hist": metrics["val_pred_hist"],
        }
        results.append(row)
        print(
            f"{mode_name}: val_loss={row['val_loss']:.8g}, "
            f"val_acc={row['val_acc']:.6f}, val_pred_hist={row['val_pred_hist']}"
        )

    save_results_csv(results, output_dir / "quantization_results.csv")
    save_accuracy_curve(results, output_dir / "quantization_accuracy_curve.png")
    save_report(report_path, args, checkpoint, val_dataset, results, metrics_by_mode)

    print("\nSaved:", output_dir / "quantization_results.csv")
    print("Saved:", output_dir / "quantization_accuracy_curve.png")
    print("Saved report:", report_path)
    print("micro device phase quantization evaluation finished")


if __name__ == "__main__":
    main()
