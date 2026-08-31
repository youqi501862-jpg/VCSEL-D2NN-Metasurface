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
from run_vcsel_near_light_v2_train import build_detector_pos, initialize_model


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

DEFAULT_CHECKPOINT = "outputs/micro_device_d2nn_train/best.pt"
DEFAULT_VAL_DIR = "data/micro_devices/val"
DEFAULT_OUTPUT_DIR = "outputs/micro_device_inspection"
DEFAULT_REPORT_PATH = "reports/micro_device_prediction_inspection_cn.md"


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect balanced micro-device D2NN predictions.")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--val-dir", type=str, default=DEFAULT_VAL_DIR)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-path", type=str, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--correct-per-class", type=int, default=5)
    parser.add_argument("--wrong-per-class", type=int, default=5)
    return parser.parse_args()


def ensure_dirs(output_dir, classes):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "wrong_samples").mkdir(parents=True, exist_ok=True)
    for class_name in classes:
        (output_dir / "correct_samples" / class_name).mkdir(parents=True, exist_ok=True)


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


def load_model(checkpoint_path, device):
    detector_pos = build_detector_pos()
    model = initialize_model(device, detector_pos)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint, detector_pos


def save_prediction_figure(
    output_path,
    input_img,
    intensity,
    detector_energy,
    true_idx,
    pred_idx,
    class_names,
):
    correct = true_idx == pred_idx
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)

    axes[0].imshow(input_img, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title(f"input\ntrue={class_names[true_idx]}")
    axes[0].axis("off")

    axes[1].imshow(intensity, cmap="inferno")
    axes[1].set_title(f"output intensity\npred={class_names[pred_idx]}")
    axes[1].axis("off")

    colors = ["tab:blue"] * len(detector_energy)
    colors[pred_idx] = "tab:orange" if correct else "tab:red"
    axes[2].bar(np.arange(len(detector_energy)), detector_energy, color=colors)
    axes[2].set_xticks(np.arange(len(detector_energy)))
    axes[2].set_xticklabels([str(i) for i in range(len(detector_energy))])
    axes[2].set_xlabel("class index")
    axes[2].set_ylabel("detector energy")
    axes[2].set_title("correct" if correct else "wrong")

    fig.suptitle(
        f"true={true_idx}:{class_names[true_idx]} | pred={pred_idx}:{class_names[pred_idx]} | correct={correct}",
        fontsize=10,
    )
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_confusion_matrix_csv(matrix, class_names, path):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["pred\\true"] + class_names)
        for i, row in enumerate(matrix):
            writer.writerow([class_names[i]] + [int(x) for x in row])


def save_confusion_matrix_png(matrix, class_names, path):
    fig, ax = plt.subplots(figsize=(6.3, 5.4), constrained_layout=True)
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_title("micro-device prediction confusion matrix")
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


def inspect_predictions(model, val_loader, output_dir, class_names, correct_limit, wrong_limit, device):
    num_classes = len(class_names)
    correct_saved = {idx: 0 for idx in range(num_classes)}
    wrong_saved = {idx: 0 for idx in range(num_classes)}
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    pred_hist = torch.zeros(num_classes, dtype=torch.int64)
    total = 0
    correct_total = 0

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(val_loader, desc="Inspect val")):
            field = torch.sqrt(images.to(device).squeeze(1))
            labels = labels.to(device).long()
            logits, intensity = model(field)
            predicted = torch.argmax(logits.float(), dim=1)

            pred_cpu = predicted.detach().cpu()
            label_cpu = labels.detach().cpu()
            for p, t in zip(pred_cpu.tolist(), label_cpu.tolist()):
                confusion[p, t] += 1
                pred_hist[p] += 1
                total += 1
                correct_total += int(p == t)

            for i in range(images.shape[0]):
                true_idx = int(label_cpu[i].item())
                pred_idx = int(pred_cpu[i].item())
                is_correct = true_idx == pred_idx
                if is_correct:
                    if correct_saved[true_idx] >= correct_limit:
                        continue
                    save_dir = output_dir / "correct_samples" / class_names[true_idx]
                    file_name = f"{class_names[true_idx]}_correct_{correct_saved[true_idx]:02d}_pred_{class_names[pred_idx]}.png"
                    correct_saved[true_idx] += 1
                else:
                    if wrong_saved[true_idx] >= wrong_limit:
                        continue
                    save_dir = output_dir / "wrong_samples"
                    file_name = (
                        f"true_{true_idx}_{class_names[true_idx]}__pred_{pred_idx}_{class_names[pred_idx]}"
                        f"__{wrong_saved[true_idx]:02d}.png"
                    )
                    wrong_saved[true_idx] += 1

                save_prediction_figure(
                    save_dir / file_name,
                    images[i, 0].detach().cpu().numpy(),
                    intensity[i].detach().cpu().numpy(),
                    logits[i].detach().cpu().numpy(),
                    true_idx,
                    pred_idx,
                    class_names,
                )

            if all(v >= correct_limit for v in correct_saved.values()) and all(
                v >= wrong_limit for v in wrong_saved.values()
            ):
                pass

    return {
        "confusion": confusion.numpy(),
        "pred_hist": pred_hist.tolist(),
        "total": total,
        "correct": correct_total,
        "accuracy": correct_total / max(1, total),
        "correct_saved": correct_saved,
        "wrong_saved": wrong_saved,
    }


def write_report(report_path, args, checkpoint, val_dataset, summary, metrics, class_names):
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# 微电子器件 D2NN 预测可视化检查报告\n\n")
    lines.append("## 1. 目的\n\n")
    lines.append(
        "本报告由 `inspect_micro_device_predictions.py` 生成。脚本不重新训练模型，只加载正式 checkpoint，对验证集做推理，并按真实类别均衡保存正确预测和错误预测样本，避免只保存 ImageFolder 前几个样本。\n\n"
    )
    lines.append("## 2. 输入\n\n")
    lines.append(f"- checkpoint：`{args.checkpoint}`\n")
    lines.append(f"- checkpoint epoch：`{checkpoint.get('epoch')}`\n")
    lines.append(f"- checkpoint best_acc：`{checkpoint.get('best_acc')}`\n")
    lines.append(f"- checkpoint history_len：`{len(checkpoint.get('history', []))}`\n")
    lines.append(f"- 验证集：`{args.val_dir}`\n")
    lines.append(f"- 验证集样本数：`{len(val_dataset)}`\n")
    lines.append(f"- 类别顺序：`{val_dataset.classes}`\n\n")
    lines.append("## 3. 总体结果\n\n")
    lines.append(f"- accuracy：`{summary['accuracy']}`\n")
    lines.append(f"- correct / total：`{summary['correct']} / {summary['total']}`\n")
    lines.append(f"- pred_hist：`{summary['pred_hist']}`\n")
    lines.append(f"- correct_saved：`{summary['correct_saved']}`\n")
    lines.append(f"- wrong_saved：`{summary['wrong_saved']}`\n\n")
    lines.append("## 4. 每类 precision / recall / F1\n\n")
    lines.append("| class | precision | recall | F1 | support |\n")
    lines.append("|---|---:|---:|---:|---:|\n")
    for row in metrics:
        name = class_names[row["class_idx"]]
        lines.append(
            f"| {name} | {row['precision']:.6f} | {row['recall']:.6f} | {row['f1']:.6f} | {row['support']} |\n"
        )
    lines.append("\n## 5. 输出文件\n\n")
    lines.append(f"- 正确样本：`{Path(args.output_dir) / 'correct_samples'}`\n")
    lines.append(f"- 错误样本：`{Path(args.output_dir) / 'wrong_samples'}`\n")
    lines.append(f"- confusion matrix CSV：`{Path(args.output_dir) / 'confusion_matrix.csv'}`\n")
    lines.append(f"- confusion matrix PNG：`{Path(args.output_dir) / 'confusion_matrix.png'}`\n")
    lines.append(f"- 每类指标 CSV：`{Path(args.output_dir) / 'classification_metrics.csv'}`\n\n")
    lines.append("## 6. 说明\n\n")
    lines.append("每张可视化图包含输入器件图案、输出强度图、detector 能量柱状图、true label、predicted label 和是否正确。正确样本按类别分别保存，错误样本统一保存到 `wrong_samples/`，文件名中包含 true/pred 类别。\n")
    report_path.write_text("".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    val_dir = Path(args.val_dir)
    output_dir = Path(args.output_dir)
    report_path = Path(args.report_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using Device:", device)
    print("Checkpoint:", checkpoint_path)
    print("Val dir:", val_dir)
    print("Output dir:", output_dir)

    val_dataset, val_loader = build_val_loader(val_dir, args.batch_size, args.num_workers)
    if val_dataset.classes != EXPECTED_CLASSES:
        raise RuntimeError(f"Unexpected classes: {val_dataset.classes}")
    ensure_dirs(output_dir, val_dataset.classes)

    model, checkpoint, _ = load_model(checkpoint_path, device)
    summary = inspect_predictions(
        model,
        val_loader,
        output_dir,
        val_dataset.classes,
        args.correct_per_class,
        args.wrong_per_class,
        device,
    )
    metrics = compute_class_metrics(summary["confusion"])

    save_confusion_matrix_csv(summary["confusion"], val_dataset.classes, output_dir / "confusion_matrix.csv")
    save_confusion_matrix_png(summary["confusion"], val_dataset.classes, output_dir / "confusion_matrix.png")
    save_metrics_csv(metrics, val_dataset.classes, output_dir / "classification_metrics.csv")
    write_report(report_path, args, checkpoint, val_dataset, summary, metrics, val_dataset.classes)

    print("classes:", val_dataset.classes)
    print("class_to_idx:", val_dataset.class_to_idx)
    print("accuracy:", summary["accuracy"])
    print("pred_hist:", summary["pred_hist"])
    print("correct_saved:", summary["correct_saved"])
    print("wrong_saved:", summary["wrong_saved"])
    print("Saved:", output_dir / "confusion_matrix.csv")
    print("Saved:", output_dir / "confusion_matrix.png")
    print("Saved:", output_dir / "classification_metrics.csv")
    print("Saved report:", report_path)
    print("micro device prediction inspection finished")


if __name__ == "__main__":
    main()
