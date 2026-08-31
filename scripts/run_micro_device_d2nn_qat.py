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
from run_vcsel_near_light_v2_train import (
    DNN,
    N_pixels,
    build_detector_pos,
    phase_to_etch_depth,
    save_curves,
    save_history_csv,
)


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

DEFAULT_DATA_ROOT = "data/micro_devices"
DEFAULT_OUTPUT_ROOT = "outputs/micro_device_qat"
DEFAULT_REPORT_PATH = "reports/micro_device_qat_training_result_cn.md"
DEFAULT_BASELINE_CKPT = "outputs/micro_device_d2nn_train/best.pt"
DEFAULT_PTQ_CSV = "outputs/micro_device_phase_quantization/quantization_results.csv"


def parse_args():
    parser = argparse.ArgumentParser(description="QAT training for micro-device D2NN classification.")
    parser.add_argument("--quant-levels", type=int, nargs="+", default=[2, 4, 8, 16], choices=[2, 4, 8, 16])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=str, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report-path", type=str, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--baseline-checkpoint", type=str, default=DEFAULT_BASELINE_CKPT)
    parser.add_argument("--ptq-csv", type=str, default=DEFAULT_PTQ_CSV)
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
    train_dataset = torchvision.datasets.ImageFolder(Path(data_root) / "train", transform=transform)
    val_dataset = torchvision.datasets.ImageFolder(Path(data_root) / "val", transform=transform)
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
    levels = np.linspace(
        0.0,
        phi_max * (quant_levels - 1) / quant_levels,
        quant_levels,
        dtype=np.float32,
    )
    idx = np.argmin(np.abs(phase_map[..., None] - levels), axis=-1)
    return levels[idx].astype(np.float32)


def export_maps(model, output_dir, quant_levels):
    phase_map = collect_phase_map(model)
    quantized_phase_map = quantize_phase_map(phase_map, quant_levels)
    height_map = phase_to_etch_depth(quantized_phase_map).astype(np.float32)
    np.save(output_dir / "phase_map.npy", phase_map.astype(np.float32))
    np.save(output_dir / "quantized_phase_map.npy", quantized_phase_map.astype(np.float32))
    np.save(output_dir / "height_map.npy", height_map)


def train_one_level(quant_levels, args, train_loader, val_loader, num_classes, class_names, device):
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
    best_epoch = 0

    print(f"\n========== Micro Device QAT {quant_levels}-level ==========")
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
            predicted = torch.argmax(logits, dim=1)
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
            best_epoch = epoch + 1
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "best_acc": best_acc,
                    "optimizer": optimizer.state_dict(),
                    "epoch": best_epoch,
                    "history": history,
                    "quant_levels": quant_levels,
                    "classes": class_names,
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
    class_metrics = compute_class_metrics(final_metrics["confusion_matrix"])

    save_history_csv(history, output_dir / "training_history.csv")
    save_curves(history, output_dir / "training_curves.png")
    export_maps(model, output_dir, quant_levels)
    save_confusion_matrix_csv(final_metrics["confusion_matrix"], class_names, output_dir / "confusion_matrix.csv")
    save_confusion_matrix_png(
        final_metrics["confusion_matrix"],
        class_names,
        f"Micro device QAT {quant_levels}-level confusion matrix",
        output_dir / "confusion_matrix.png",
    )
    save_metrics_csv(class_metrics, class_names, output_dir / "classification_metrics.csv")

    return {
        "quant_levels": quant_levels,
        "best_epoch": best_epoch,
        "best_acc": best_acc,
        "final_val_loss": final_metrics["val_loss"],
        "final_val_acc": final_metrics["val_acc"],
        "final_val_pred_hist": final_metrics["val_pred_hist"],
        "output_dir": str(output_dir),
    }


def read_baseline_result(checkpoint_path):
    path = Path(checkpoint_path)
    if not path.exists():
        return None
    checkpoint = torch.load(path, map_location="cpu")
    history = checkpoint.get("history", [])
    last = history[-1] if history else {}
    return {
        "epoch": checkpoint.get("epoch"),
        "best_acc": checkpoint.get("best_acc"),
        "val_loss": last.get("val_loss", ""),
        "val_acc": last.get("val_acc", checkpoint.get("best_acc", "")),
        "val_pred_hist": last.get("val_pred_hist", ""),
    }


def read_ptq_results(csv_path):
    path = Path(csv_path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as f:
        return {row["mode"]: row for row in csv.DictReader(f)}


def write_summary_csv(results, output_root):
    path = Path(output_root) / "qat_summary.csv"
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
        for row in results:
            out = dict(row)
            out["final_val_pred_hist"] = " ".join(str(x) for x in row["final_val_pred_hist"])
            writer.writerow(out)
    return path


def write_report(results, args, train_dataset, val_dataset):
    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    baseline = read_baseline_result(args.baseline_checkpoint)
    ptq = read_ptq_results(args.ptq_csv)

    lines = []
    lines.append("# Micro Device D2NN QAT 训练结果\n\n")
    lines.append(f"脚本：`run_micro_device_d2nn_qat.py`  \n")
    lines.append(f"数据集：`{args.data_root}`  \n")
    lines.append(f"输出目录：`{args.output_root}`\n\n")
    lines.append("## 1. 目标\n\n")
    lines.append("本脚本在 micro device 五分类任务上进行相位量化感知训练，对比 2/4/8/16-level QAT 与已有 PTQ 结果。训练逻辑保留 D2NN 的 sqrt 输入、复数相位调制、FFT/角谱传播、输出强度和 detector 分类。\n\n")
    lines.append("## 2. 数据集\n\n")
    lines.append(f"- train samples：`{len(train_dataset)}`\n")
    lines.append(f"- val samples：`{len(val_dataset)}`\n")
    lines.append(f"- classes：`{train_dataset.classes}`\n\n")
    lines.append("## 3. baseline / PTQ / QAT 对比\n\n")
    lines.append("| 方法 | levels | val_loss | val_acc | val_pred_hist | 说明 |\n")
    lines.append("|---|---:|---:|---:|---|---|\n")
    if baseline is not None:
        lines.append(
            f"| continuous baseline | continuous | {baseline['val_loss']} | {baseline['val_acc']} | {baseline['val_pred_hist']} | micro device baseline |\n"
        )
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
    for row in results:
        lines.append(f"- `{row['output_dir']}`：best.pt、training_history.csv、training_curves.png、phase_map.npy、quantized_phase_map.npy、height_map.npy、confusion_matrix.csv/png、classification_metrics.csv。\n")
    lines.append(f"- 汇总：`{Path(args.output_root) / 'qat_summary.csv'}`\n\n")
    lines.append("## 5. 说明\n\n")
    lines.append("本报告由当前运行结果生成。若只进行了 smoke test，则表中只包含已运行的 quant level；正式训练 2/4/8/16 后会生成完整对比表。\n")
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
    print("Output root:", output_root)
    print("Quant levels:", args.quant_levels)
    print("Batch size:", args.batch_size)
    print("Epochs:", args.epochs)
    print("Learning rate:", args.learning_rate)

    train_dataset, val_dataset, train_loader, val_loader = build_dataloaders(
        args.data_root, args.batch_size, args.num_workers
    )
    if train_dataset.classes != EXPECTED_CLASSES:
        raise RuntimeError(f"Unexpected train classes: {train_dataset.classes}")
    if val_dataset.classes != EXPECTED_CLASSES:
        raise RuntimeError(f"Unexpected val classes: {val_dataset.classes}")

    print("train classes:", train_dataset.class_to_idx)
    print("val classes:", val_dataset.class_to_idx)
    print("train samples:", len(train_dataset))
    print("val samples:", len(val_dataset))

    results = []
    for level in args.quant_levels:
        results.append(
            train_one_level(
                level,
                args,
                train_loader,
                val_loader,
                len(train_dataset.classes),
                train_dataset.classes,
                device,
            )
        )

    summary_path = write_summary_csv(results, output_root)
    write_report(results, args, train_dataset, val_dataset)
    print("Saved:", summary_path)
    print("Saved:", args.report_path)
    print("micro device QAT training finished")


if __name__ == "__main__":
    main()
