import argparse
import csv
import os
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from run_vcsel_near_light_v2_train import (
    build_detector_pos,
    export_phase_height_distance,
    initialize_model,
    save_curves,
    save_history_csv,
    save_sample_outputs,
    set_trainable_phase_only,
    train,
)


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

DEFAULT_DATA_ROOT = "data/micro_devices"
DEFAULT_OUTPUT_DIR = "outputs/micro_device_d2nn_train"
DEFAULT_REPORT_PATH = "reports/micro_device_baseline_result_cn.md"
EXPECTED_CLASSES = ["0_diode", "1_bjt", "2_nmos", "3_pmos", "4_resistor"]


def parse_args():
    parser = argparse.ArgumentParser(description="Train D2NN baseline on synthetic micro-device symbols.")
    parser.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-path", type=str, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--sample-count", type=int, default=8)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--quick-train-per-class", type=int, default=40)
    parser.add_argument("--quick-val-per-class", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_data_root(data_root):
    candidates = [Path(data_root)]
    script_parent = Path(__file__).resolve().parent
    candidates.append(script_parent / data_root)
    candidates.append(script_parent.parent / data_root)
    for candidate in candidates:
        if (candidate / "train").exists() and (candidate / "val").exists():
            return candidate
    return Path(data_root)


def ensure_output_dirs(output_dir, report_path):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "samples").mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)


def build_full_datasets(data_root):
    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ]
    )
    train_dataset = torchvision.datasets.ImageFolder(data_root / "train", transform=transform)
    val_dataset = torchvision.datasets.ImageFolder(data_root / "val", transform=transform)
    return train_dataset, val_dataset


def subset_by_class(dataset, per_class):
    selected = []
    counts = {idx: 0 for idx in range(len(dataset.classes))}
    for idx, (_, label) in enumerate(dataset.samples):
        if counts[label] < per_class:
            selected.append(idx)
            counts[label] += 1
        if all(count >= per_class for count in counts.values()):
            break
    return Subset(dataset, selected)


def build_dataloaders(data_root, batch_size, num_workers, quick, quick_train_per_class, quick_val_per_class):
    train_dataset_full, val_dataset_full = build_full_datasets(data_root)
    train_dataset = train_dataset_full
    val_dataset = val_dataset_full
    if quick:
        train_dataset = subset_by_class(train_dataset_full, quick_train_per_class)
        val_dataset = subset_by_class(val_dataset_full, quick_val_per_class)

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
    return train_dataset_full, val_dataset_full, train_dataset, val_dataset, train_loader, val_loader


def dataset_count_by_class(dataset):
    counts = {name: 0 for name in dataset.classes}
    for _, label in dataset.samples:
        counts[dataset.classes[label]] += 1
    return counts


def inspect_first_batch(model, loader, device):
    images, labels = next(iter(loader))
    images_device = torch.sqrt(images.to(device).squeeze(1))
    with torch.no_grad():
        logits, intensity = model(images_device)
    return {
        "batch_shape": tuple(images.shape),
        "labels_shape": tuple(labels.shape),
        "logits_shape": tuple(logits.shape),
        "intensity_shape": tuple(intensity.shape),
        "image_min": float(images.min().item()),
        "image_max": float(images.max().item()),
    }


def read_last_history(history_path):
    if not history_path.exists():
        return None
    with history_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else None


def load_checkpoint_history(best_path):
    if not best_path.exists():
        return None
    checkpoint = torch.load(best_path, map_location="cpu")
    history = checkpoint.get("history")
    if not history:
        return None
    return history


def save_history_from_checkpoint_if_available(best_path, output_dir):
    checkpoint_history = load_checkpoint_history(best_path)
    if checkpoint_history is None:
        return None
    save_history_csv(checkpoint_history, output_dir / "training_history.csv")
    save_curves(checkpoint_history, output_dir / "training_curves.png")
    return checkpoint_history


def write_report(
    report_path,
    args,
    data_root,
    output_dir,
    train_dataset_full,
    val_dataset_full,
    effective_train_dataset,
    effective_val_dataset,
    batch_info,
    history,
    best_path,
):
    last = history[-1] if history else {}
    mode = "quick smoke test" if args.quick else "formal baseline training"
    lines = []
    lines.append("# 微电子器件结构图案 D2NN baseline 训练结果\n\n")
    lines.append(f"生成脚本：`run_micro_device_d2nn_train.py`  \n")
    lines.append(f"运行模式：`{mode}`  \n")
    lines.append(f"数据目录：`{data_root}`  \n")
    lines.append(f"输出目录：`{output_dir}`\n\n")
    lines.append("## 1. 任务目标\n\n")
    lines.append(
        "本脚本基于 V2 baseline 的 D2NN 光学传播结构，将输入数据从 VCSEL near-light 五分类替换为微电子器件结构图案五分类。原始 V2 训练脚本没有被修改。\n\n"
    )
    lines.append("## 2. 类别与数据规模\n\n")
    lines.append(f"- 类别顺序：`{train_dataset_full.classes}`\n")
    lines.append(f"- `class_to_idx`：`{train_dataset_full.class_to_idx}`\n")
    lines.append(f"- 完整 train 样本数：`{len(train_dataset_full)}`\n")
    lines.append(f"- 完整 val 样本数：`{len(val_dataset_full)}`\n")
    lines.append(f"- 本次实际 train 样本数：`{len(effective_train_dataset)}`\n")
    lines.append(f"- 本次实际 val 样本数：`{len(effective_val_dataset)}`\n\n")
    lines.append("## 3. 保留的物理逻辑\n\n")
    lines.append("- 输入图像通过 `sqrt(I)` 转为电场振幅。\n")
    lines.append("- 相位调制使用 `exp(1j * phase)`。\n")
    lines.append("- 传播使用 FFT / 角谱法。\n")
    lines.append("- 输出强度使用 `abs(E) ** 2`。\n")
    lines.append("- 5 个 detector 区域光强求和得到五分类 logits。\n\n")
    lines.append("## 4. quick 检查信息\n\n")
    lines.append(f"- batch shape：`{batch_info['batch_shape']}`\n")
    lines.append(f"- logits shape：`{batch_info['logits_shape']}`\n")
    lines.append(f"- output intensity shape：`{batch_info['intensity_shape']}`\n")
    lines.append(f"- 图像像素范围：`{batch_info['image_min']}` 到 `{batch_info['image_max']}`\n\n")
    lines.append("## 5. 本次训练结果\n\n")
    if last:
        lines.append("| 指标 | 数值 |\n")
        lines.append("|---|---:|\n")
        lines.append(f"| epoch | `{last['epoch']}` |\n")
        lines.append(f"| train_loss | `{last['train_loss']}` |\n")
        lines.append(f"| val_loss | `{last['val_loss']}` |\n")
        lines.append(f"| train_acc | `{last['train_acc']}` |\n")
        lines.append(f"| val_acc | `{last['val_acc']}` |\n")
        lines.append(f"| val_pred_hist | `{last['val_pred_hist']}` |\n\n")
    else:
        lines.append("当前尚未记录训练 history。\n\n")
    lines.append("## 6. 输出文件\n\n")
    lines.append(f"- `best.pt`：`{best_path}`\n")
    lines.append(f"- `training_history.csv`：`{output_dir / 'training_history.csv'}`\n")
    lines.append(f"- `training_curves.png`：`{output_dir / 'training_curves.png'}`\n")
    lines.append(f"- `phase_map.npy`：`{output_dir / 'phase_map.npy'}`\n")
    lines.append(f"- `height_map.npy`：`{output_dir / 'height_map.npy'}`\n")
    lines.append(f"- `trained_distances.npy`：`{output_dir / 'trained_distances.npy'}`\n")
    lines.append(f"- `samples/`：`{output_dir / 'samples'}`\n\n")
    lines.append("## 7. 下一步建议\n\n")
    lines.append(
        "如果 quick smoke test 能稳定跑通，下一步可以运行正式 30 epoch baseline，并观察是否出现类别塌缩、loss 是否下降、`val_pred_hist` 是否接近五类均衡。\n"
    )
    report_path.write_text("".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    data_root = resolve_data_root(args.data_root)
    output_dir = Path(args.output_dir)
    report_path = Path(args.report_path)
    ensure_output_dirs(output_dir, report_path)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    effective_epochs = 1 if args.quick else args.epochs

    print("Using Device:", device)
    print("Data root:", data_root)
    print("Output dir:", output_dir)
    print("Quick mode:", args.quick)
    print("Epochs:", effective_epochs)
    print("Batch size:", args.batch_size)
    print("Learning rate:", args.learning_rate)

    (
        train_dataset_full,
        val_dataset_full,
        train_dataset,
        val_dataset,
        train_loader,
        val_loader,
    ) = build_dataloaders(
        data_root,
        args.batch_size,
        args.num_workers,
        args.quick,
        args.quick_train_per_class,
        args.quick_val_per_class,
    )

    print("classes:", train_dataset_full.classes)
    print("class_to_idx:", train_dataset_full.class_to_idx)
    print("train samples:", len(train_dataset))
    print("val samples:", len(val_dataset))
    print("full train counts:", dataset_count_by_class(train_dataset_full))
    print("full val counts:", dataset_count_by_class(val_dataset_full))

    if train_dataset_full.classes != EXPECTED_CLASSES:
        raise RuntimeError(f"Unexpected classes: {train_dataset_full.classes}")
    if len(train_dataset_full.classes) != 5:
        raise RuntimeError("Micro-device task must have exactly 5 classes.")

    detector_pos = build_detector_pos()
    if len(detector_pos) != 5:
        raise RuntimeError(f"Expected 5 detector regions, got {len(detector_pos)}.")

    model = initialize_model(device, detector_pos)
    batch_info = inspect_first_batch(model, train_loader, device)
    print("batch shape:", batch_info["batch_shape"])
    print("logits shape:", batch_info["logits_shape"])
    print("intensity shape:", batch_info["intensity_shape"])
    print("image min/max:", batch_info["image_min"], batch_info["image_max"])

    params_to_update = set_trainable_phase_only(model)
    optimizer = torch.optim.Adam(params_to_update, lr=args.learning_rate)
    loss_function = torch.nn.CrossEntropyLoss().to(device)

    model, history, best_path = train(
        model,
        loss_function,
        optimizer,
        train_loader,
        val_loader,
        effective_epochs,
        device,
        output_dir,
        len(train_dataset_full.classes),
    )
    checkpoint_history = save_history_from_checkpoint_if_available(best_path, output_dir)
    report_history = checkpoint_history if checkpoint_history is not None else history

    save_sample_outputs(
        model,
        val_loader,
        device,
        output_dir,
        args.sample_count,
        detector_pos,
        train_dataset_full.classes,
    )
    export_phase_height_distance(model, output_dir)
    write_report(
        report_path,
        args,
        data_root,
        output_dir,
        train_dataset_full,
        val_dataset_full,
        train_dataset,
        val_dataset,
        batch_info,
        report_history,
        best_path,
    )

    last = report_history[-1]
    print("Final train_loss:", last["train_loss"])
    print("Final val_loss:", last["val_loss"])
    print("Final val_acc:", last["val_acc"])
    print("Report:", report_path)
    print("micro device D2NN baseline finished")


if __name__ == "__main__":
    main()
