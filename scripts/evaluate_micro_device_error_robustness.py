import argparse
import csv
import json
import os
import shutil
from datetime import datetime
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
    T_OXIDE,
    build_detector_pos,
    collect_phase_map,
    initialize_model,
)


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

DEFAULT_VAL_DIR = "data/micro_devices/val"
DEFAULT_BASELINE_CKPT = "outputs/micro_device_d2nn_train/best.pt"
DEFAULT_QAT_ROOT = "outputs/micro_device_qat"
DEFAULT_OUTPUT_DIR = "outputs/micro_device_error_robustness"
DEFAULT_REPORT_PATH = "reports/micro_device_error_robustness_result_cn.md"

PHI_MAX = np.pi
HEIGHT_PHASE_MAX = 3 * np.pi / 4


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate micro-device D2NN robustness under phase, height, and alignment errors."
    )
    parser.add_argument("--val-dir", type=str, default=DEFAULT_VAL_DIR)
    parser.add_argument("--baseline-checkpoint", type=str, default=DEFAULT_BASELINE_CKPT)
    parser.add_argument("--qat-root", type=str, default=DEFAULT_QAT_ROOT)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-path", type=str, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--mc-runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--phase-noise-std", type=float, nargs="+", default=[0.0, 0.02, 0.05, 0.10, 0.20])
    parser.add_argument("--height-noise-std-nm", type=float, nargs="+", default=[0.0, 5.0, 10.0, 20.0, 40.0])
    parser.add_argument("--alignment-shift-px", type=int, nargs="+", default=[0, 1, 2, 4])
    return parser.parse_args()


def ensure_dirs(output_dir, report_path):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "curves").mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)


def backup_existing_file(path, backup_dir):
    path = Path(path)
    if not path.exists():
        return None
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"{path.stem}_{stamp}{path.suffix}"
    shutil.copy2(path, backup_path)
    return backup_path


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


def quantize_phase_map(phase_map, levels):
    phase_map = np.asarray(phase_map, dtype=np.float32)
    phase_levels = np.linspace(
        0.0,
        PHI_MAX * (levels - 1) / levels,
        num=levels,
        dtype=np.float32,
    )
    idx = np.argmin(np.abs(phase_map[..., None] - phase_levels), axis=-1)
    return phase_levels[idx].astype(np.float32)


def raw_param_from_phase_np(phase_map, eps=1e-6):
    ratio = np.clip(phase_map / PHI_MAX, eps, 1.0 - eps)
    return np.log(ratio / (1.0 - ratio)).astype(np.float32)


def make_model_from_phase_map(phase_map, reference_state_dict, device):
    detector_pos = build_detector_pos()
    model = initialize_model(device, detector_pos)
    model.load_state_dict(reference_state_dict, strict=False)
    phase_map = np.clip(np.asarray(phase_map, dtype=np.float32), 0.0, PHI_MAX)
    raw_map = raw_param_from_phase_np(phase_map)
    with torch.no_grad():
        for layer_idx in range(raw_map.shape[0]):
            param = getattr(model, f"phase_{layer_idx}")
            param.copy_(torch.from_numpy(raw_map[layer_idx]).to(device=device, dtype=param.dtype))
    model.quantize_during_train = False
    model.to(device)
    model.eval()
    return model


def initialize_qat_model_for_loading(device, detector_pos, quant_levels):
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


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device)
    if "state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint does not contain state_dict: {path}")
    return checkpoint


def load_continuous_phase_map(checkpoint_path, device):
    detector_pos = build_detector_pos()
    model = initialize_model(device, detector_pos)
    checkpoint = load_checkpoint(checkpoint_path, device)
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model.to(device)
    model.eval()
    return collect_phase_map(model), checkpoint["state_dict"]


def load_qat_phase_map(checkpoint_path, device, quant_levels):
    detector_pos = build_detector_pos()
    model = initialize_qat_model_for_loading(device, detector_pos, quant_levels)
    checkpoint = load_checkpoint(checkpoint_path, device)
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model.to(device)
    model.eval()
    continuous_phase = collect_phase_map(model)
    effective_phase = quantize_phase_map(continuous_phase, quant_levels)
    return effective_phase, checkpoint["state_dict"]


def build_model_specs(args, device):
    specs = []
    baseline_path = Path(args.baseline_checkpoint)
    if baseline_path.exists():
        baseline_phase, baseline_state = load_continuous_phase_map(baseline_path, device)
        specs.append(
            {
                "model_name": "continuous_baseline",
                "family": "continuous",
                "levels": "continuous",
                "phase_map": baseline_phase,
                "state_dict": baseline_state,
                "checkpoint": str(baseline_path),
            }
        )
        for levels in [2, 4, 8, 16]:
            specs.append(
                {
                    "model_name": f"ptq_{levels}level",
                    "family": "PTQ",
                    "levels": str(levels),
                    "phase_map": quantize_phase_map(baseline_phase, levels),
                    "state_dict": baseline_state,
                    "checkpoint": str(baseline_path),
                }
            )
    else:
        print(f"WARNING: baseline checkpoint not found: {baseline_path}")

    for levels in [2, 4, 8, 16]:
        qat_path = Path(args.qat_root) / f"{levels}level" / "best.pt"
        if not qat_path.exists():
            print(f"WARNING: QAT checkpoint not found, skipped: {qat_path}")
            continue
        phase_map, state_dict = load_qat_phase_map(qat_path, device, levels)
        specs.append(
            {
                "model_name": f"qat_{levels}level",
                "family": "QAT",
                "levels": str(levels),
                "phase_map": phase_map,
                "state_dict": state_dict,
                "checkpoint": str(qat_path),
            }
        )
    return specs


def perturb_phase_map(base_phase_map, perturbation, strength, rng):
    phase_map = np.array(base_phase_map, dtype=np.float32, copy=True)
    if perturbation == "phase_noise":
        if strength > 0:
            phase_map += rng.normal(0.0, strength, size=phase_map.shape).astype(np.float32)
    elif perturbation == "height_noise":
        if strength > 0:
            height_std_m = strength * 1e-9
            height_noise = rng.normal(0.0, height_std_m, size=phase_map.shape).astype(np.float32)
            phase_map += (-(HEIGHT_PHASE_MAX / T_OXIDE) * height_noise).astype(np.float32)
    elif perturbation == "alignment_shift":
        max_shift = int(strength)
        if max_shift > 0:
            shifted_layers = []
            for layer in phase_map:
                dy = int(rng.integers(-max_shift, max_shift + 1))
                dx = int(rng.integers(-max_shift, max_shift + 1))
                shifted_layers.append(np.roll(layer, shift=(dy, dx), axis=(0, 1)))
            phase_map = np.stack(shifted_layers, axis=0).astype(np.float32)
    else:
        raise ValueError(f"Unknown perturbation: {perturbation}")
    return np.clip(phase_map, 0.0, PHI_MAX).astype(np.float32)


def evaluate_model(model, val_loader, criterion, device, num_classes):
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    pred_hist = torch.zeros(num_classes, dtype=torch.int64)

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Eval", leave=False):
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
            for c in range(num_classes):
                pred_hist[c] += (pred_cpu == c).sum()

    return {
        "val_loss": val_loss / max(1, len(val_loader)),
        "val_acc": val_correct / max(1, val_total),
        "val_pred_hist": pred_hist.tolist(),
    }


def aggregate_trial_metrics(trials):
    acc = np.array([row["val_acc"] for row in trials], dtype=np.float64)
    loss = np.array([row["val_loss"] for row in trials], dtype=np.float64)
    hist = np.array([row["val_pred_hist"] for row in trials], dtype=np.float64)
    return {
        "mean_acc": float(acc.mean()),
        "std_acc": float(acc.std(ddof=0)),
        "mean_loss": float(loss.mean()),
        "std_loss": float(loss.std(ddof=0)),
        "mean_pred_hist": hist.mean(axis=0).tolist(),
        "trial_pred_hist": [row["val_pred_hist"] for row in trials],
    }


def stable_seed(seed, model_name, perturbation, strength, trial):
    text = f"{model_name}|{perturbation}|{strength}|{trial}"
    value = 0
    for ch in text:
        value = (value * 131 + ord(ch)) % 1_000_000_007
    return seed + value


def run_robustness(args, specs, val_loader, num_classes, device):
    criterion = torch.nn.CrossEntropyLoss().to(device)
    perturbation_grid = {
        "phase_noise": args.phase_noise_std,
        "height_noise": args.height_noise_std_nm,
        "alignment_shift": args.alignment_shift_px,
    }
    rows = []
    for spec in specs:
        print(f"\n========== {spec['model_name']} ==========")
        for perturbation, strengths in perturbation_grid.items():
            for strength in strengths:
                trials = []
                for trial in range(args.mc_runs):
                    rng = np.random.default_rng(
                        stable_seed(args.seed, spec["model_name"], perturbation, float(strength), trial)
                    )
                    phase_map = perturb_phase_map(spec["phase_map"], perturbation, strength, rng)
                    model = make_model_from_phase_map(phase_map, spec["state_dict"], device)
                    metrics = evaluate_model(model, val_loader, criterion, device, num_classes)
                    trials.append(metrics)
                    print(
                        f"{spec['model_name']} {perturbation}={strength} "
                        f"trial={trial + 1}/{args.mc_runs} acc={metrics['val_acc']:.6f}"
                    )

                agg = aggregate_trial_metrics(trials)
                rows.append(
                    {
                        "model_name": spec["model_name"],
                        "family": spec["family"],
                        "levels": spec["levels"],
                        "checkpoint": spec["checkpoint"],
                        "perturbation": perturbation,
                        "strength": strength,
                        "strength_unit": strength_unit(perturbation),
                        "mc_runs": args.mc_runs,
                        "mean_acc": agg["mean_acc"],
                        "std_acc": agg["std_acc"],
                        "mean_loss": agg["mean_loss"],
                        "std_loss": agg["std_loss"],
                        "mean_pred_hist": agg["mean_pred_hist"],
                        "trial_pred_hist": agg["trial_pred_hist"],
                    }
                )
    return rows


def strength_unit(perturbation):
    if perturbation == "phase_noise":
        return "rad_std"
    if perturbation == "height_noise":
        return "nm_std"
    if perturbation == "alignment_shift":
        return "max_pixel_shift"
    return ""


def save_results_csv(rows, path):
    fieldnames = [
        "model_name",
        "family",
        "levels",
        "checkpoint",
        "perturbation",
        "strength",
        "strength_unit",
        "mc_runs",
        "mean_acc",
        "std_acc",
        "mean_loss",
        "std_loss",
        "mean_pred_hist",
        "trial_pred_hist",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["mean_pred_hist"] = json.dumps(row["mean_pred_hist"], ensure_ascii=False)
            out["trial_pred_hist"] = json.dumps(row["trial_pred_hist"], ensure_ascii=False)
            writer.writerow(out)


def save_accuracy_curves(rows, output_dir):
    perturbations = sorted({row["perturbation"] for row in rows})
    model_names = sorted({row["model_name"] for row in rows})
    for perturbation in perturbations:
        fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
        for model_name in model_names:
            subset = [
                row
                for row in rows
                if row["perturbation"] == perturbation and row["model_name"] == model_name
            ]
            if not subset:
                continue
            subset = sorted(subset, key=lambda x: float(x["strength"]))
            x = [float(row["strength"]) for row in subset]
            y = [float(row["mean_acc"]) for row in subset]
            yerr = [float(row["std_acc"]) for row in subset]
            ax.errorbar(x, y, yerr=yerr, marker="o", capsize=3, label=model_name)
        ax.set_xlabel(f"{perturbation} ({strength_unit(perturbation)})")
        ax.set_ylabel("mean val accuracy")
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
        fig.savefig(output_dir / "curves" / f"accuracy_vs_{perturbation}.png", dpi=150)
        plt.close(fig)


def save_report(rows, specs, args, val_dataset, report_path):
    report_path = Path(report_path)
    lines = []
    lines.append("# Micro Device D2NN 误差扰动鲁棒性评估结果\n\n")
    lines.append(f"结果来源：`{Path(args.output_dir) / 'robustness_results.csv'}`  \n")
    lines.append(f"验证集：`{args.val_dir}`，样本数 `{len(val_dataset)}`\n\n")
    lines.append("## 1. 评估目标\n\n")
    lines.append("本评估不重新训练模型，只加载 micro device continuous baseline、PTQ 2/4/8/16 和 QAT 2/4/8/16，在验证集上评估 phase noise、height noise 和 alignment shift 下的鲁棒性。\n\n")
    lines.append("需要说明：这些结果是基于规则生成的 micro device 数据集和 PyTorch D2NN 光学传播模型的算法鲁棒性评估，不是真实硬件实验结果。\n\n")
    lines.append("## 2. 模型列表\n\n")
    for spec in specs:
        lines.append(f"- `{spec['model_name']}`：`{spec['checkpoint']}`\n")
    lines.append("\n## 3. 保留的核心物理逻辑\n\n")
    lines.append("- 输入图像通过 `sqrt(I)` 转为电场振幅。\n")
    lines.append("- 相位调制使用 `exp(1j * phase)`。\n")
    lines.append("- 传播使用 FFT / 角谱法。\n")
    lines.append("- 输出强度使用 `abs(E) ** 2`。\n")
    lines.append("- detector 区域能量求和得到分类 logits。\n\n")
    lines.append("## 4. 扰动设置\n\n")
    lines.append(f"- phase noise：`{args.phase_noise_std}` rad std\n")
    lines.append(f"- height noise：`{args.height_noise_std_nm}` nm std\n")
    lines.append(f"- alignment shift：`{args.alignment_shift_px}` px\n")
    lines.append(f"- Monte Carlo 次数：`{args.mc_runs}`\n\n")

    if rows:
        lines.append("## 5. 运行结果摘要\n\n")
        lines.append("| model | perturbation | strength | mean_acc | std_acc | mean_loss | std_loss | mean_pred_hist |\n")
        lines.append("|---|---|---:|---:|---:|---:|---:|---|\n")
        for row in rows:
            lines.append(
                f"| {row['model_name']} | {row['perturbation']} | {row['strength']} | "
                f"{row['mean_acc']:.6f} | {row['std_acc']:.6f} | "
                f"{row['mean_loss']:.6g} | {row['std_loss']:.6g} | "
                f"`{json.dumps(row['mean_pred_hist'], ensure_ascii=False)}` |\n"
            )
    else:
        lines.append("## 5. 运行结果摘要\n\n")
        lines.append("当前报告尚未包含评估结果。\n")

    lines.append("\n## 6. 输出文件\n\n")
    lines.append(f"- `robustness_results.csv`：`{Path(args.output_dir) / 'robustness_results.csv'}`\n")
    lines.append(f"- phase noise 曲线：`{Path(args.output_dir) / 'curves' / 'accuracy_vs_phase_noise.png'}`\n")
    lines.append(f"- height noise 曲线：`{Path(args.output_dir) / 'curves' / 'accuracy_vs_height_noise.png'}`\n")
    lines.append(f"- alignment shift 曲线：`{Path(args.output_dir) / 'curves' / 'accuracy_vs_alignment_shift.png'}`\n")
    report_path.write_text("".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    report_path = Path(args.report_path)
    ensure_dirs(output_dir, report_path)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using Device:", device)
    print("Output dir:", output_dir)
    print("Monte Carlo runs:", args.mc_runs)

    val_dataset, val_loader = build_val_loader(args.val_dir, args.batch_size, args.num_workers)
    if val_dataset.classes != EXPECTED_CLASSES:
        raise RuntimeError(f"Unexpected val classes: {val_dataset.classes}")
    print("Val classes:", val_dataset.classes)
    print("Val samples:", len(val_dataset))

    specs = build_model_specs(args, device)
    if not specs:
        raise RuntimeError("No model checkpoints found.")
    print("Models:", [spec["model_name"] for spec in specs])

    rows = run_robustness(args, specs, val_loader, len(val_dataset.classes), device)
    backup_dir = output_dir / "backups"
    backup_existing_file(output_dir / "robustness_results.csv", backup_dir)
    backup_existing_file(report_path, backup_dir)
    save_results_csv(rows, output_dir / "robustness_results.csv")
    save_accuracy_curves(rows, output_dir)
    save_report(rows, specs, args, val_dataset, report_path)

    print("Saved:", output_dir / "robustness_results.csv")
    print("Saved curves:", output_dir / "curves")
    print("Saved report:", report_path)
    print("micro device error robustness evaluation finished")


if __name__ == "__main__":
    main()
