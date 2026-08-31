"""Reproduce a summary from existing D2NN artifacts without model training."""

from __future__ import annotations

import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from project_paths import REPORTS_DIR, ROOT, ensure_dirs


LEVELS = (2, 4, 8, 16)


@dataclass(frozen=True)
class ReproduceSummary:
    micro_baseline_val_acc: float
    micro_ptq: dict[int, float]
    micro_qat: dict[int, float]
    alignment_shift: dict[float, float]
    dense_lut_radius_min: float
    dense_lut_radius_max: float
    dense_lut_points: int
    dense_lut_phase_min: float
    dense_lut_phase_max: float
    dense_lut_coverage: float
    dense_lut_low_t_points: int
    dense_mapping_shape: tuple[int, ...]
    dense_mapping_radii: np.ndarray
    mean_mapped_transmittance: float
    low_transmittance_ratio: float


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"结果 CSV 不存在：{path}")
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except Exception as exc:
        raise ValueError(f"结果 CSV 无法读取：{path}；{exc}") from exc
    if not rows:
        raise ValueError(f"结果 CSV 没有数据行：{path}")
    return rows


def _required_float(row: dict[str, str], key: str, path: Path) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"CSV {path} 的 {key!r} 列含无效数值") from exc


def _level_results(rows: list[dict[str, str]], level_key: str, acc_key: str, path: Path) -> dict[int, float]:
    results: dict[int, float] = {}
    for row in rows:
        try:
            level = int(float(row[level_key]))
        except (KeyError, TypeError, ValueError):
            continue
        if level in LEVELS:
            results[level] = _required_float(row, acc_key, path)
    missing = set(LEVELS) - set(results)
    if missing:
        raise ValueError(f"CSV {path} 缺少量化级别：{sorted(missing)}")
    return results


def collect_summary(root: Path) -> ReproduceSummary:
    root = root.resolve()
    outputs = root / "outputs"
    comsol = root / "comsol_results"

    baseline_path = outputs / "micro_device_d2nn_train" / "training_history.csv"
    baseline_rows = _read_csv(baseline_path)
    baseline_acc = max(_required_float(row, "val_acc", baseline_path) for row in baseline_rows)

    ptq_path = outputs / "micro_device_phase_quantization" / "quantization_results.csv"
    ptq = _level_results(_read_csv(ptq_path), "levels", "val_acc", ptq_path)
    qat_path = outputs / "micro_device_qat" / "qat_summary.csv"
    qat = _level_results(_read_csv(qat_path), "quant_levels", "final_val_acc", qat_path)

    robustness_path = outputs / "micro_device_error_robustness" / "robustness_results.csv"
    alignment: dict[float, float] = {}
    for row in _read_csv(robustness_path):
        if row.get("model_name") == "continuous_baseline" and row.get("perturbation") == "alignment_shift":
            alignment[_required_float(row, "strength", robustness_path)] = _required_float(
                row, "mean_acc", robustness_path
            )
    if not alignment:
        raise ValueError(f"未在 {robustness_path} 中找到 continuous_baseline alignment_shift 数据")

    lut_path = comsol / "phase_radius_lut_dense.csv"
    lut_rows = _read_csv(lut_path)
    radius = np.asarray([_required_float(row, "radius_nm", lut_path) for row in lut_rows], dtype=np.float64)
    phase = np.asarray([_required_float(row, "phase_wrapped_rad", lut_path) for row in lut_rows], dtype=np.float64)
    t_key = "T" if "T" in lut_rows[0] else "transmittance" if "transmittance" in lut_rows[0] else None
    if t_key is None:
        raise ValueError(f"dense LUT 缺少 T/transmittance 列：{lut_path}")
    transmittance = np.asarray([_required_float(row, t_key, lut_path) for row in lut_rows], dtype=np.float64)

    radius_map_path = comsol / "mapped_micro_device_4level_dense" / "radius_map.npy"
    try:
        radius_map = np.load(radius_map_path, allow_pickle=False)
    except Exception as exc:
        raise ValueError(f"dense radius map 无法读取：{radius_map_path}；{exc}") from exc
    used_radii = np.unique(radius_map.astype(np.float64))
    t_by_radius = dict(zip(radius, transmittance))
    missing_radii = [value for value in used_radii if value not in t_by_radius]
    if missing_radii:
        raise ValueError(f"radius map 中的半径不在 dense LUT：{missing_radii}")
    mapped_t = np.vectorize(t_by_radius.__getitem__, otypes=[np.float64])(radius_map.astype(np.float64))

    return ReproduceSummary(
        micro_baseline_val_acc=baseline_acc,
        micro_ptq=ptq,
        micro_qat=qat,
        alignment_shift=dict(sorted(alignment.items())),
        dense_lut_radius_min=float(radius.min()),
        dense_lut_radius_max=float(radius.max()),
        dense_lut_points=len(radius),
        dense_lut_phase_min=float(phase.min()),
        dense_lut_phase_max=float(phase.max()),
        dense_lut_coverage=float((phase.max() - phase.min()) / (2.0 * math.pi)),
        dense_lut_low_t_points=int((transmittance < 0.8).sum()),
        dense_mapping_shape=tuple(radius_map.shape),
        dense_mapping_radii=used_radii,
        mean_mapped_transmittance=float(mapped_t.mean()),
        low_transmittance_ratio=float((mapped_t < 0.8).mean()),
    )


def render_markdown(summary: ReproduceSummary) -> str:
    ptq_rows = "\n".join(
        f"| {level} | {summary.micro_ptq[level]:.6f} | {summary.micro_qat[level]:.6f} |"
        for level in LEVELS
    )
    alignment_rows = "\n".join(
        f"| {strength:g} pixel | {accuracy:.6f} |" for strength, accuracy in summary.alignment_shift.items()
    )
    nonzero_alignment = [(strength, acc) for strength, acc in summary.alignment_shift.items() if strength > 0]
    strongest = min(nonzero_alignment, key=lambda item: item[1]) if nonzero_alignment else min(summary.alignment_shift.items())
    radii = ", ".join(f"{value:.0f}" for value in summary.dense_mapping_radii)
    return f"""# D2NN 已有结果统一复现摘要

> 本脚本只读取已有 CSV 和 NPY，没有重新训练模型。当前成果属于算法仿真 + COMSOL 单元仿真 + phase-to-radius 结构映射 preview，不是真实硬件实验，也不是最终加工版设计。

## 1. Micro Device Baseline 与量化

- baseline val_acc：`{summary.micro_baseline_val_acc:.6f}`。

| phase levels | PTQ val_acc | QAT val_acc |
|---:|---:|---:|
{ptq_rows}

已有结果表明，QAT 对低位相位量化尤其重要；2-level 的提升应结合表中已有数值表述，不外推到真实硬件。

## 2. Alignment Shift 鲁棒性

| 对准偏移 | continuous baseline mean_acc |
|---:|---:|
{alignment_rows}

随着偏移增大，准确率明显下降；在已有采样中，`{strongest[0]:g}` pixel 对应 `{strongest[1]:.6f}`。因此 alignment shift 是当前最敏感误差，应优先考虑 alignment-aware training 或装调容差设计。

## 3. Dense COMSOL LUT

| 指标 | 数值 |
|---|---:|
| 半径范围 | `{summary.dense_lut_radius_min:.0f}-{summary.dense_lut_radius_max:.0f} nm` |
| LUT 点数 | `{summary.dense_lut_points}` |
| phase_wrapped_rad 范围 | `{summary.dense_lut_phase_min:.5f}-{summary.dense_lut_phase_max:.5f} rad` |
| 相位覆盖比例 | `{summary.dense_lut_coverage:.3f} of 2π` |
| `T < 0.8` 点数 | `{summary.dense_lut_low_t_points}` |

## 4. Dense Phase-to-Radius Mapping

| 指标 | 数值 |
|---|---:|
| radius map shape | `{summary.dense_mapping_shape}` |
| 使用半径 | `{radii} nm` |
| mean mapped transmittance | `{summary.mean_mapped_transmittance:.6f}` |
| low-transmittance ratio | `{summary.low_transmittance_ratio:.6f}` |

## 5. 当前局限

1. 分类结果来自合成数据和 PyTorch 算法仿真，不等同于真实器件测试。
2. COMSOL 结果为 meta-atom 单元扫描 LUT，尚不能代表完整 128×128 阵列全波响应。
3. radius map 是最近邻结构映射 preview，没有纳入邻近单元耦合、版图规则和加工误差，不能直接作为最终加工文件。
"""


def terminal_summary(summary: ReproduceSummary) -> str:
    lines = [
        "D2NN 已有结果摘要（未重新训练）",
        f"micro baseline val_acc: {summary.micro_baseline_val_acc:.6f}",
    ]
    lines.extend(
        f"micro {level}-level: PTQ={summary.micro_ptq[level]:.6f}, QAT={summary.micro_qat[level]:.6f}"
        for level in LEVELS
    )
    lines.append(
        "alignment shift: "
        + ", ".join(f"{strength:g}px={acc:.6f}" for strength, acc in summary.alignment_shift.items())
    )
    lines.append(
        f"dense LUT: {summary.dense_lut_radius_min:.0f}-{summary.dense_lut_radius_max:.0f} nm, "
        f"{summary.dense_lut_points} points, coverage={summary.dense_lut_coverage:.3f}, "
        f"T<0.8 points={summary.dense_lut_low_t_points}"
    )
    lines.append(
        f"dense mapping: shape={summary.dense_mapping_shape}, radii="
        + ",".join(f"{value:.0f}" for value in summary.dense_mapping_radii)
        + f" nm, mean T={summary.mean_mapped_transmittance:.6f}, low-T ratio={summary.low_transmittance_ratio:.6f}"
    )
    lines.append("定位：算法仿真 + COMSOL 单元仿真 + 结构映射 preview，非真实硬件实验。")
    return "\n".join(lines)


def main() -> int:
    try:
        summary = collect_summary(ROOT)
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"结果汇总失败：{exc}", file=sys.stderr)
        return 1
    ensure_dirs()
    report_path = REPORTS_DIR / "reproduce_summary_cn.md"
    report_path.write_text(render_markdown(summary), encoding="utf-8")
    print(terminal_summary(summary))
    print(f"报告已生成：{report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
