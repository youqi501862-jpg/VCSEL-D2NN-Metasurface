from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LUT_PATH = PROJECT_ROOT / "comsol_results" / "phase_radius_lut.csv"
PHASE_MAP_PATH = PROJECT_ROOT / "outputs" / "micro_device_qat" / "4level" / "quantized_phase_map.npy"
OUTPUT_DIR = PROJECT_ROOT / "comsol_results" / "mapped_micro_device_4level_v1"

RADIUS_MAP_PATH = OUTPUT_DIR / "radius_map.npy"
PREVIEW_PATH = OUTPUT_DIR / "radius_preview.png"
REPORT_PATH = OUTPUT_DIR / "phase_radius_mapping_report_cn.md"

LOW_T_THRESHOLD = 0.8
TWO_PI = 2.0 * math.pi


def wrap_phase_0_2pi(phase: np.ndarray) -> np.ndarray:
    return np.mod(phase.astype(np.float64), TWO_PI)


def circular_distance(target: np.ndarray, lut_phase: np.ndarray) -> np.ndarray:
    diff = np.abs(target[..., None] - lut_phase)
    return np.minimum(diff, TWO_PI - diff)


def read_phase_radius_lut(path: Path) -> dict[str, np.ndarray | str | list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"LUT is empty: {path}")

    columns = list(rows[0].keys())
    required_columns = {"radius_nm", "phase_rad", "phase_wrapped_rad"}
    missing = required_columns - set(columns)
    if missing:
        raise ValueError(f"LUT missing required columns: {sorted(missing)}")

    if "transmittance" in columns:
        transmittance_key = "transmittance"
    elif "T" in columns:
        transmittance_key = "T"
    else:
        raise ValueError("LUT missing transmittance column: expected 'T' or 'transmittance'")

    return {
        "columns": columns,
        "transmittance_key": transmittance_key,
        "radius_nm": np.array([float(row["radius_nm"]) for row in rows], dtype=np.float64),
        "transmittance": np.array([float(row[transmittance_key]) for row in rows], dtype=np.float64),
        "phase_rad": np.array([float(row["phase_rad"]) for row in rows], dtype=np.float64),
        "phase_wrapped_rad": np.array([float(row["phase_wrapped_rad"]) for row in rows], dtype=np.float64),
    }


def map_phase_to_radius(phase_map: np.ndarray, lut: dict[str, np.ndarray | str | list[str]]) -> dict[str, np.ndarray]:
    wrapped_phase = wrap_phase_0_2pi(phase_map)
    lut_phase = np.asarray(lut["phase_wrapped_rad"], dtype=np.float64)
    distances = circular_distance(wrapped_phase, lut_phase)
    nearest_idx = np.argmin(distances, axis=-1)

    radius_nm = np.asarray(lut["radius_nm"], dtype=np.float64)
    transmittance = np.asarray(lut["transmittance"], dtype=np.float64)

    return {
        "wrapped_phase": wrapped_phase,
        "nearest_idx": nearest_idx,
        "radius_map_nm": radius_nm[nearest_idx],
        "mapped_transmittance": transmittance[nearest_idx],
        "phase_error_rad": np.take_along_axis(distances, nearest_idx[..., None], axis=-1)[..., 0],
    }


def first_layer(array: np.ndarray) -> np.ndarray:
    if array.ndim == 2:
        return array
    if array.ndim == 3:
        return array[0]
    raise ValueError(f"Expected a 2D or 3D map, got shape {array.shape}")


def save_radius_preview(mapped: dict[str, np.ndarray]) -> None:
    phase = first_layer(mapped["wrapped_phase"])
    radius = first_layer(mapped["radius_map_nm"])
    transmittance = first_layer(mapped["mapped_transmittance"])
    phase_error = first_layer(mapped["phase_error_rad"])
    low_t_mask = transmittance < LOW_T_THRESHOLD

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    im0 = axes[0, 0].imshow(phase, cmap="twilight", vmin=0.0, vmax=TWO_PI)
    axes[0, 0].set_title("Wrapped phase, layer 0")
    axes[0, 0].axis("off")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im1 = axes[0, 1].imshow(radius, cmap="viridis")
    axes[0, 1].set_title("Mapped radius (nm), layer 0")
    axes[0, 1].axis("off")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    im2 = axes[1, 0].imshow(transmittance, cmap="magma", vmin=0.0, vmax=1.0)
    if np.any(low_t_mask):
        axes[1, 0].contour(low_t_mask.astype(float), levels=[0.5], colors="cyan", linewidths=0.8)
    axes[1, 0].set_title("Mapped transmittance, layer 0")
    axes[1, 0].axis("off")
    fig.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im3 = axes[1, 1].imshow(phase_error, cmap="inferno")
    axes[1, 1].set_title("Nearest phase error (rad), layer 0")
    axes[1, 1].axis("off")
    fig.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)

    fig.suptitle("Micro Device QAT 4-level: phase-to-radius v1 preview", fontsize=13)
    fig.tight_layout()
    fig.savefig(PREVIEW_PATH, dpi=200)
    plt.close(fig)


def markdown_count_table(values: np.ndarray, counts: np.ndarray) -> str:
    if len(values) == 0:
        return "| 无 | 0 |"
    return "\n".join(f"| {value:.0f} | {int(count)} |" for value, count in zip(values, counts))


def save_report(
    phase_map: np.ndarray,
    lut: dict[str, np.ndarray | str | list[str]],
    mapped: dict[str, np.ndarray],
) -> None:
    radius_map = mapped["radius_map_nm"]
    mapped_t = mapped["mapped_transmittance"]
    phase_error = mapped["phase_error_rad"]
    low_t_mask = mapped_t < LOW_T_THRESHOLD

    used_radii, used_counts = np.unique(radius_map, return_counts=True)
    low_radii, low_counts = np.unique(radius_map[low_t_mask], return_counts=True)

    lut_phase = np.asarray(lut["phase_wrapped_rad"], dtype=np.float64)
    lut_t = np.asarray(lut["transmittance"], dtype=np.float64)
    lut_radius = np.asarray(lut["radius_nm"], dtype=np.float64)
    lut_low_mask = lut_t < LOW_T_THRESHOLD
    phase_coverage = (float(lut_phase.max()) - float(lut_phase.min())) / TWO_PI
    has_low_t_mapping = bool(np.any(low_t_mask))

    lut_low_rows = "\n".join(
        f"| {radius:.0f} | {t:.5f} | {phase:.5f} |"
        for radius, t, phase in zip(lut_radius[lut_low_mask], lut_t[lut_low_mask], lut_phase[lut_low_mask])
    )
    if not lut_low_rows:
        lut_low_rows = "| 无 | 无 | 无 |"

    if phase_coverage >= 0.95:
        coverage_judgement = "相位覆盖较完整"
    else:
        coverage_judgement = "存在覆盖风险：当前 LUT 没有完整覆盖 0 到 2π"

    report = f"""# Micro Device 4-level 第一版 LUT 结构映射 Preview 报告

## 1. 任务说明

本报告使用第一版 COMSOL phase-radius LUT 对 micro device QAT 4-level 的量化相位图进行 phase-to-radius 映射验证。该结果是结构映射 preview，不是最终加工版设计，也不是真实硬件验证。

本次没有生成 dense LUT，没有修改任何 baseline、PTQ、QAT 或 robustness 训练/评估脚本。

## 2. 输入与输出

| 类型 | 路径 |
|---|---|
| 输入 LUT | `{LUT_PATH}` |
| 输入相位图 | `{PHASE_MAP_PATH}` |
| 输出目录 | `{OUTPUT_DIR}` |
| radius map | `{RADIUS_MAP_PATH}` |
| preview 图 | `{PREVIEW_PATH}` |

## 3. LUT 检查

| 检查项 | 结果 |
|---|---|
| CSV 列名 | `{', '.join(lut['columns'])}` |
| `radius_nm` | 存在 |
| 透射率列 | 使用 `{lut['transmittance_key']}`，兼容 `T` / `transmittance` |
| `phase_rad` | 存在 |
| `phase_wrapped_rad` | 存在 |
| LUT 相位范围 | `{lut_phase.min():.5f}` 到 `{lut_phase.max():.5f}` rad |
| LUT 相位覆盖比例 | `{phase_coverage:.3f}` of `2π` |
| 覆盖风险判断 | {coverage_judgement} |

LUT 中 `T < {LOW_T_THRESHOLD}` 的半径点：

| radius_nm | transmittance | phase_wrapped_rad |
|---:|---:|---:|
{lut_low_rows}

## 4. 映射方法

对相位图中每个目标相位先 wrap 到 `[0, 2π)`，然后在 LUT 的 `phase_wrapped_rad` 中寻找圆周相位距离最近的半径：

```text
distance = min(abs(target_phase - lut_phase), 2π - abs(target_phase - lut_phase))
```

该方法是最近邻映射，优点是简单、可解释；局限是没有插值，也没有把透射率作为优化目标。

## 5. 映射统计

| 指标 | 数值 |
|---|---:|
| phase map shape | `{tuple(phase_map.shape)}` |
| radius map shape | `{tuple(radius_map.shape)}` |
| radius min / max | `{radius_map.min():.0f}` nm / `{radius_map.max():.0f}` nm |
| 使用到的半径数量 | `{len(used_radii)}` |
| mean mapped transmittance | `{mapped_t.mean():.6f}` |
| min mapped transmittance | `{mapped_t.min():.6f}` |
| low-transmittance pixels | `{int(low_t_mask.sum())}` |
| low-transmittance ratio | `{float(low_t_mask.mean()):.6f}` |
| 是否映射到 `T < {LOW_T_THRESHOLD}` 的半径点 | `{'是' if has_low_t_mapping else '否'}` |
| mean nearest phase error | `{phase_error.mean():.6f}` rad |
| max nearest phase error | `{phase_error.max():.6f}` rad |

使用到的半径列表及像素数：

| radius_nm | pixel_count |
|---:|---:|
{markdown_count_table(used_radii, used_counts)}

映射到低透射率半径的像素分布：

| radius_nm | pixel_count |
|---:|---:|
{markdown_count_table(low_radii, low_counts)}

## 6. 结论

1. 当前第一版 LUT 可以用于 phase-to-radius 的初步映射 preview，因为 `phase_wrapped_rad` 已覆盖约 `{phase_coverage:.1%}` 的 `0 到 2π` 范围，并且本次 4-level 相位图只使用了有限的离散相位。
2. 当前 LUT 仍存在相位覆盖风险：它没有完整覆盖 `0 到 2π`，并且 `170` 到 `220` nm 的若干半径点透射率低于 `{LOW_T_THRESHOLD}`。
3. 本次映射是否触及低透射率半径点：`{'是' if has_low_t_mapping else '否'}`。如果后续其他 phase map 映射到低透射率半径，需要在结构设计中考虑能量损失。
4. 当前结果适合作为“第一版 COMSOL LUT 到 D2NN 相位图”的接口验证，不应作为最终加工版设计。正式设计仍需要更完整的 COMSOL phase-radius LUT、透射率约束，以及版图生成规则。
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    lut = read_phase_radius_lut(LUT_PATH)
    phase_map = np.load(PHASE_MAP_PATH)
    mapped = map_phase_to_radius(phase_map, lut)

    np.save(RADIUS_MAP_PATH, mapped["radius_map_nm"].astype(np.float32))
    save_radius_preview(mapped)
    save_report(phase_map, lut, mapped)

    low_t_mask = mapped["mapped_transmittance"] < LOW_T_THRESHOLD
    used_radii = np.unique(mapped["radius_map_nm"])
    print(f"Loaded LUT: {LUT_PATH}")
    print(f"Loaded phase map: {PHASE_MAP_PATH}")
    print(f"Saved radius map: {RADIUS_MAP_PATH}")
    print(f"Saved preview: {PREVIEW_PATH}")
    print(f"Saved report: {REPORT_PATH}")
    print(f"phase map shape: {tuple(phase_map.shape)}")
    print(f"radius min/max: {mapped['radius_map_nm'].min():.0f} / {mapped['radius_map_nm'].max():.0f} nm")
    print("used radii:", ", ".join(f"{r:.0f}" for r in used_radii), "nm")
    print(f"mean mapped transmittance: {mapped['mapped_transmittance'].mean():.6f}")
    print(f"low-transmittance pixels: {int(low_t_mask.sum())}")
    print(f"low-transmittance ratio: {float(low_t_mask.mean()):.6f}")


if __name__ == "__main__":
    main()
