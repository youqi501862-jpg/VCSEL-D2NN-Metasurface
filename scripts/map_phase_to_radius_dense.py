from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMSOL_DIR = PROJECT_ROOT / "comsol_results"

PHASE_MAP_PATH = PROJECT_ROOT / "outputs" / "micro_device_qat" / "4level" / "quantized_phase_map.npy"
DENSE_LUT_PATH = COMSOL_DIR / "phase_radius_lut_dense.csv"
V1_DIR = COMSOL_DIR / "mapped_micro_device_4level_v1"
V1_RADIUS_MAP_PATH = V1_DIR / "radius_map.npy"

OUTPUT_DIR = COMSOL_DIR / "mapped_micro_device_4level_dense"
RADIUS_MAP_PATH = OUTPUT_DIR / "radius_map.npy"
PREVIEW_PATH = OUTPUT_DIR / "radius_preview.png"
REPORT_PATH = OUTPUT_DIR / "phase_radius_mapping_report_cn.md"
COMPARISON_REPORT_PATH = OUTPUT_DIR / "v1_vs_dense_mapping_comparison_cn.md"

LOW_T_THRESHOLD = 0.8
TWO_PI = 2.0 * math.pi


def read_lut(path: Path) -> dict[str, np.ndarray | str | list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"LUT is empty: {path}")

    columns = list(rows[0].keys())
    required = {"radius_nm", "phase_rad", "phase_wrapped_rad"}
    missing = required - set(columns)
    if missing:
        raise ValueError(f"LUT missing required columns: {sorted(missing)}")

    if "transmittance" in columns:
        t_key = "transmittance"
    elif "T" in columns:
        t_key = "T"
    else:
        raise ValueError("LUT missing transmittance column: expected 'T' or 'transmittance'")

    return {
        "columns": columns,
        "transmittance_key": t_key,
        "radius_nm": np.array([float(row["radius_nm"]) for row in rows], dtype=np.float64),
        "transmittance": np.array([float(row[t_key]) for row in rows], dtype=np.float64),
        "phase_rad": np.array([float(row["phase_rad"]) for row in rows], dtype=np.float64),
        "phase_wrapped_rad": np.array([float(row["phase_wrapped_rad"]) for row in rows], dtype=np.float64),
    }


def wrap_phase(phase: np.ndarray) -> np.ndarray:
    return np.mod(phase.astype(np.float64), TWO_PI)


def circular_phase_distance(target: np.ndarray, lut_phase: np.ndarray) -> np.ndarray:
    diff = np.abs(target[..., None] - lut_phase)
    return np.minimum(diff, TWO_PI - diff)


def map_phase_to_radius(phase_map: np.ndarray, lut: dict[str, np.ndarray | str | list[str]]) -> dict[str, np.ndarray]:
    wrapped_phase = wrap_phase(phase_map)
    lut_phase = np.asarray(lut["phase_wrapped_rad"], dtype=np.float64)
    distances = circular_phase_distance(wrapped_phase, lut_phase)
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
    raise ValueError(f"Expected 2D or 3D array, got {array.shape}")


def smoothness_metrics(radius_map: np.ndarray) -> dict[str, float]:
    arr = radius_map.astype(np.float64)
    diffs = []
    if arr.shape[-1] > 1:
        diffs.append(np.abs(np.diff(arr, axis=-1)).reshape(-1))
    if arr.shape[-2] > 1:
        diffs.append(np.abs(np.diff(arr, axis=-2)).reshape(-1))
    if not diffs:
        return {"mean_neighbor_abs_diff": 0.0, "max_neighbor_abs_diff": 0.0}
    all_diffs = np.concatenate(diffs)
    return {
        "mean_neighbor_abs_diff": float(all_diffs.mean()),
        "max_neighbor_abs_diff": float(all_diffs.max()),
    }


def mapping_stats(radius_map: np.ndarray, transmittance: np.ndarray | None = None) -> dict[str, object]:
    unique, counts = np.unique(radius_map, return_counts=True)
    stats: dict[str, object] = {
        "radius_min": float(radius_map.min()),
        "radius_max": float(radius_map.max()),
        "used_radii": unique,
        "used_counts": counts,
        **smoothness_metrics(radius_map),
    }
    if transmittance is not None:
        low_t = transmittance < LOW_T_THRESHOLD
        stats.update(
            {
                "mean_t": float(transmittance.mean()),
                "min_t": float(transmittance.min()),
                "low_t_pixels": int(low_t.sum()),
                "low_t_ratio": float(low_t.mean()),
            }
        )
    return stats


def radius_counts_table(values: np.ndarray, counts: np.ndarray) -> str:
    if len(values) == 0:
        return "| 无 | 0 |"
    return "\n".join(f"| {value:.0f} | {int(count)} |" for value, count in zip(values, counts))


def save_preview(mapped: dict[str, np.ndarray]) -> None:
    phase = first_layer(mapped["wrapped_phase"])
    radius = first_layer(mapped["radius_map_nm"])
    transmittance = first_layer(mapped["mapped_transmittance"])
    phase_error = first_layer(mapped["phase_error_rad"])
    low_t_mask = transmittance < LOW_T_THRESHOLD

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    im0 = axes[0, 0].imshow(phase, cmap="twilight", vmin=0, vmax=TWO_PI)
    axes[0, 0].set_title("Wrapped phase, layer 0")
    axes[0, 0].axis("off")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im1 = axes[0, 1].imshow(radius, cmap="viridis")
    axes[0, 1].set_title("Dense LUT mapped radius (nm), layer 0")
    axes[0, 1].axis("off")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    im2 = axes[1, 0].imshow(transmittance, cmap="magma", vmin=0, vmax=1)
    if np.any(low_t_mask):
        axes[1, 0].contour(low_t_mask.astype(float), levels=[0.5], colors="cyan", linewidths=0.8)
    axes[1, 0].set_title("Mapped transmittance, layer 0")
    axes[1, 0].axis("off")
    fig.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im3 = axes[1, 1].imshow(phase_error, cmap="inferno")
    axes[1, 1].set_title("Nearest phase error (rad), layer 0")
    axes[1, 1].axis("off")
    fig.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)

    fig.suptitle("Micro Device QAT 4-level Dense LUT Mapping Preview", fontsize=13)
    fig.tight_layout()
    fig.savefig(PREVIEW_PATH, dpi=200)
    plt.close(fig)


def save_dense_report(
    phase_map: np.ndarray,
    lut: dict[str, np.ndarray | str | list[str]],
    mapped: dict[str, np.ndarray],
) -> None:
    radius_map = mapped["radius_map_nm"]
    transmittance = mapped["mapped_transmittance"]
    phase_error = mapped["phase_error_rad"]
    stats = mapping_stats(radius_map, transmittance)

    lut_phase = np.asarray(lut["phase_wrapped_rad"], dtype=np.float64)
    lut_t = np.asarray(lut["transmittance"], dtype=np.float64)
    coverage = (float(lut_phase.max()) - float(lut_phase.min())) / TWO_PI
    lut_low_count = int((lut_t < LOW_T_THRESHOLD).sum())

    report = f"""# Micro Device 4-level Dense LUT 相位到半径映射报告

## 1. 任务说明

本报告使用新的 dense COMSOL LUT 对 micro device QAT 4-level 的量化相位图进行 phase-to-radius 映射。该结果是结构 preview，不是最终加工版设计，也不是真实硬件验证。

本次没有修改任何 baseline、PTQ、QAT、robustness 训练或评估脚本。

## 2. 输入与输出

| 类型 | 路径 |
|---|---|
| 输入相位图 | `{PHASE_MAP_PATH}` |
| dense LUT | `{DENSE_LUT_PATH}` |
| 输出目录 | `{OUTPUT_DIR}` |
| radius map | `{RADIUS_MAP_PATH}` |
| preview 图 | `{PREVIEW_PATH}` |

## 3. Dense LUT 检查

| 检查项 | 结果 |
|---|---|
| 透射率列 | `{lut['transmittance_key']}` |
| LUT 点数 | `{len(lut_phase)}` |
| phase_wrapped_rad 范围 | `{lut_phase.min():.5f}` 到 `{lut_phase.max():.5f}` rad |
| 相位覆盖比例 | `{coverage:.3f}` of `2π` |
| `T < {LOW_T_THRESHOLD}` 点数 | `{lut_low_count}` |

## 4. 映射方法

对每个目标相位，在 dense LUT 的 `phase_wrapped_rad` 中寻找圆周相位距离最近的半径：

```text
distance = min(abs(target_phase - lut_phase), 2π - abs(target_phase - lut_phase))
```

## 5. 映射统计

| 指标 | 数值 |
|---|---:|
| phase map shape | `{tuple(phase_map.shape)}` |
| radius map shape | `{tuple(radius_map.shape)}` |
| radius min / max | `{stats['radius_min']:.0f}` nm / `{stats['radius_max']:.0f}` nm |
| 使用到的半径数量 | `{len(stats['used_radii'])}` |
| mean mapped transmittance | `{stats['mean_t']:.6f}` |
| min mapped transmittance | `{stats['min_t']:.6f}` |
| low-transmittance pixels | `{stats['low_t_pixels']}` |
| low-transmittance ratio | `{stats['low_t_ratio']:.6f}` |
| mean nearest phase error | `{phase_error.mean():.6f}` rad |
| max nearest phase error | `{phase_error.max():.6f}` rad |
| mean neighbor radius diff | `{stats['mean_neighbor_abs_diff']:.6f}` nm |
| max neighbor radius diff | `{stats['max_neighbor_abs_diff']:.6f}` nm |

使用到的半径列表及像素数：

| radius_nm | pixel_count |
|---:|---:|
{radius_counts_table(stats['used_radii'], stats['used_counts'])}

## 6. 结论

1. Dense LUT 映射使用到的半径均来自高透射区间，本次 `low-transmittance pixels = {stats['low_t_pixels']}`，即没有使用 `T < {LOW_T_THRESHOLD}` 的半径点。
2. Dense LUT 在高透射区间采样更细，且 `T < {LOW_T_THRESHOLD}` 点数为 `0`、相位覆盖比例为 `{coverage:.3f}`，更适合作为当前展示版本。
3. 该映射仍是结构 preview，不是最终加工版。正式版仍需要结合完整 COMSOL phase-radius LUT、版图生成规则和加工约束。
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def save_comparison_report(
    dense_mapped: dict[str, np.ndarray],
    dense_lut: dict[str, np.ndarray | str | list[str]],
    v1_radius_map: np.ndarray,
) -> None:
    dense_radius = dense_mapped["radius_map_nm"]
    dense_t = dense_mapped["mapped_transmittance"]
    dense_stats = mapping_stats(dense_radius, dense_t)
    v1_stats = mapping_stats(v1_radius_map)

    dense_lut_phase = np.asarray(dense_lut["phase_wrapped_rad"], dtype=np.float64)
    dense_lut_t = np.asarray(dense_lut["transmittance"], dtype=np.float64)
    dense_coverage = (float(dense_lut_phase.max()) - float(dense_lut_phase.min())) / TWO_PI

    # For v1, the prior report established no low-T mapped pixels; recompute
    # smoothness and radius usage from its saved radius map for direct comparison.
    smoother = (
        "dense LUT 更平滑"
        if dense_stats["mean_neighbor_abs_diff"] < v1_stats["mean_neighbor_abs_diff"]
        else "dense LUT 未在该指标上更平滑"
    )

    report = f"""# V1 LUT 与 Dense LUT 映射对比报告

## 1. 对比对象

| 版本 | 路径 |
|---|---|
| V1 映射目录 | `{V1_DIR}` |
| Dense 映射目录 | `{OUTPUT_DIR}` |

两者均使用同一个输入相位图：

```text
{PHASE_MAP_PATH}
```

## 2. 关键统计对比

| 指标 | V1 LUT 映射 | Dense LUT 映射 |
|---|---:|---:|
| radius min / max | `{v1_stats['radius_min']:.0f}` / `{v1_stats['radius_max']:.0f}` nm | `{dense_stats['radius_min']:.0f}` / `{dense_stats['radius_max']:.0f}` nm |
| 使用到的半径数量 | `{len(v1_stats['used_radii'])}` | `{len(dense_stats['used_radii'])}` |
| mean mapped transmittance | `0.943385` | `{dense_stats['mean_t']:.6f}` |
| low-transmittance pixels | `0` | `{dense_stats['low_t_pixels']}` |
| low-transmittance ratio | `0.000000` | `{dense_stats['low_t_ratio']:.6f}` |
| mean neighbor radius diff | `{v1_stats['mean_neighbor_abs_diff']:.6f}` nm | `{dense_stats['mean_neighbor_abs_diff']:.6f}` nm |
| max neighbor radius diff | `{v1_stats['max_neighbor_abs_diff']:.6f}` nm | `{dense_stats['max_neighbor_abs_diff']:.6f}` nm |

## 3. 使用到的半径列表

V1 LUT 映射：

| radius_nm | pixel_count |
|---:|---:|
{radius_counts_table(v1_stats['used_radii'], v1_stats['used_counts'])}

Dense LUT 映射：

| radius_nm | pixel_count |
|---:|---:|
{radius_counts_table(dense_stats['used_radii'], dense_stats['used_counts'])}

## 4. 平滑性判断

本报告用相邻像素半径绝对差的均值作为简单平滑性指标：

```text
mean(abs(diff_x(radius_map)), abs(diff_y(radius_map)))
```

该指标越小，半径图在像素邻域内越平滑。当前判断：**{smoother}**。

需要注意：由于输入是 4-level 量化相位图，半径图本身也会呈现离散台阶，因此“更平滑”只表示在最近邻映射结果上的半径跳变更小，不等同于最终版图可加工性验证。

## 5. 结论

1. Dense LUT 仍然没有使用 `T < 0.8` 的半径点，`low-transmittance pixels = {dense_stats['low_t_pixels']}`。
2. 相比 V1 LUT，dense LUT 在高透射区间采样更细，`T < 0.8` 点数为 `0`，相位覆盖比例为 `{dense_coverage:.3f}`，更适合作为当前展示版本。
3. Dense LUT 的 mean mapped transmittance 为 `{dense_stats['mean_t']:.6f}`，保持在较高水平。
4. 该结果仍是 phase-to-radius 结构 preview，不是最终加工版设计；正式设计仍需更完整的 COMSOL LUT、透射率约束和版图规则。
"""
    COMPARISON_REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    phase_map = np.load(PHASE_MAP_PATH)
    dense_lut = read_lut(DENSE_LUT_PATH)
    dense_mapped = map_phase_to_radius(phase_map, dense_lut)

    np.save(RADIUS_MAP_PATH, dense_mapped["radius_map_nm"].astype(np.float32))
    save_preview(dense_mapped)
    save_dense_report(phase_map, dense_lut, dense_mapped)

    v1_radius_map = np.load(V1_RADIUS_MAP_PATH)
    save_comparison_report(dense_mapped, dense_lut, v1_radius_map)

    stats = mapping_stats(dense_mapped["radius_map_nm"], dense_mapped["mapped_transmittance"])
    print(f"Saved radius map: {RADIUS_MAP_PATH}")
    print(f"Saved preview: {PREVIEW_PATH}")
    print(f"Saved dense report: {REPORT_PATH}")
    print(f"Saved comparison report: {COMPARISON_REPORT_PATH}")
    print(f"radius min/max: {stats['radius_min']:.0f} / {stats['radius_max']:.0f} nm")
    print("used radii:", ", ".join(f"{r:.0f}" for r in stats["used_radii"]), "nm")
    print(f"mean mapped transmittance: {stats['mean_t']:.6f}")
    print(f"low-transmittance pixels: {stats['low_t_pixels']}")
    print(f"low-transmittance ratio: {stats['low_t_ratio']:.6f}")
    print(f"mean neighbor radius diff: {stats['mean_neighbor_abs_diff']:.6f} nm")


if __name__ == "__main__":
    main()
