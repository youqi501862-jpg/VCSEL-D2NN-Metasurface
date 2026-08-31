"""Map a quantized phase map to meta-atom radii using a COMSOL LUT."""

from __future__ import annotations

import argparse
import csv
import math
import sys
import colorsys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


TWO_PI = 2.0 * math.pi


@dataclass(frozen=True)
class LutData:
    columns: tuple[str, ...]
    transmittance_column: str
    radius_nm: np.ndarray
    transmittance: np.ndarray
    phase_rad: np.ndarray
    phase_wrapped_rad: np.ndarray


@dataclass(frozen=True)
class MappingResult:
    wrapped_phase: np.ndarray
    radius_map_nm: np.ndarray
    mapped_transmittance: np.ndarray
    phase_error_rad: np.ndarray


def read_lut(path: Path) -> LutData:
    if not path.is_file():
        raise FileNotFoundError(f"LUT 文件不存在：{path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        columns = tuple(reader.fieldnames or ())
    if not rows:
        raise ValueError(f"LUT 为空：{path}")
    required = {"radius_nm", "phase_wrapped_rad"}
    missing = required - set(columns)
    if missing:
        raise ValueError(f"LUT 缺少必要列：{', '.join(sorted(missing))}")
    if "T" in columns:
        transmittance_column = "T"
    elif "transmittance" in columns:
        transmittance_column = "transmittance"
    else:
        raise ValueError("LUT 缺少透射率列，需要 'T' 或 'transmittance'")

    def values(column: str) -> np.ndarray:
        try:
            return np.asarray([float(row[column]) for row in rows], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"LUT 列 {column!r} 含不可解析数值：{path}") from exc

    wrapped = values("phase_wrapped_rad")
    phase_rad = values("phase_rad") if "phase_rad" in columns else wrapped.copy()
    lut = LutData(
        columns=columns,
        transmittance_column=transmittance_column,
        radius_nm=values("radius_nm"),
        transmittance=values(transmittance_column),
        phase_rad=phase_rad,
        phase_wrapped_rad=np.mod(wrapped, TWO_PI),
    )
    lengths = {len(lut.radius_nm), len(lut.transmittance), len(lut.phase_wrapped_rad)}
    if lengths != {len(rows)}:
        raise ValueError(f"LUT 列长度不一致：{path}")
    if not all(np.all(np.isfinite(array)) for array in (lut.radius_nm, lut.transmittance, lut.phase_wrapped_rad)):
        raise ValueError(f"LUT 包含 NaN 或无穷值：{path}")
    return lut


def circular_phase_distance(target: np.ndarray, reference: np.ndarray) -> np.ndarray:
    difference = np.abs(target[..., None] - reference)
    return np.minimum(difference, TWO_PI - difference)


def map_phase_array(phase_map: np.ndarray, lut: LutData) -> MappingResult:
    if phase_map.ndim not in (2, 3):
        raise ValueError(f"相位图必须是二维或三维，实际 shape={phase_map.shape}")
    if phase_map.size == 0 or not np.all(np.isfinite(phase_map)):
        raise ValueError("相位图为空或包含 NaN/无穷值")
    wrapped = np.mod(phase_map.astype(np.float64), TWO_PI)
    distances = circular_phase_distance(wrapped, lut.phase_wrapped_rad)
    nearest = np.argmin(distances, axis=-1)
    errors = np.take_along_axis(distances, nearest[..., None], axis=-1)[..., 0]
    return MappingResult(
        wrapped_phase=wrapped,
        radius_map_nm=lut.radius_nm[nearest],
        mapped_transmittance=lut.transmittance[nearest],
        phase_error_rad=errors,
    )


def coverage_ratio(lut: LutData) -> float:
    return float((lut.phase_wrapped_rad.max() - lut.phase_wrapped_rad.min()) / TWO_PI)


def coverage_judgement(lut: LutData) -> str:
    ratio = coverage_ratio(lut)
    if ratio >= 0.95:
        return "相位范围覆盖较完整，但仍需结合目标相位和加工约束验证"
    return "存在覆盖风险：当前 LUT 的相位范围未达到 0.95 × 2π"


def _first_layer(array: np.ndarray) -> np.ndarray:
    return array if array.ndim == 2 else array[0]


def _normalize(array: np.ndarray, vmin: float | None = None, vmax: float | None = None) -> np.ndarray:
    lower = float(np.min(array)) if vmin is None else vmin
    upper = float(np.max(array)) if vmax is None else vmax
    if upper <= lower:
        return np.zeros(array.shape, dtype=np.float64)
    return np.clip((array.astype(np.float64) - lower) / (upper - lower), 0.0, 1.0)


def _colorize(array: np.ndarray, palette: str, vmin: float | None = None, vmax: float | None = None) -> np.ndarray:
    normalized = _normalize(array, vmin, vmax)
    if palette == "phase":
        flat = normalized.reshape(-1)
        rgb = np.asarray([colorsys.hsv_to_rgb(float(value), 0.85, 0.95) for value in flat]).reshape(
            normalized.shape + (3,)
        )
    elif palette == "radius":
        rgb = np.stack(
            (0.18 + 0.72 * normalized, 0.12 + 0.78 * np.sqrt(normalized), 0.55 - 0.42 * normalized), axis=-1
        )
    elif palette == "transmittance":
        rgb = np.stack((np.sqrt(normalized), normalized**2, 0.28 * (1.0 - normalized)), axis=-1)
    else:
        rgb = np.stack((normalized, 0.22 * normalized, 0.05 + 0.1 * (1.0 - normalized)), axis=-1)
    return np.asarray(np.clip(rgb * 255.0, 0, 255), dtype=np.uint8)


def _panel(data: np.ndarray, title: str, palette: str, vmin: float | None, vmax: float | None) -> Image.Image:
    rgb = _colorize(data, palette, vmin, vmax)
    image = Image.fromarray(rgb, mode="RGB").resize((640, 480), Image.Resampling.NEAREST)
    panel = Image.new("RGB", (680, 540), "white")
    panel.paste(image, (20, 45))
    draw = ImageDraw.Draw(panel)
    draw.text((20, 15), title, fill="black")
    return panel


def save_preview(result: MappingResult, output_path: Path, label: str, low_t_threshold: float) -> None:
    phase = _first_layer(result.wrapped_phase)
    radius = _first_layer(result.radius_map_nm)
    transmittance = _first_layer(result.mapped_transmittance)
    error = _first_layer(result.phase_error_rad)
    low_mask = transmittance < low_t_threshold
    panels = [
        _panel(phase, "Wrapped phase, layer 0", "phase", 0.0, TWO_PI),
        _panel(radius, "Mapped radius (nm), layer 0", "radius", None, None),
        _panel(transmittance, "Mapped transmittance, layer 0", "transmittance", 0.0, 1.0),
        _panel(error, "Nearest phase error (rad), layer 0", "error", 0.0, None),
    ]
    if np.any(low_mask):
        overlay = np.asarray(panels[2].crop((20, 45, 660, 525))).copy()
        scaled_mask = Image.fromarray(low_mask.astype(np.uint8) * 255).resize((640, 480), Image.Resampling.NEAREST)
        mask = np.asarray(scaled_mask) > 0
        overlay[mask] = np.array([0, 255, 255], dtype=np.uint8)
        panels[2].paste(Image.fromarray(overlay, mode="RGB"), (20, 45))
    canvas = Image.new("RGB", (1360, 1140), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 10), f"{label}: phase-to-radius structure preview", fill="black")
    positions = ((0, 55), (680, 55), (0, 595), (680, 595))
    for panel, position in zip(panels, positions):
        canvas.paste(panel, position)
    canvas.save(output_path, format="PNG")


def save_report(
    phase_map_path: Path,
    lut_path: Path,
    output_dir: Path,
    label: str,
    low_t_threshold: float,
    lut: LutData,
    result: MappingResult,
) -> None:
    radii, counts = np.unique(result.radius_map_nm, return_counts=True)
    low_mask = result.mapped_transmittance < low_t_threshold
    radius_rows = "\n".join(f"| {radius:.0f} | {int(count)} |" for radius, count in zip(radii, counts))
    report = f"""# {label} Phase-to-Radius 结构映射报告

> 本结果属于算法相位图与 COMSOL 单元 LUT 之间的结构映射 preview，不是真实硬件实验，也不是最终加工版设计。

## 输入与输出

| 项目 | 路径 |
|---|---|
| phase map | `{phase_map_path}` |
| LUT | `{lut_path}` |
| output | `{output_dir}` |

## 映射方法

相位先 wrap 到 `[0, 2π)`，再按圆周相位距离在 `phase_wrapped_rad` 中寻找最近半径。LUT 透射率列使用 `{lut.transmittance_column}`。

## 统计结果

| 指标 | 数值 |
|---|---:|
| phase map shape | `{tuple(result.wrapped_phase.shape)}` |
| radius min / max | `{result.radius_map_nm.min():.0f}` / `{result.radius_map_nm.max():.0f}` nm |
| 使用到的半径 | `{', '.join(f'{value:.0f}' for value in radii)}` nm |
| mean mapped transmittance | `{result.mapped_transmittance.mean():.6f}` |
| low-transmittance threshold | `{low_t_threshold:.3f}` |
| low-transmittance pixels | `{int(low_mask.sum())}` |
| low-transmittance ratio | `{float(low_mask.mean()):.6f}` |
| 最大相位匹配误差 | `{result.phase_error_rad.max():.6f}` rad |
| 平均相位匹配误差 | `{result.phase_error_rad.mean():.6f}` rad |
| LUT 相位覆盖比例 | `{coverage_ratio(lut):.3f} of 2π` |
| LUT 覆盖风险 | {coverage_judgement(lut)} |

## 半径像素统计

| radius_nm | pixel_count |
|---:|---:|
{radius_rows}

## 局限说明

最近邻映射没有模拟相邻 meta-atom 耦合、阵列级全波响应、版图设计规则或制造误差。输出只能用于结构接口验证和展示，不能直接作为加工文件。
"""
    (output_dir / "phase_radius_mapping_report_cn.md").write_text(report, encoding="utf-8")


def run_mapping(
    phase_map_path: Path,
    lut_path: Path,
    output_dir: Path,
    label: str,
    low_t_threshold: float,
) -> MappingResult:
    if not phase_map_path.is_file():
        raise FileNotFoundError(f"相位图文件不存在：{phase_map_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        phase_map = np.load(phase_map_path, allow_pickle=False)
    except Exception as exc:
        raise ValueError(f"相位图无法读取：{phase_map_path}；{exc}") from exc
    lut = read_lut(lut_path)
    result = map_phase_array(phase_map, lut)
    np.save(output_dir / "radius_map.npy", result.radius_map_nm.astype(np.float32))
    save_preview(result, output_dir / "radius_preview.png", label, low_t_threshold)
    save_report(phase_map_path, lut_path, output_dir, label, low_t_threshold, lut, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用 COMSOL LUT 生成 phase-to-radius 结构预览")
    parser.add_argument("--phase-map", required=True, type=Path, help="quantized_phase_map.npy 路径")
    parser.add_argument("--lut", required=True, type=Path, help="phase-radius LUT CSV 路径")
    parser.add_argument("--output-dir", required=True, type=Path, help="独立输出目录")
    parser.add_argument("--label", default="D2NN", help="报告和预览图标签")
    parser.add_argument("--low-t-threshold", type=float, default=0.8, help="低透射率阈值")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.low_t_threshold <= 1.0:
        print("错误：--low-t-threshold 必须位于 [0, 1]。", file=sys.stderr)
        return 2
    try:
        result = run_mapping(args.phase_map, args.lut, args.output_dir, args.label, args.low_t_threshold)
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"映射失败：{exc}", file=sys.stderr)
        return 1
    radii = np.unique(result.radius_map_nm)
    low_mask = result.mapped_transmittance < args.low_t_threshold
    print(f"映射完成：{args.output_dir}")
    print(f"phase map shape: {tuple(result.wrapped_phase.shape)}")
    print("used radii (nm): " + ", ".join(f"{value:.0f}" for value in radii))
    print(f"mean mapped transmittance: {result.mapped_transmittance.mean():.6f}")
    print(f"low-transmittance ratio: {float(low_mask.mean()):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
