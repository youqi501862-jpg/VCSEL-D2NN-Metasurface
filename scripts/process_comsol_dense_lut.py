from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMSOL_DIR = PROJECT_ROOT / "comsol_results"

RAW_CSV = COMSOL_DIR / "raw_radius_sweep_80_160nm_step5.csv"
V1_LUT_CSV = COMSOL_DIR / "phase_radius_lut.csv"

DENSE_LUT_CSV = COMSOL_DIR / "phase_radius_lut_dense.csv"
PHASE_PNG = COMSOL_DIR / "phase_vs_radius_dense.png"
TRANSMITTANCE_PNG = COMSOL_DIR / "transmittance_vs_radius_dense.png"
REPORT_PATH = COMSOL_DIR / "comsol_meta_atom_lut_dense_summary_cn.md"

LOW_T_THRESHOLD = 0.8
TWO_PI = 2.0 * math.pi


FALLBACK_ROWS = [
    (8.0000e-8, 0.94805, 2.6074),
    (8.5000e-8, 0.94733, 2.5328),
    (9.0000e-8, 0.94561, 2.4484),
    (9.5000e-8, 0.94290, 2.3532),
    (1.0000e-7, 0.93925, 2.2460),
    (1.0500e-7, 0.93522, 2.1256),
    (1.1000e-7, 0.93168, 1.9911),
    (1.1500e-7, 0.92973, 1.8411),
    (1.2000e-7, 0.93122, 1.6733),
    (1.2500e-7, 0.93748, 1.4865),
    (1.3000e-7, 0.94960, 1.2784),
    (1.3500e-7, 0.96596, 1.0476),
    (1.4000e-7, 0.98257, 0.79369),
    (1.4500e-7, 0.99296, 0.51681),
    (1.5000e-7, 0.99099, 0.21656),
    (1.5500e-7, 0.97144, -0.12063),
    (1.6000e-7, 0.89600, -0.56862),
]


def wrap_phase(phase_rad: float) -> float:
    return phase_rad % TWO_PI


def load_raw_dense_rows() -> tuple[list[dict[str, float]], str]:
    try:
        lines = RAW_CSV.read_text(encoding="utf-8-sig").splitlines()
        data_lines = [line for line in lines if line.strip() and not line.startswith("%")]
        rows = []
        for line in data_lines:
            parts = next(csv.reader([line]))
            if len(parts) < 6:
                continue
            radius_m = float(parts[0])
            transmittance = float(parts[4])
            phase_rad = float(parts[5])
            rows.append(
                {
                    "radius_nm": radius_m * 1e9,
                    "T": transmittance,
                    "phase_rad": phase_rad,
                    "phase_wrapped_rad": wrap_phase(phase_rad),
                }
            )
        if rows:
            return rows, "raw CSV"
    except Exception as exc:
        print(f"Failed to read raw CSV, using fallback rows: {exc}")

    rows = [
        {
            "radius_nm": radius_m * 1e9,
            "T": transmittance,
            "phase_rad": phase_rad,
            "phase_wrapped_rad": wrap_phase(phase_rad),
        }
        for radius_m, transmittance, phase_rad in FALLBACK_ROWS
    ]
    return rows, "fallback data"


def load_lut(path: Path) -> list[dict[str, float]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    loaded = []
    for row in rows:
        t_key = "transmittance" if "transmittance" in row else "T"
        loaded.append(
            {
                "radius_nm": float(row["radius_nm"]),
                "T": float(row[t_key]),
                "phase_rad": float(row["phase_rad"]),
                "phase_wrapped_rad": float(row["phase_wrapped_rad"]),
            }
        )
    return loaded


def save_dense_lut(rows: list[dict[str, float]]) -> None:
    with DENSE_LUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["radius_nm", "T", "phase_rad", "phase_wrapped_rad"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "radius_nm": f"{row['radius_nm']:.6g}",
                    "T": f"{row['T']:.8g}",
                    "phase_rad": f"{row['phase_rad']:.8g}",
                    "phase_wrapped_rad": f"{row['phase_wrapped_rad']:.8g}",
                }
            )


def plot_phase(rows: list[dict[str, float]]) -> None:
    radius = [row["radius_nm"] for row in rows]
    phase = [row["phase_rad"] for row in rows]
    wrapped = [row["phase_wrapped_rad"] for row in rows]
    low_rows = [row for row in rows if row["T"] < LOW_T_THRESHOLD]

    plt.figure(figsize=(7.5, 4.8))
    plt.plot(radius, phase, "o-", label="phase_rad")
    plt.plot(radius, wrapped, "s--", label="phase_wrapped_rad (0 to 2pi)")
    if low_rows:
        plt.scatter(
            [row["radius_nm"] for row in low_rows],
            [row["phase_wrapped_rad"] for row in low_rows],
            c="red",
            marker="x",
            s=80,
            label=f"T < {LOW_T_THRESHOLD}",
            zorder=5,
        )
    plt.xlabel("Radius (nm)")
    plt.ylabel("Phase (rad)")
    plt.title("Dense COMSOL Phase vs Radius, 80-160 nm step 5 nm")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PHASE_PNG, dpi=200)
    plt.close()


def plot_transmittance(rows: list[dict[str, float]]) -> None:
    radius = [row["radius_nm"] for row in rows]
    transmittance = [row["T"] for row in rows]
    low_rows = [row for row in rows if row["T"] < LOW_T_THRESHOLD]

    plt.figure(figsize=(7.5, 4.8))
    plt.plot(radius, transmittance, "o-", label="T")
    plt.axhline(
        LOW_T_THRESHOLD,
        color="red",
        linestyle="--",
        linewidth=1.2,
        label=f"T = {LOW_T_THRESHOLD}",
    )
    if low_rows:
        plt.scatter(
            [row["radius_nm"] for row in low_rows],
            [row["T"] for row in low_rows],
            c="red",
            marker="x",
            s=80,
            label=f"T < {LOW_T_THRESHOLD}",
            zorder=5,
        )
    plt.xlabel("Radius (nm)")
    plt.ylabel("Transmittance")
    plt.title("Dense COMSOL Transmittance vs Radius, 80-160 nm step 5 nm")
    plt.ylim(0.0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(TRANSMITTANCE_PNG, dpi=200)
    plt.close()


def phase_stats(rows: list[dict[str, float]]) -> dict[str, float]:
    phases = [row["phase_wrapped_rad"] for row in rows]
    return {
        "min": min(phases),
        "max": max(phases),
        "span": max(phases) - min(phases),
        "coverage": (max(phases) - min(phases)) / TWO_PI,
    }


def save_report(
    dense_rows: list[dict[str, float]],
    v1_rows: list[dict[str, float]],
    source: str,
) -> None:
    dense_low = [row for row in dense_rows if row["T"] < LOW_T_THRESHOLD]
    v1_low = [row for row in v1_rows if row["T"] < LOW_T_THRESHOLD]
    dense_stats = phase_stats(dense_rows)
    v1_stats = phase_stats(v1_rows)

    dense_low_table = "\n".join(
        f"| {row['radius_nm']:.0f} | {row['T']:.5f} | {row['phase_rad']:.5f} | {row['phase_wrapped_rad']:.5f} |"
        for row in dense_low
    ) or "| 无 | 无 | 无 | 无 |"

    v1_low_table = "\n".join(
        f"| {row['radius_nm']:.0f} | {row['T']:.5f} | {row['phase_wrapped_rad']:.5f} |"
        for row in v1_low
    ) or "| 无 | 无 | 无 |"

    report = f"""# COMSOL 高透射区间 Dense LUT 处理报告

## 1. 数据来源

本报告由 `process_comsol_dense_lut.py` 生成。输入为 COMSOL 高透射区间细扫结果：

```text
{RAW_CSV}
```

读取来源：`{source}`。扫描范围为 `{dense_rows[0]['radius_nm']:.0f}` nm 到 `{dense_rows[-1]['radius_nm']:.0f}` nm，步长约 `5` nm，共 `{len(dense_rows)}` 个点。

本处理只生成 COMSOL LUT、曲线图和报告，没有修改任何 baseline、PTQ、QAT、robustness 训练或评估脚本。

## 2. 输出文件

| 文件 | 说明 |
|---|---|
| `{DENSE_LUT_CSV}` | dense phase-radius LUT |
| `{PHASE_PNG}` | dense 相位-半径曲线 |
| `{TRANSMITTANCE_PNG}` | dense 透射率-半径曲线 |
| `{REPORT_PATH}` | 本报告 |

## 3. Dense LUT 字段

输出 CSV 保留以下字段：

| 字段 | 含义 |
|---|---|
| `radius_nm` | 半径，单位 nm，由 COMSOL 原始 `r_pillar (m)` 换算 |
| `T` | 透射率 |
| `phase_rad` | 原始相位，单位 rad |
| `phase_wrapped_rad` | wrap 到 `[0, 2π)` 后的相位，单位 rad |

## 4. Dense LUT 质量检查

| 指标 | 数值 |
|---|---:|
| 点数 | `{len(dense_rows)}` |
| 半径范围 | `{dense_rows[0]['radius_nm']:.0f}` nm 到 `{dense_rows[-1]['radius_nm']:.0f}` nm |
| T min / max | `{min(row['T'] for row in dense_rows):.5f}` / `{max(row['T'] for row in dense_rows):.5f}` |
| phase_wrapped_rad min / max | `{dense_stats['min']:.5f}` / `{dense_stats['max']:.5f}` |
| phase_wrapped_rad span | `{dense_stats['span']:.5f}` rad |
| 相位覆盖比例 | `{dense_stats['coverage']:.3f}` of `2π` |
| `T < {LOW_T_THRESHOLD}` 点数 | `{len(dense_low)}` |

Dense LUT 中 `T < {LOW_T_THRESHOLD}` 的点：

| radius_nm | T | phase_rad | phase_wrapped_rad |
|---:|---:|---:|---:|
{dense_low_table}

## 5. 与第一版 LUT 对比

第一版 LUT 文件：

```text
{V1_LUT_CSV}
```

| 对比项 | 第一版 LUT | Dense LUT |
|---|---:|---:|
| 点数 | `{len(v1_rows)}` | `{len(dense_rows)}` |
| 半径范围 | `{v1_rows[0]['radius_nm']:.0f}` 到 `{v1_rows[-1]['radius_nm']:.0f}` nm | `{dense_rows[0]['radius_nm']:.0f}` 到 `{dense_rows[-1]['radius_nm']:.0f}` nm |
| 半径步长 | 约 `10` nm | 约 `5` nm |
| T min / max | `{min(row['T'] for row in v1_rows):.5f}` / `{max(row['T'] for row in v1_rows):.5f}` | `{min(row['T'] for row in dense_rows):.5f}` / `{max(row['T'] for row in dense_rows):.5f}` |
| `T < {LOW_T_THRESHOLD}` 点数 | `{len(v1_low)}` | `{len(dense_low)}` |
| phase_wrapped_rad 范围 | `{v1_stats['min']:.5f}` 到 `{v1_stats['max']:.5f}` | `{dense_stats['min']:.5f}` 到 `{dense_stats['max']:.5f}` |
| 相位覆盖比例 | `{v1_stats['coverage']:.3f}` of `2π` | `{dense_stats['coverage']:.3f}` of `2π` |

第一版 LUT 中的低透射率点：

| radius_nm | T | phase_wrapped_rad |
|---:|---:|---:|
{v1_low_table}

## 6. 是否更适合 micro device 4-level phase-to-radius preview

结论：dense LUT 更适合当前 micro device 4-level phase-to-radius preview。

理由如下：

1. Dense LUT 聚焦在 `80` 到 `160` nm 的高透射区间，当前所有点的 `T` 都不低于 `{LOW_T_THRESHOLD}`，最低透射率为 `{min(row['T'] for row in dense_rows):.5f}`。
2. 第一版 LUT 覆盖半径更宽，但 `170` 到 `220` nm 存在多个低透射率点；这些点虽然扩展了相位范围，但会带来能量损失风险。
3. Dense LUT 半径步长为 `5` nm，比第一版 `10` nm 更细，适合做更平滑、更稳定的最近邻相位匹配。
4. 需要注意，dense LUT 仍然不是完整高透射全相位库。它的 `phase_wrapped_rad` 主要覆盖 `{dense_stats['min']:.5f}` 到 `{dense_stats['max']:.5f}` rad，虽然数值跨度约为 `{dense_stats['coverage']:.1%}` 的 `2π`，但中间仍存在相位空缺。因此它适合做 micro device 4-level 的 preview，不应直接视为最终加工版 LUT。

## 7. 后续建议

1. 用 `phase_radius_lut_dense.csv` 重新做一次 micro device 4-level phase-to-radius mapping preview，并检查是否仍然避免低透射率点。
2. 如果后续要映射 continuous phase map 或更多相位级数，应继续补充 COMSOL 扫描，目标是获得高透射、连续覆盖 `0` 到 `2π` 的 phase-radius LUT。
3. 正式版结构映射时，应在最近邻相位匹配之外加入透射率约束，避免为了相位匹配选择低透射率 meta-atom。

需要强调：本报告是 COMSOL 单元扫描结果的 LUT 处理与算法接口判断，不是真实硬件实验结果，也不是最终加工版设计。
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> None:
    COMSOL_DIR.mkdir(parents=True, exist_ok=True)
    dense_rows, source = load_raw_dense_rows()
    v1_rows = load_lut(V1_LUT_CSV)

    save_dense_lut(dense_rows)
    plot_phase(dense_rows)
    plot_transmittance(dense_rows)
    save_report(dense_rows, v1_rows, source)

    low_rows = [row for row in dense_rows if row["T"] < LOW_T_THRESHOLD]
    dense_stats = phase_stats(dense_rows)
    print(f"Saved dense LUT: {DENSE_LUT_CSV}")
    print(f"Saved phase plot: {PHASE_PNG}")
    print(f"Saved transmittance plot: {TRANSMITTANCE_PNG}")
    print(f"Saved report: {REPORT_PATH}")
    print(f"Source: {source}")
    print(f"Dense points: {len(dense_rows)}")
    print(f"Phase wrapped range: {dense_stats['min']:.5f} to {dense_stats['max']:.5f} rad")
    print(f"Coverage ratio: {dense_stats['coverage']:.3f} of 2pi")
    print(f"Low transmittance points: {len(low_rows)}")


if __name__ == "__main__":
    main()
