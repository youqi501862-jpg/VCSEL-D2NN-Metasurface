from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "comsol_results"
REPORT_DIR = PROJECT_ROOT / "reports"

LUT_CSV = OUTPUT_DIR / "phase_radius_lut.csv"
PHASE_PNG = OUTPUT_DIR / "phase_vs_radius.png"
TRANSMITTANCE_PNG = OUTPUT_DIR / "transmittance_vs_radius.png"
REPORT_PATH = REPORT_DIR / "comsol_meta_atom_lut_summary_cn.md"

LOW_T_THRESHOLD = 0.8

# COMSOL radius sweep. The values provided by the user are in nm; here they are
# stored as meters first, then converted back to nm to keep the unit conversion
# step explicit in the processing script.
RAW_SWEEP = [
    {"radius_m": 50e-9, "T": 0.94074, "phase_rad": 2.8958},
    {"radius_m": 60e-9, "T": 0.94425, "phase_rad": 2.8251},
    {"radius_m": 70e-9, "T": 0.94704, "phase_rad": 2.7308},
    {"radius_m": 80e-9, "T": 0.94805, "phase_rad": 2.6074},
    {"radius_m": 90e-9, "T": 0.94565, "phase_rad": 2.4483},
    {"radius_m": 100e-9, "T": 0.93925, "phase_rad": 2.2460},
    {"radius_m": 110e-9, "T": 0.93168, "phase_rad": 1.9911},
    {"radius_m": 120e-9, "T": 0.93122, "phase_rad": 1.6733},
    {"radius_m": 130e-9, "T": 0.94960, "phase_rad": 1.2784},
    {"radius_m": 140e-9, "T": 0.98257, "phase_rad": 0.79369},
    {"radius_m": 150e-9, "T": 0.99099, "phase_rad": 0.21656},
    {"radius_m": 160e-9, "T": 0.89600, "phase_rad": -0.56862},
    {"radius_m": 170e-9, "T": 0.20477, "phase_rad": 0.27227},
    {"radius_m": 180e-9, "T": 0.55673, "phase_rad": -0.53078},
    {"radius_m": 190e-9, "T": 0.61159, "phase_rad": -0.89406},
    {"radius_m": 200e-9, "T": 0.64291, "phase_rad": -1.1663},
    {"radius_m": 210e-9, "T": 0.64316, "phase_rad": -1.4393},
    {"radius_m": 220e-9, "T": 0.78798, "phase_rad": -1.5366},
]


def wrap_phase_0_2pi(phase_rad: float) -> float:
    return phase_rad % (2.0 * math.pi)


def build_lut() -> list[dict[str, float]]:
    rows = []
    for item in RAW_SWEEP:
        radius_nm = item["radius_m"] * 1e9
        phase_rad = item["phase_rad"]
        rows.append(
            {
                "radius_nm": radius_nm,
                "T": item["T"],
                "phase_rad": phase_rad,
                "phase_wrapped_rad": wrap_phase_0_2pi(phase_rad),
            }
        )
    return rows


def save_lut(rows: list[dict[str, float]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with LUT_CSV.open("w", encoding="utf-8", newline="") as f:
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
    phase_wrapped = [row["phase_wrapped_rad"] for row in rows]
    low_t_radius = [row["radius_nm"] for row in rows if row["T"] < LOW_T_THRESHOLD]
    low_t_phase = [row["phase_wrapped_rad"] for row in rows if row["T"] < LOW_T_THRESHOLD]

    plt.figure(figsize=(7.5, 4.8))
    plt.plot(radius, phase, "o-", label="phase_rad")
    plt.plot(radius, phase_wrapped, "s--", label="phase_wrapped_rad (0 to 2pi)")
    if low_t_radius:
        plt.scatter(
            low_t_radius,
            low_t_phase,
            c="red",
            marker="x",
            s=70,
            label=f"T < {LOW_T_THRESHOLD}",
            zorder=5,
        )
    plt.xlabel("Radius (nm)")
    plt.ylabel("Phase (rad)")
    plt.title("COMSOL Meta-Atom Phase vs Radius")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PHASE_PNG, dpi=200)
    plt.close()


def plot_transmittance(rows: list[dict[str, float]]) -> None:
    radius = [row["radius_nm"] for row in rows]
    transmittance = [row["T"] for row in rows]
    low_t_radius = [row["radius_nm"] for row in rows if row["T"] < LOW_T_THRESHOLD]
    low_t_values = [row["T"] for row in rows if row["T"] < LOW_T_THRESHOLD]

    plt.figure(figsize=(7.5, 4.8))
    plt.plot(radius, transmittance, "o-", label="T")
    plt.axhline(
        LOW_T_THRESHOLD,
        color="red",
        linestyle="--",
        linewidth=1.2,
        label=f"T = {LOW_T_THRESHOLD}",
    )
    if low_t_radius:
        plt.scatter(
            low_t_radius,
            low_t_values,
            c="red",
            marker="x",
            s=70,
            label=f"T < {LOW_T_THRESHOLD}",
            zorder=5,
        )
    plt.xlabel("Radius (nm)")
    plt.ylabel("Transmittance")
    plt.title("COMSOL Meta-Atom Transmittance vs Radius")
    plt.ylim(0.0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(TRANSMITTANCE_PNG, dpi=200)
    plt.close()


def save_report(rows: list[dict[str, float]]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    low_t_rows = [row for row in rows if row["T"] < LOW_T_THRESHOLD]
    high_t_rows = [row for row in rows if row["T"] >= LOW_T_THRESHOLD]
    phase_values = [row["phase_wrapped_rad"] for row in rows]

    low_t_table = "\n".join(
        f"| {row['radius_nm']:.0f} | {row['T']:.5f} | {row['phase_rad']:.5f} | {row['phase_wrapped_rad']:.5f} |"
        for row in low_t_rows
    )
    if not low_t_table:
        low_t_table = "| 无 | 无 | 无 | 无 |"

    report = f"""# COMSOL 纳米柱相位-半径 LUT 处理报告

## 1. 数据来源

本报告由 `process_comsol_radius_sweep.py` 根据 COMSOL 纳米柱半径扫描结果生成。输入扫描点共 `{len(rows)}` 个，半径范围为 `{rows[0]['radius_nm']:.0f}` nm 到 `{rows[-1]['radius_nm']:.0f}` nm。

脚本中将半径先按米 `m` 保存，再统一换算为 `nm` 输出到 LUT 文件，避免后续 D2NN 相位图映射时出现单位混淆。

## 2. 输出文件

| 文件 | 说明 |
|---|---|
| `{LUT_CSV}` | phase-radius LUT，包含 `radius_nm`、`T`、`phase_rad`、`phase_wrapped_rad` |
| `{PHASE_PNG}` | 相位随半径变化曲线 |
| `{TRANSMITTANCE_PNG}` | 透射率随半径变化曲线，并标出低透射率点 |

## 3. LUT 字段说明

| 字段 | 含义 |
|---|---|
| `radius_nm` | 纳米柱半径，单位 nm |
| `T` | COMSOL 计算得到的透射率 |
| `phase_rad` | COMSOL 原始相位，单位 rad |
| `phase_wrapped_rad` | 将 `phase_rad` wrap 到 `[0, 2π)` 后的相位，单位 rad |

当前 `phase_wrapped_rad` 范围为 `{min(phase_values):.5f}` 到 `{max(phase_values):.5f}` rad。

## 4. 低透射率半径点

透射率阈值设为 `T < {LOW_T_THRESHOLD}`。低透射率点如下：

| radius_nm | T | phase_rad | phase_wrapped_rad |
|---:|---:|---:|---:|
{low_t_table}

低透射率点数量为 `{len(low_t_rows)}` 个；透射率不低于 `{LOW_T_THRESHOLD}` 的点数量为 `{len(high_t_rows)}` 个。

## 5. 初步判断

1. 半径 `50` 到 `160` nm 区间整体透射率较高，其中 `150` nm 附近透射率达到 `0.99099`。
2. 半径 `170` nm 之后出现明显低透射率点，尤其 `170` nm 的透射率仅为 `0.20477`，需要谨慎用于 phase map 到结构半径的映射。
3. 后续生成 nanopillar radius map 时，应优先在高透射率区间内选择满足目标相位的半径；若某些目标相位只能由低透射率半径实现，需要在报告中说明能量损失风险。
4. 该 LUT 是 COMSOL 单元扫描结果到算法 phase map 的接口雏形，仍需要结合实际周期、材料折射率、柱高、偏振、边界条件等仿真设置进一步确认。

## 6. 与 D2NN 后续工作的关系

该 LUT 可用于后续将 D2NN 训练得到的 `phase_map.npy` 映射为纳米柱半径分布。建议下一步实现：

```text
phase_map.npy
-> wrap 到 [0, 2π)
-> 根据 phase_radius_lut.csv 做最近邻或插值匹配
-> 得到 nanopillar_radius_map.npy / .csv
-> 检查对应透射率分布
```

需要注意：当前 LUT 只说明相位-半径对应关系，还不等同于完整可加工版图，也不是真实硬件验证结果。
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> None:
    rows = build_lut()
    save_lut(rows)
    plot_phase(rows)
    plot_transmittance(rows)
    save_report(rows)

    low_t = [row for row in rows if row["T"] < LOW_T_THRESHOLD]
    print(f"Saved LUT: {LUT_CSV}")
    print(f"Saved phase plot: {PHASE_PNG}")
    print(f"Saved transmittance plot: {TRANSMITTANCE_PNG}")
    print(f"Saved report: {REPORT_PATH}")
    print(
        "Low transmittance radii (T < 0.8): "
        + ", ".join(f"{row['radius_nm']:.0f} nm" for row in low_t)
    )


if __name__ == "__main__":
    main()
