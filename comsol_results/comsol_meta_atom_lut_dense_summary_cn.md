# COMSOL 高透射区间 Dense LUT 处理报告

## 1. 数据来源

本报告由 `process_comsol_dense_lut.py` 生成。输入为 COMSOL 高透射区间细扫结果：

```text
comsol_results/raw_radius_sweep_80_160nm_step5.csv
```

读取来源：`raw CSV`。扫描范围为 `80` nm 到 `160` nm，步长约 `5` nm，共 `17` 个点。

本处理只生成 COMSOL LUT、曲线图和报告，没有修改任何 baseline、PTQ、QAT、robustness 训练或评估脚本。

## 2. 输出文件

| 文件 | 说明 |
|---|---|
| `comsol_results/phase_radius_lut_dense.csv` | dense phase-radius LUT |
| `comsol_results/phase_vs_radius_dense.png` | dense 相位-半径曲线 |
| `comsol_results/transmittance_vs_radius_dense.png` | dense 透射率-半径曲线 |
| `comsol_results/comsol_meta_atom_lut_dense_summary_cn.md` | 本报告 |

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
| 点数 | `17` |
| 半径范围 | `80` nm 到 `160` nm |
| T min / max | `0.89600` / `0.99296` |
| phase_wrapped_rad min / max | `0.21656` / `6.16256` |
| phase_wrapped_rad span | `5.94600` rad |
| 相位覆盖比例 | `0.946` of `2π` |
| `T < 0.8` 点数 | `0` |

Dense LUT 中 `T < 0.8` 的点：

| radius_nm | T | phase_rad | phase_wrapped_rad |
|---:|---:|---:|---:|
| 无 | 无 | 无 | 无 |

## 5. 与第一版 LUT 对比

第一版 LUT 文件：

```text
comsol_results/phase_radius_lut.csv
```

| 对比项 | 第一版 LUT | Dense LUT |
|---|---:|---:|
| 点数 | `18` | `17` |
| 半径范围 | `50` 到 `220` nm | `80` 到 `160` nm |
| 半径步长 | 约 `10` nm | 约 `5` nm |
| T min / max | `0.20477` / `0.99099` | `0.89600` / `0.99296` |
| `T < 0.8` 点数 | `6` | `0` |
| phase_wrapped_rad 范围 | `0.21656` 到 `5.75241` | `0.21656` 到 `6.16256` |
| 相位覆盖比例 | `0.881` of `2π` | `0.946` of `2π` |

第一版 LUT 中的低透射率点：

| radius_nm | T | phase_wrapped_rad |
|---:|---:|---:|
| 170 | 0.20477 | 0.27227 |
| 180 | 0.55673 | 5.75241 |
| 190 | 0.61159 | 5.38913 |
| 200 | 0.64291 | 5.11689 |
| 210 | 0.64316 | 4.84389 |
| 220 | 0.78798 | 4.74659 |

## 6. 是否更适合 micro device 4-level phase-to-radius preview

结论：dense LUT 更适合当前 micro device 4-level phase-to-radius preview。

理由如下：

1. Dense LUT 聚焦在 `80` 到 `160` nm 的高透射区间，当前所有点的 `T` 都不低于 `0.8`，最低透射率为 `0.89600`。
2. 第一版 LUT 覆盖半径更宽，但 `170` 到 `220` nm 存在多个低透射率点；这些点虽然扩展了相位范围，但会带来能量损失风险。
3. Dense LUT 半径步长为 `5` nm，比第一版 `10` nm 更细，适合做更平滑、更稳定的最近邻相位匹配。
4. 需要注意，dense LUT 仍然不是完整高透射全相位库。它的 `phase_wrapped_rad` 主要覆盖 `0.21656` 到 `6.16256` rad，虽然数值跨度约为 `94.6%` 的 `2π`，但中间仍存在相位空缺。因此它适合做 micro device 4-level 的 preview，不应直接视为最终加工版 LUT。

## 7. 后续建议

1. 用 `phase_radius_lut_dense.csv` 重新做一次 micro device 4-level phase-to-radius mapping preview，并检查是否仍然避免低透射率点。
2. 如果后续要映射 continuous phase map 或更多相位级数，应继续补充 COMSOL 扫描，目标是获得高透射、连续覆盖 `0` 到 `2π` 的 phase-radius LUT。
3. 正式版结构映射时，应在最近邻相位匹配之外加入透射率约束，避免为了相位匹配选择低透射率 meta-atom。

需要强调：本报告是 COMSOL 单元扫描结果的 LUT 处理与算法接口判断，不是真实硬件实验结果，也不是最终加工版设计。
