# Micro Device 4-level Dense LUT 相位到半径映射报告

## 1. 任务说明

本报告使用新的 dense COMSOL LUT 对 micro device QAT 4-level 的量化相位图进行 phase-to-radius 映射。该结果是结构 preview，不是最终加工版设计，也不是真实硬件验证。

本次没有修改任何 baseline、PTQ、QAT、robustness 训练或评估脚本。

## 2. 输入与输出

| 类型 | 路径 |
|---|---|
| 输入相位图 | `outputs/micro_device_qat/4level/quantized_phase_map.npy`（未纳入公开仓库） |
| dense LUT | `comsol_results/phase_radius_lut_dense.csv` |
| 输出目录 | `comsol_results/mapped_micro_device_4level_dense` |
| radius map | `comsol_results/mapped_micro_device_4level_dense/radius_map.npy`（未纳入公开仓库） |
| preview 图 | `assets/mapping/radius_preview.png` |

## 3. Dense LUT 检查

| 检查项 | 结果 |
|---|---|
| 透射率列 | `T` |
| LUT 点数 | `17` |
| phase_wrapped_rad 范围 | `0.21656` 到 `6.16256` rad |
| 相位覆盖比例 | `0.946` of `2π` |
| `T < 0.8` 点数 | `0` |

## 4. 映射方法

对每个目标相位，在 dense LUT 的 `phase_wrapped_rad` 中寻找圆周相位距离最近的半径：

```text
distance = min(abs(target_phase - lut_phase), 2π - abs(target_phase - lut_phase))
```

## 5. 映射统计

| 指标 | 数值 |
|---|---:|
| phase map shape | `(3, 128, 128)` |
| radius map shape | `(3, 128, 128)` |
| radius min / max | `95` nm / `155` nm |
| 使用到的半径数量 | `4` |
| mean mapped transmittance | `0.946660` |
| min mapped transmittance | `0.937479` |
| low-transmittance pixels | `0` |
| low-transmittance ratio | `0.000000` |
| mean nearest phase error | `0.055797` rad |
| max nearest phase error | `0.120628` rad |
| mean neighbor radius diff | `9.215982` nm |
| max neighbor radius diff | `60.000000` nm |

使用到的半径列表及像素数：

| radius_nm | pixel_count |
|---:|---:|
| 95 | 9307 |
| 125 | 30872 |
| 140 | 8635 |
| 155 | 338 |

## 6. 结论

1. Dense LUT 映射使用到的半径均来自高透射区间，本次 `low-transmittance pixels = 0`，即没有使用 `T < 0.8` 的半径点。
2. Dense LUT 在高透射区间采样更细，且 `T < 0.8` 点数为 `0`、相位覆盖比例为 `0.946`，更适合作为当前展示版本。
3. 该映射仍是结构 preview，不是最终加工版。正式版仍需要结合完整 COMSOL phase-radius LUT、版图生成规则和加工约束。
