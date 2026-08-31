# 核心结果摘要

> 所有数值均来自原项目已有 CSV、Markdown 报告或 NPY 文件。本仓库没有重新训练模型。当前成果属于算法仿真、COMSOL meta-atom 单元仿真和 phase-to-radius 结构映射 preview，不是真实硬件实验，也不是最终加工版设计。

## Micro Device 五分类

| 方法 | phase levels | val_acc |
|---|---:|---:|
| Continuous baseline | continuous | `0.993000` |
| PTQ | 2 | `0.353667` |
| PTQ | 4 | `0.988667` |
| PTQ | 8 | `0.993000` |
| PTQ | 16 | `0.993667` |
| QAT | 2 | `0.960000` |
| QAT | 4 | `0.994333` |
| QAT | 8 | `0.995667` |
| QAT | 16 | `0.994333` |

2-level PTQ 出现明显性能下降，而 2-level QAT 将验证准确率从 `0.353667` 提升到 `0.960000`，说明 QAT 能显著改善低位相位量化性能。

## 误差鲁棒性

已有 phase noise、height noise 和 alignment shift 仿真结果表明，**alignment shift 是当前最敏感误差**。micro device continuous baseline 在 0/1/2/4 pixel 偏移下的 mean accuracy 分别为 `0.993000`、`0.830067`、`0.762267` 和 `0.482200`。

## Dense COMSOL LUT

| 指标 | 数值 |
|---|---:|
| radius 范围 | `80-160 nm` |
| 半径步长 | `5 nm` |
| 采样点数 | `17` |
| phase_wrapped_rad 范围 | `0.21656-6.16256 rad` |
| 相位覆盖比例 | `0.946 of 2π` |
| `T < 0.8` 点数 | `0` |

该 LUT 来自 COMSOL meta-atom 单元扫描结果，不代表完整阵列或真实器件响应。

## Dense Phase-to-Radius Mapping

| 指标 | 数值 |
|---|---:|
| radius map shape | `(3, 128, 128)` |
| 使用半径 | `95, 125, 140, 155 nm` |
| mean mapped transmittance | `0.946660` |
| low-transmittance ratio | `0` |

该映射使用最近圆周相位匹配，仅用于结构接口验证和展示。它没有包含邻近单元耦合、完整阵列全波仿真、版图规则或加工公差，因此不能直接作为最终加工文件。
