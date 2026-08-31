# D2NN 已有结果统一复现摘要

> 本脚本只读取已有 CSV 和 NPY，没有重新训练模型。当前成果属于算法仿真 + COMSOL 单元仿真 + phase-to-radius 结构映射 preview，不是真实硬件实验，也不是最终加工版设计。

## 1. Micro Device Baseline 与量化

- baseline val_acc：`0.993000`。

| phase levels | PTQ val_acc | QAT val_acc |
|---:|---:|---:|
| 2 | 0.353667 | 0.960000 |
| 4 | 0.988667 | 0.994333 |
| 8 | 0.993000 | 0.995667 |
| 16 | 0.993667 | 0.994333 |

已有结果表明，QAT 对低位相位量化尤其重要；2-level 的提升应结合表中已有数值表述，不外推到真实硬件。

## 2. Alignment Shift 鲁棒性

| 对准偏移 | continuous baseline mean_acc |
|---:|---:|
| 0 pixel | 0.993000 |
| 1 pixel | 0.830067 |
| 2 pixel | 0.762267 |
| 4 pixel | 0.482200 |

随着偏移增大，准确率明显下降；在已有采样中，`4` pixel 对应 `0.482200`。因此 alignment shift 是当前最敏感误差，应优先考虑 alignment-aware training 或装调容差设计。

## 3. Dense COMSOL LUT

| 指标 | 数值 |
|---|---:|
| 半径范围 | `80-160 nm` |
| LUT 点数 | `17` |
| phase_wrapped_rad 范围 | `0.21656-6.16256 rad` |
| 相位覆盖比例 | `0.946 of 2π` |
| `T < 0.8` 点数 | `0` |

## 4. Dense Phase-to-Radius Mapping

| 指标 | 数值 |
|---|---:|
| radius map shape | `(3, 128, 128)` |
| 使用半径 | `95, 125, 140, 155 nm` |
| mean mapped transmittance | `0.946660` |
| low-transmittance ratio | `0.000000` |

## 5. 当前局限

1. 分类结果来自合成数据和 PyTorch 算法仿真，不等同于真实器件测试。
2. COMSOL 结果为 meta-atom 单元扫描 LUT，尚不能代表完整 128×128 阵列全波响应。
3. radius map 是最近邻结构映射 preview，没有纳入邻近单元耦合、版图规则和加工误差，不能直接作为最终加工文件。
