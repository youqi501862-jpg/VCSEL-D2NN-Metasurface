# 大创代码与仿真阶段成果总报告

更新日期：2026-08-09
原始项目目录：本仓库未包含本机绝对路径；公开内容位于仓库根目录。

> 本报告所有数值来自项目中已有 CSV、Markdown 报告、PNG 和 NPY 文件。本次整理未重新训练模型，未修改 baseline、PTQ、QAT 或 robustness 脚本的核心逻辑。当前成果属于**算法仿真 + COMSOL meta-atom 单元仿真 + phase-to-radius 结构映射 preview**，不是真实硬件实验，也不是最终加工版设计。

## 1. 项目目标

项目面向 VCSEL 近场模式识别和微器件结构图样识别，研究基于衍射深度神经网络（D2NN）的光学分类方法，并进一步考察离散相位量化、制造误差敏感性以及算法相位图到纳米柱半径参数的映射可行性。阶段目标不是完成真实器件加工，而是建立从算法模型到单元结构参数 preview 的可追溯流程。

## 2. 总体技术路线

```text
合成数据集
→ 连续相位 D2NN baseline
→ prediction inspection
→ PTQ 训练后量化
→ QAT 量化感知训练
→ phase / height / alignment 误差鲁棒性评估
→ COMSOL meta-atom 半径扫描
→ phase-radius LUT 后处理
→ 量化相位图到 radius map 的结构映射 preview
```

D2NN 计算链路使用输入强度开方得到场振幅，通过 FFT/角谱传播和相位层调制，最后在探测器区域统计输出能量并完成分类。COMSOL 部分提供单元结构的透射率与相位响应，映射脚本使用圆周相位距离把离散相位匹配到 LUT 半径。

## 3. VCSEL 近场光场模式识别结果

数据集位于 `D2NN-with-Pytorch-main/data/vcsel_near_synth`，包含 fundamental、first_order、second_order、third_order、fourth_order 五类；训练集 24000 张，验证集 6000 张。

| 方法 | val_acc | val_loss | 数据来源 |
|---|---:|---:|---|
| continuous baseline | 1.000000 | 0.000003168（第 30 epoch） | `outputs/vcsel_near_v2_train/training_history.csv` |
| PTQ 2-level | 0.160667 | 106.804175 | `outputs/vcsel_near_v2_quantization/quantization_results.csv` |
| PTQ 4-level | 0.239833 | 17.254093 | 同上 |
| PTQ 8-level | 0.984167 | 0.062592 | 同上 |
| PTQ 16-level | 0.999667 | 0.000873 | 同上 |
| QAT 2-level | 1.000000 | 0.000639 | `outputs/vcsel_near_v2_qat/qat_summary.csv` |
| QAT 4-level | 1.000000 | 0.000284 | 同上 |
| QAT 8-level | 1.000000 | 0.000310 | 同上 |
| QAT 16-level | 1.000000 | 0.000257 | 同上 |

VCSEL 结果说明：当前合成数据上的连续相位模型已经充分收敛；2-level 和 4-level PTQ 会严重破坏相位分布，而 QAT 能在本数据集上恢复低位量化性能。该高准确率不能外推为真实 VCSEL 硬件识别精度。

## 4. micro device 五分类识别结果

micro device 数据集包含 diode、BJT、NMOS、PMOS 和 resistor 五类，训练集 15000 张、验证集 3000 张，图像尺寸为 `128×128`。数据为规则生成的简化结构图样，不是真实版图或显微图。

| 指标 | 数值 | 来源 |
|---|---:|---|
| baseline train_acc | 0.998800（第 28 epoch） | `outputs/micro_device_d2nn_train/training_history.csv` |
| baseline val_acc | **0.993000** | 同上 |
| baseline val_loss | 0.040183 | 同上 |
| inspection overall accuracy | 0.993000 | `reports/micro_device_baseline_result_cn.md` 与 inspection CSV |
| inspection correct/total | 2979/3000 | 同上 |

预测检查显示主要错误集中在 NMOS 与 PMOS 等结构相近类别之间。baseline 结果证明当前 D2NN 仿真链路能够学习规则生成图样的类别差异，但尚不能代表真实芯片图像识别性能。

## 5. PTQ 与 QAT 对比

| 方法 | micro device val_acc | 相对判断 |
|---|---:|---|
| continuous baseline | 0.993000 | 连续相位参考 |
| PTQ 2-level | **0.353667** | 严重退化 |
| QAT 2-level | **0.960000** | 较 PTQ 提升 0.606333 |
| PTQ 4-level | 0.988667 | 基本可用 |
| QAT 4-level | **0.994333** | 达到/略高于 baseline |
| PTQ 8-level | 0.993000 | 接近 baseline |
| QAT 8-level | **0.995667** | 当前最高值 |
| PTQ 16-level | 0.993667 | 接近 baseline |
| QAT 16-level | **0.994333** | 保持高准确率 |

两条任务线共同表明：训练后直接低位量化可能导致明显性能损失，而把离散相位约束纳入训练过程的 QAT 能显著改善低位相位量化性能。该结论基于现有仿真数据与单次训练结果。

## 6. 误差鲁棒性分析

现有评估对每种扰动强度执行 5 次 Monte Carlo，包含 phase noise、height noise 和 alignment shift。关键点如下。

| 任务/模型 | phase noise 0.2 rad | height noise 40 nm | shift 1 px | shift 2 px | shift 4 px |
|---|---:|---:|---:|---:|---:|
| VCSEL continuous | 0.999367 | 0.999933 | 0.968033 | 0.668400 | 0.411067 |
| VCSEL QAT 4-level | 0.998867 | 0.999833 | 0.936467 | 0.667900 | 0.537800 |
| micro continuous | 0.991200 | 0.992133 | 0.830067 | 0.762267 | 0.482200 |
| micro QAT 4-level | 0.992267 | 0.992200 | 0.702800 | 0.643933 | 0.450933 |
| micro QAT 8-level | 0.993667 | 0.995667 | 0.783067 | 0.645733 | 0.528267 |

在当前扰动定义下，phase noise 和 height noise 对多数高性能模型影响较小，而 alignment shift 会系统性改变层间光场和探测器能量分布，是当前最敏感误差。该结果应解释为算法鲁棒性评估，不是真实制造误差统计。

## 7. COMSOL meta-atom 单元仿真

项目已有两组半径扫描 CSV：

- v1 原始扫描：`comsol_results/raw_radius_sweep_50_220nm.csv`。
- dense 原始扫描：`comsol_results/raw_radius_sweep_80_160nm_step5.csv`。

后处理脚本分别为 `process_comsol_radius_sweep.py` 和 `process_comsol_dense_lut.py`，输出半径、透射率 `T`、相位 `phase_rad` 与包裹相位 `phase_wrapped_rad`。这部分属于 meta-atom 单元响应仿真与数据后处理。当前目录未发现 COMSOL `.mph` 模型文件，也未发现 `comsol_meta_atom_field.png`，因此最终提交前必须补充模型归档和仿真设置说明。

## 8. dense LUT 结果

| 指标 | dense LUT 结果 |
|---|---:|
| 半径范围 | **80–160 nm** |
| 步长 | **5 nm** |
| 点数 | **17** |
| phase_wrapped_rad 范围 | **0.21656–6.16256 rad** |
| 相位覆盖比例 | **0.946 of 2π** |
| `T < 0.8` 点数 | **0** |

相比 v1 LUT（50–220 nm、18 点、相位覆盖约 0.881 of 2π、`T<0.8` 点数 6），dense LUT 集中在高透射率区间，采样更细，更适合作为当前阶段展示版 LUT。但它仍不是包含全部材料、几何和加工约束的最终单元库。

## 9. phase-to-radius 半径映射

映射输入为 `outputs/micro_device_qat/4level/quantized_phase_map.npy`，其 shape 为 `(3,128,128)`。dense 映射结果如下。

| 指标 | dense mapping |
|---|---:|
| radius map shape | **(3, 128, 128)** |
| 使用半径 | **95, 125, 140, 155 nm** |
| 最小/最大半径 | 95/155 nm |
| mean mapped transmittance | **0.946660** |
| low-transmittance pixels | 0 |
| low-transmittance ratio | **0** |
| mean nearest phase error | 0.055797 rad |
| max nearest phase error | 0.120628 rad |

NPY 实际检查显示 radius map 由四个离散半径构成，像素数分别为 9307、30872、8635 和 338。该 radius map 只代表基于单元 LUT 最近相位匹配得到的结构 preview，不包含邻近单元耦合、版图设计规则、最小间距、工艺偏差和阵列级全波验证，不能作为最终加工文件。

## 10. 当前局限

1. 两类识别数据均为合成或规则生成数据，没有真实实验采集数据验证。
2. D2NN 传播、相位量化和误差扰动均在 PyTorch 中模拟，不等价于真实光学平台。
3. COMSOL 部分为 meta-atom 单元扫描，尚未归档原始 `.mph` 模型和完整仿真设置。
4. dense LUT 相位覆盖为 0.946 of 2π，仍存在相位缺口。
5. radius map 为结构映射 preview，不包含阵列耦合和最终加工约束。
6. alignment shift 仅完成评估，尚未系统进入训练。
7. 现有高准确率主要来自合成任务，不宜直接与真实硬件性能比较。

## 11. 下一步计划

1. 完成最终报告、正式 PPT 和全量结果归档。
2. 保存 COMSOL `.mph` 模型、材料参数、边界条件、网格和 sweep 设置。
3. 完成 VCSEL 4-level phase-to-radius 映射作为可选增强。
4. 扩展 COMSOL 参数组合，优化高透射率条件下的完整 `2π` 相位覆盖。
5. 开展 alignment-aware training 或层间错位增强。
6. 在后续条件允许时，引入真实 VCSEL 近场或真实器件图样数据验证。

## 12. 阶段性结论

项目已经形成两条五分类算法链路的 baseline、PTQ、QAT 和 robustness 闭环，并完成 COMSOL 单元 LUT 到 micro device 4-level radius map 的结构映射 preview。micro device 2-level PTQ 从 `0.353667` 经 QAT 提升到 `0.960000`，验证了 QAT 对低位相位量化的明显改善；误差分析识别出 alignment shift 为当前最主要风险；dense LUT 在 80–160 nm 高透射区间实现 0.946 of 2π 相位覆盖，dense mapping 平均映射透射率为 `0.946660` 且 low-transmittance ratio 为 0。

因此，当前阶段可以表述为：**算法仿真闭环已完成，COMSOL 单元仿真和结构映射预览已打通，具备中期/阶段性成果汇报条件；真实硬件实验、完整阵列仿真和最终加工设计尚未完成。**
