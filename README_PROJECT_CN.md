# D2NN 项目工程说明

## 1. 项目简介

本项目研究衍射深度神经网络（D2NN）在 VCSEL 近场光场模式识别和 micro device 图样五分类中的算法实现，并建立量化相位到 COMSOL meta-atom 半径参数的映射流程。

当前成果严格定位为：**算法仿真 + COMSOL 单元仿真 + phase-to-radius 结构映射 preview**。项目尚未完成真实硬件实验、完整阵列全波仿真或最终加工版设计。

## 2. 目录结构

| 目录 | 用途 |
|---|---|
| `D2NN-with-Pytorch-main/` | VCSEL 近场算法脚本及历史结果 |
| `data/micro_devices/` | micro device 合成五分类数据集 |
| `outputs/` | micro device baseline、PTQ、QAT、inspection 和 robustness 结果 |
| `comsol_results/` | COMSOL 半径扫描、LUT 和 phase-to-radius 映射结果 |
| `reports/` | 阶段报告、检查报告、复现摘要和成果索引 |
| `presentation_assets/` | 面向汇报 PPT 的统一命名图片 |

## 3. 核心模块说明

1. VCSEL near-light：完成 baseline、PTQ、QAT 和误差鲁棒性算法仿真。
2. micro device：完成数据集生成与检查、baseline、PTQ、QAT、分类检查和误差鲁棒性算法仿真。
3. COMSOL meta-atom：完成单元半径扫描数据后处理及 v1/dense phase-radius LUT。
4. phase-to-radius：将离散量化相位按圆周相位距离映射为 LUT 中最接近的纳米柱半径。

## 4. 主要脚本说明

| 脚本 | 作用 |
|---|---|
| `project_paths.py` | 统一定义项目目录，供新增工程脚本使用 |
| `check_project_status.py` | 检查 CSV、checkpoint、图片、LUT、mapping 和报告完整性 |
| `run_reproduce_summary.py` | 只读现有结果并生成统一核心数值摘要 |
| `build_result_index.py` | 扫描成果目录并生成文件用途与提交/PPT 建议索引 |
| `process_comsol_radius_sweep.py` | 处理第一版 COMSOL 半径扫描数据 |
| `process_comsol_dense_lut.py` | 处理 80-160 nm、5 nm 步长的 dense LUT |
| `map_phase_to_radius.py` | 参数化执行通用相位到半径最近邻映射 |

旧 baseline、PTQ、QAT 和 robustness 脚本保持原样；上述新入口不会调用训练流程。

## 5. 已完成实验结果

以下数值来自已有 CSV/NPY，并可通过 `run_reproduce_summary.py` 重新读取核对：

- micro device baseline val_acc 为 `0.993000`。
- micro device 2/4/8/16-level PTQ val_acc 分别为 `0.353667`、`0.988667`、`0.993000`、`0.993667`。
- micro device 2/4/8/16-level QAT val_acc 分别为 `0.960000`、`0.994333`、`0.995667`、`0.994333`。
- dense LUT 覆盖 80-160 nm，共 17 点，相位覆盖比例为 `0.946 of 2π`，`T < 0.8` 点数为 0。
- dense radius map shape 为 `(3, 128, 128)`，使用 95、125、140、155 nm，mean mapped transmittance 为 `0.946660`，low-transmittance ratio 为 0。
- 已有 robustness CSV 显示 alignment shift 是当前最敏感误差。

## 6. 如何复现结果汇总

以下命令只读取已有结果，不训练模型：

```powershell
python check_project_status.py
python run_reproduce_summary.py
python build_result_index.py
```

对应输出为 `reports/project_status_check_cn.md`、`reports/reproduce_summary_cn.md` 和 `reports/result_index_cn.md`。

## 7. 如何处理 COMSOL LUT

第一版扫描和 dense 扫描分别由以下脚本处理：

```powershell
python scripts/process_comsol_radius_sweep.py
python scripts/process_comsol_dense_lut.py
```

这两个脚本读取 COMSOL 导出的半径、透射率和相位数据。运行前应确认原始 CSV 来自已保存的 COMSOL 单元模型；本工程整理不重新执行 COMSOL 求解。

## 8. 如何执行 Phase-to-Radius 映射

```powershell
python map_phase_to_radius.py `
  --phase-map outputs\micro_device_qat\4level\quantized_phase_map.npy `
  --lut comsol_results\phase_radius_lut_dense.csv `
  --output-dir comsol_results\mapped_micro_device_4level_dense_unified `
  --label "Micro Device 4-level Dense LUT" `
  --low-t-threshold 0.8
```

该命令生成 `radius_map.npy`、`radius_preview.png` 和 `phase_radius_mapping_report_cn.md`，不会覆盖原 `mapped_micro_device_4level_dense`。

## 9. 当前局限

1. 分类任务使用合成数据，准确率不能直接等同于真实器件性能。
2. robustness 是算法扰动模拟，不是机械装调或硬件误差实测。
3. COMSOL 仅完成 meta-atom 单元扫描，当前目录仍需归档 `.mph` 模型、材料参数、边界条件和网格设置。
4. radius map 未考虑阵列耦合、版图设计规则、最小间距和制造公差，只能作为结构 preview。

## 10. 下一步计划

必须完成正式报告和 PPT 检查、COMSOL 模型归档、CSV/图片统一保存、路径核对和局限说明。建议进一步开展 VCSEL 4-level 半径映射、更多 COMSOL 参数组合、alignment-aware training 和更完整的 phase coverage 优化。完整 128×128 阵列 COMSOL 全波仿真、真实器件加工及所有量化模型的半径映射不属于当前阶段必须项。
