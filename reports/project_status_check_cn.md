# D2NN 项目工程状态检查报告

> 本检查只读取已有结果，没有重新训练模型。当前项目定位为算法仿真 + COMSOL 单元仿真 + phase-to-radius 结构映射 preview，不是真实硬件实验，也不是最终加工版设计。

## A. VCSEL near-light

| 检查项 | 状态 | 路径 | 说明 |
|---|---|---|---|
| QAT summary | 通过 | `D2NN-with-Pytorch-main/outputs/vcsel_near_v2_qat/qat_summary.csv` | CSV 可读取，列数 7 |
| robustness results | 通过 | `D2NN-with-Pytorch-main/outputs/vcsel_near_v2_error_robustness/robustness_results.csv` | CSV 可读取，列数 14 |
| robustness curves | 通过 | `D2NN-with-Pytorch-main/outputs/vcsel_near_v2_error_robustness/curves` | 文件齐全 |
| 2-level QAT checkpoint | 通过 | `D2NN-with-Pytorch-main/outputs/vcsel_near_v2_qat/2level/best.pt` | checkpoint 可读取，类型 dict |
| 4-level QAT checkpoint | 通过 | `D2NN-with-Pytorch-main/outputs/vcsel_near_v2_qat/4level/best.pt` | checkpoint 可读取，类型 dict |
| 8-level QAT checkpoint | 通过 | `D2NN-with-Pytorch-main/outputs/vcsel_near_v2_qat/8level/best.pt` | checkpoint 可读取，类型 dict |
| 16-level QAT checkpoint | 通过 | `D2NN-with-Pytorch-main/outputs/vcsel_near_v2_qat/16level/best.pt` | checkpoint 可读取，类型 dict |

## B. micro device

| 检查项 | 状态 | 路径 | 说明 |
|---|---|---|---|
| dataset | 通过 | `data/micro_devices` | 文件齐全 |
| baseline checkpoint | 通过 | `outputs/micro_device_d2nn_train/best.pt` | checkpoint 可读取，类型 dict |
| PTQ quantization results | 通过 | `outputs/micro_device_phase_quantization/quantization_results.csv` | CSV 可读取，列数 5 |
| QAT summary | 通过 | `outputs/micro_device_qat/qat_summary.csv` | CSV 可读取，列数 7 |
| robustness results | 通过 | `outputs/micro_device_error_robustness/robustness_results.csv` | CSV 可读取，列数 14 |
| inspection results | 通过 | `outputs/micro_device_inspection` | 文件齐全 |

## C. COMSOL

| 检查项 | 状态 | 路径 | 说明 |
|---|---|---|---|
| phase_radius_lut.csv | 通过 | `comsol_results/phase_radius_lut.csv` | CSV 可读取，列数 4 |
| phase_radius_lut_dense.csv | 通过 | `comsol_results/phase_radius_lut_dense.csv` | CSV 可读取，列数 4 |
| raw_radius_sweep_80_160nm_step5.csv | 通过 | `comsol_results/raw_radius_sweep_80_160nm_step5.csv` | CSV 可读取，列数 2 |
| phase_vs_radius_dense.png | 通过 | `comsol_results/phase_vs_radius_dense.png` | 存在 |
| transmittance_vs_radius_dense.png | 通过 | `comsol_results/transmittance_vs_radius_dense.png` | 存在 |

## D. mapping

| 检查项 | 状态 | 路径 | 说明 |
|---|---|---|---|
| v1 radius_map.npy | 通过 | `comsol_results/mapped_micro_device_4level_v1/radius_map.npy` | 存在 |
| dense radius_map.npy | 通过 | `comsol_results/mapped_micro_device_4level_dense/radius_map.npy` | 存在 |
| dense radius_preview.png | 通过 | `comsol_results/mapped_micro_device_4level_dense/radius_preview.png` | 存在 |
| dense phase_radius_mapping_report_cn.md | 通过 | `comsol_results/mapped_micro_device_4level_dense/phase_radius_mapping_report_cn.md` | 存在 |

## E. reports

| 检查项 | 状态 | 路径 | 说明 |
|---|---|---|---|
| code_stage_summary_cn.md | 通过 | `reports/code_stage_summary_cn.md` | 存在 |
| final_gap_checklist_cn.md | 通过 | `reports/final_gap_checklist_cn.md` | 存在 |
| final_deliverables_file_list_cn.md | 通过 | `reports/final_deliverables_file_list_cn.md` | 存在 |
| midterm_check_form_text_cn.md | 通过 | `reports/midterm_check_form_text_cn.md` | 存在 |
| next_steps_before_final_submission_cn.md | 通过 | `reports/next_steps_before_final_submission_cn.md` | 存在 |

## 汇总

- 已完成项：27 项。
- 缺失或异常项：0 项。
- 缺失项：本次必检清单内无缺失。
- 建议补齐项：保存并归档 COMSOL `.mph` 模型及仿真设置；制作并检查正式中期/结题 PPT；补充 COMSOL meta-atom 单元场分布图；统一最终提交路径；在报告和 PPT 中明确结构 preview 与真实硬件的边界。
- 当前项目工程完整度评分：**100%**（必检项通过数 / 必检项总数）。

该评分衡量现有代码与结果文件的工程归档完整度，不代表真实硬件或最终加工设计的完成度。
