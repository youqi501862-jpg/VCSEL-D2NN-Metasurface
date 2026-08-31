# 大创最终成果文件清单

更新日期：2026-08-09
基准目录：仓库根目录

> 本清单列出适合交付、复核或归档的正式脚本与结果。原始训练数据图片、smoke 临时结果和缓存文件不作为主要交付物。当前未发现正式 PPT 文件和 COMSOL `.mph` 模型文件。

## 1. 训练脚本

| 文件 | 用途 |
|---|---|
| `D2NN-with-Pytorch-main/run_vcsel_near_light_v2_train.py` | 训练 VCSEL near-light 连续相位 baseline。 |
| `D2NN-with-Pytorch-main/run_vcsel_near_light_v2_qat.py` | 训练 VCSEL 2/4/8/16-level QAT 模型。 |
| `D2NN-with-Pytorch-main/run_micro_device_d2nn_train.py` | 训练 micro device 五分类 baseline。 |
| `D2NN-with-Pytorch-main/run_micro_device_d2nn_qat.py` | 训练 micro device 2/4/8/16-level QAT 模型。 |
| `D2NN-with-Pytorch-main/generate_vcsel_near light_dataset.py` | 生成 VCSEL near-light 合成数据集。 |
| `D2NN-with-Pytorch-main/generate_micro_device_dataset.py` | 生成 micro device 五分类规则图样数据集。 |

## 2. 评估与检查脚本

| 文件 | 用途 |
|---|---|
| `D2NN-with-Pytorch-main/evaluate_phase_quantization_v2.py` | 对 VCSEL baseline 执行 2/4/8/16-level PTQ 评估。 |
| `D2NN-with-Pytorch-main/evaluate_v2_qat_error_robustness.py` | 评估 VCSEL baseline/PTQ/QAT 的 phase、height 和 alignment 鲁棒性。 |
| `D2NN-with-Pytorch-main/evaluate_micro_device_phase_quantization.py` | 对 micro device baseline 执行 PTQ 评估。 |
| `D2NN-with-Pytorch-main/evaluate_micro_device_error_robustness.py` | 评估 micro device 各模型的误差鲁棒性。 |
| `D2NN-with-Pytorch-main/inspect_micro_device_predictions.py` | 导出 micro device 正确/错误样本、混淆矩阵和分类指标。 |
| `D2NN-with-Pytorch-main/recover_micro_device_training_outputs.py` | 从已有 checkpoint 恢复 micro baseline 的结果文件，不用于重新训练。 |

## 3. COMSOL 后处理脚本

| 文件 | 用途 |
|---|---|
| `process_comsol_radius_sweep.py` | 将 v1 半径扫描数据处理为 phase-radius LUT 和曲线。 |
| `process_comsol_dense_lut.py` | 处理 80–160 nm、5 nm 步长的 dense LUT 并生成统计与图片。 |

## 4. phase-to-radius 映射脚本

| 文件 | 用途 |
|---|---|
| `map_phase_to_radius_preview.py` | 早期 phase-to-radius 映射 preview。 |
| `map_phase_to_radius_v1.py` | 使用 v1 LUT 将 micro 4-level 相位图映射为半径图。 |
| `map_phase_to_radius_dense.py` | 使用 dense LUT 生成正式阶段展示用 radius map 和报告。 |

## 5. CSV 结果文件

### VCSEL near-light

| 文件 | 用途 |
|---|---|
| `D2NN-with-Pytorch-main/outputs/vcsel_near_v2_train/training_history.csv` | baseline 各 epoch 训练/验证指标。 |
| `D2NN-with-Pytorch-main/outputs/vcsel_near_v2_quantization/quantization_results.csv` | continuous 与 2/4/8/16-level PTQ 汇总。 |
| `D2NN-with-Pytorch-main/outputs/vcsel_near_v2_quantization/confusion_matrices/continuous_confusion_matrix.csv` | continuous 混淆矩阵。 |
| `.../confusion_matrices/2level_confusion_matrix.csv` | 2-level PTQ 混淆矩阵。 |
| `.../confusion_matrices/4level_confusion_matrix.csv` | 4-level PTQ 混淆矩阵。 |
| `.../confusion_matrices/8level_confusion_matrix.csv` | 8-level PTQ 混淆矩阵。 |
| `.../confusion_matrices/16level_confusion_matrix.csv` | 16-level PTQ 混淆矩阵。 |
| `D2NN-with-Pytorch-main/outputs/vcsel_near_v2_qat/qat_summary.csv` | 2/4/8/16-level QAT 汇总。 |
| `D2NN-with-Pytorch-main/outputs/vcsel_near_v2_qat/{2,4,8,16}level/training_history.csv` | 各 QAT 模型训练历史，每个 level 各一份。 |
| `D2NN-with-Pytorch-main/outputs/vcsel_near_v2_qat/{2,4,8,16}level/confusion_matrix.csv` | 各 QAT 模型混淆矩阵，每个 level 各一份。 |
| `D2NN-with-Pytorch-main/outputs/vcsel_near_v2_error_robustness/robustness_results.csv` | VCSEL 三类误差扰动的完整 Monte Carlo 结果。 |

### micro device

| 文件 | 用途 |
|---|---|
| `outputs/micro_device_d2nn_train/training_history.csv` | micro baseline 各 epoch 指标。 |
| `outputs/micro_device_inspection/confusion_matrix.csv` | baseline 预测检查混淆矩阵。 |
| `outputs/micro_device_inspection/classification_metrics.csv` | 五类 precision、recall、F1。 |
| `outputs/micro_device_phase_quantization/quantization_results.csv` | continuous 与 2/4/8/16-level PTQ 汇总。 |
| `outputs/micro_device_phase_quantization/confusion_matrices/{continuous,2level,4level,8level,16level}_confusion_matrix.csv` | PTQ 各模式混淆矩阵。 |
| `outputs/micro_device_phase_quantization/classification_metrics/{continuous,2level,4level,8level,16level}_classification_metrics.csv` | PTQ 各模式分类指标。 |
| `outputs/micro_device_qat/qat_summary.csv` | micro 2/4/8/16-level QAT 汇总。 |
| `outputs/micro_device_qat/{2,4,8,16}level/training_history.csv` | 各 QAT 模型训练历史。 |
| `outputs/micro_device_qat/{2,4,8,16}level/confusion_matrix.csv` | 各 QAT 模型混淆矩阵。 |
| `outputs/micro_device_qat/{2,4,8,16}level/classification_metrics.csv` | 各 QAT 模型分类指标。 |
| `outputs/micro_device_error_robustness/robustness_results.csv` | micro 三类误差扰动的完整 Monte Carlo 结果。 |

### COMSOL 与映射

| 文件 | 用途 |
|---|---|
| `comsol_results/raw_radius_sweep_50_220nm.csv` | v1 COMSOL 半径扫描原始导出。 |
| `comsol_results/phase_radius_lut.csv` | v1 phase-radius LUT。 |
| `comsol_results/raw_radius_sweep_80_160nm_step5.csv` | dense COMSOL 半径扫描原始导出。 |
| `comsol_results/phase_radius_lut_dense.csv` | dense phase-radius LUT，17 点。 |

## 6. 图片结果

| 文件 | 用途 |
|---|---|
| `D2NN-with-Pytorch-main/outputs/vcsel_near_v2_train/training_curves.png` | VCSEL baseline 收敛曲线。 |
| `D2NN-with-Pytorch-main/outputs/vcsel_near_v2_quantization/quantization_accuracy_curve.png` | VCSEL PTQ level-accuracy 对比。 |
| `D2NN-with-Pytorch-main/outputs/vcsel_near_v2_quantization/confusion_matrices/*_confusion_matrix.png` | VCSEL continuous/PTQ 混淆矩阵组。 |
| `D2NN-with-Pytorch-main/outputs/vcsel_near_v2_qat/{2,4,8,16}level/training_curves.png` | VCSEL QAT 各 level 收敛曲线。 |
| `D2NN-with-Pytorch-main/outputs/vcsel_near_v2_qat/{2,4,8,16}level/confusion_matrix.png` | VCSEL QAT 各 level 混淆矩阵。 |
| `D2NN-with-Pytorch-main/outputs/vcsel_near_v2_error_robustness/curves/accuracy_vs_{phase_noise,height_noise,alignment_shift}.png` | VCSEL 三类鲁棒性曲线。 |
| `outputs/micro_device_dataset_preview/preview_grid.png` | micro 五分类数据集预览。 |
| `outputs/micro_device_d2nn_train/training_curves.png` | micro baseline 收敛曲线。 |
| `outputs/micro_device_inspection/confusion_matrix.png` | micro baseline 混淆矩阵。 |
| `outputs/micro_device_inspection/correct_samples/` | 各类正确预测示例。 |
| `outputs/micro_device_inspection/wrong_samples/` | 典型错误预测示例。 |
| `outputs/micro_device_phase_quantization/quantization_accuracy_curve.png` | micro PTQ level-accuracy 对比。 |
| `outputs/micro_device_phase_quantization/confusion_matrices/*_confusion_matrix.png` | micro continuous/PTQ 混淆矩阵组。 |
| `outputs/micro_device_qat/{2,4,8,16}level/training_curves.png` | micro QAT 各 level 收敛曲线。 |
| `outputs/micro_device_qat/{2,4,8,16}level/confusion_matrix.png` | micro QAT 各 level 混淆矩阵。 |
| `outputs/micro_device_error_robustness/curves/accuracy_vs_{phase_noise,height_noise,alignment_shift}.png` | micro 三类鲁棒性曲线。 |
| `comsol_results/phase_vs_radius.png` | v1 半径-相位曲线。 |
| `comsol_results/transmittance_vs_radius.png` | v1 半径-透射率曲线。 |
| `comsol_results/phase_vs_radius_dense.png` | dense 半径-相位曲线。 |
| `comsol_results/transmittance_vs_radius_dense.png` | dense 半径-透射率曲线。 |
| `comsol_results/mapped_micro_device_4level_v1/radius_preview.png` | v1 结构映射预览。 |
| `comsol_results/mapped_micro_device_4level_dense/radius_preview.png` | dense 结构映射预览。 |
| `presentation_assets/*.png` | 面向 PPT 的统一命名图片资产。 |

## 7. 报告文件

| 文件 | 用途 |
|---|---|
| `reports/code_stage_summary_cn.md` | 当前阶段总报告。 |
| `reports/final_gap_checklist_cn.md` | 完成度、关键文件、缺口和优先级检查。 |
| `reports/final_deliverables_file_list_cn.md` | 最终可交付文件索引。 |
| `reports/midterm_check_form_text_cn.md` | 学校中期检查系统可粘贴文本。 |
| `reports/next_steps_before_final_submission_cn.md` | 最终提交前分级待办。 |
| `reports/micro_device_baseline_result_cn.md` | micro baseline 结果说明。 |
| `reports/micro_device_qat_training_result_cn.md` | micro QAT 结果说明。 |
| `reports/micro_device_error_robustness_result_cn.md` | micro robustness 结果说明。 |
| `reports/comsol_meta_atom_lut_summary_cn.md` | v1 COMSOL LUT 报告。 |
| `comsol_results/comsol_meta_atom_lut_dense_summary_cn.md` | dense LUT 报告。 |
| `comsol_results/mapped_micro_device_4level_v1/phase_radius_mapping_report_cn.md` | v1 映射报告。 |
| `comsol_results/mapped_micro_device_4level_dense/phase_radius_mapping_report_cn.md` | dense 映射报告。 |
| `comsol_results/mapped_micro_device_4level_dense/v1_vs_dense_mapping_comparison_cn.md` | v1/dense 映射对比。 |
| `D2NN-with-Pytorch-main/reports/v2_algorithm_stage_summary_cn.md` | VCSEL 算法阶段历史总结。 |
| `D2NN-with-Pytorch-main/reports/v2_error_robustness_result_cn.md` | VCSEL robustness 历史报告。 |

## 8. PPT 文件

当前项目扫描未发现 `.pptx` 或 `.ppt` 文件。`presentation_assets` 仅为图片素材目录。正式提交前需建立 PPT 文件并将其加入清单。

## 9. checkpoint / phase_map / height_map / radius_map

| 文件 | 用途 |
|---|---|
| `D2NN-with-Pytorch-main/outputs/vcsel_near_v2_train/best.pt` | VCSEL continuous baseline 最佳 checkpoint。 |
| `D2NN-with-Pytorch-main/outputs/vcsel_near_v2_train/phase_map.npy` | VCSEL baseline 连续相位图。 |
| `D2NN-with-Pytorch-main/outputs/vcsel_near_v2_train/height_map.npy` | VCSEL baseline 简化高度映射。 |
| `D2NN-with-Pytorch-main/outputs/vcsel_near_v2_train/trained_distances.npy` | VCSEL 训练后距离参数。 |
| `D2NN-with-Pytorch-main/outputs/vcsel_near_v2_quantization/{2,4,8,16}level_phase_map.npy` | VCSEL PTQ 各 level 相位图。 |
| `D2NN-with-Pytorch-main/outputs/vcsel_near_v2_qat/{2,4,8,16}level/best.pt` | VCSEL QAT 各 level checkpoint。 |
| `D2NN-with-Pytorch-main/outputs/vcsel_near_v2_qat/{2,4,8,16}level/{phase_map,quantized_phase_map,height_map}.npy` | VCSEL QAT 各 level 连续相位、量化相位与高度图。 |
| `outputs/micro_device_d2nn_train/best.pt` | micro continuous baseline 最佳 checkpoint。 |
| `outputs/micro_device_d2nn_train/phase_map.npy` | micro baseline 连续相位图，shape `(3,128,128)`。 |
| `outputs/micro_device_d2nn_train/height_map.npy` | micro baseline 简化高度图。 |
| `outputs/micro_device_d2nn_train/trained_distances.npy` | micro 训练后距离参数。 |
| `outputs/micro_device_phase_quantization/{2,4,8,16}level_phase_map.npy` | micro PTQ 各 level 相位图。 |
| `outputs/micro_device_qat/{2,4,8,16}level/best.pt` | micro QAT 各 level checkpoint。 |
| `outputs/micro_device_qat/{2,4,8,16}level/{phase_map,quantized_phase_map,height_map}.npy` | micro QAT 各 level 连续相位、量化相位与高度图。 |
| `comsol_results/mapped_micro_device_4level_v1/radius_map.npy` | v1 LUT 映射半径图。 |
| `comsol_results/mapped_micro_device_4level_dense/radius_map.npy` | dense LUT 映射半径图，shape `(3,128,128)`。 |

## 10. 尚缺的交付物

- COMSOL 原始 `.mph` 模型文件及参数说明。
- `comsol_meta_atom_field.png` 或等价单元场分布图。
- 正式中期/结题 PPT 文件。

以上缺失不影响现有算法与结构 preview 结果的可追溯性，但属于最终提交前必须或建议补齐的材料。
