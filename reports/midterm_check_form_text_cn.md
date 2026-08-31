# 大创中期检查可粘贴文本

## 1. 已取得成果

项目已完成 VCSEL near-light 五分类与 micro device 五分类两条算法仿真主线，形成 baseline、PTQ、QAT 和误差鲁棒性评估闭环。VCSEL baseline 验证准确率为 1.000000，micro device baseline 验证准确率为 0.993000。针对低位相位量化退化问题，完成了 2/4/8/16-level QAT，其中 micro device 2-level QAT 从 PTQ 的 0.353667 提升至 0.960000，4/8/16-level QAT 分别达到 0.994333、0.995667、0.994333。项目还完成 COMSOL meta-atom 半径扫描后处理、v1/dense LUT 构建，以及 micro device 4-level 相位图到纳米柱半径图的结构映射 preview。

## 2. 项目进展情况小结

目前算法部分已具备从数据生成、D2NN baseline 训练、预测检查、训练后相位量化、量化感知训练到 phase/height/alignment 误差评估的完整流程。鲁棒性结果表明，phase noise 与 height noise 对多数非低性能 PTQ 模型影响较小，而 alignment shift 是当前最敏感误差。COMSOL dense LUT 覆盖 80–160 nm、步长 5 nm，共 17 点，`T<0.8` 点数为 0，相位覆盖比例为 0.946 of 2π。基于该 LUT 的半径图 shape 为 `(3,128,128)`，平均映射透射率为 0.946660，未使用低透射率点。

## 3. 项目存在问题及解决方案、下一步计划

当前数据主要为合成数据，误差分析为 PyTorch 算法扰动模拟；COMSOL 仅完成 meta-atom 单元扫描和 LUT 后处理，radius map 也仅是结构映射 preview，尚未完成真实器件加工、阵列级全波验证或硬件实验。下一步首先归档 COMSOL 模型、统一结果路径并完成最终报告和 PPT；技术上建议补充 VCSEL 4-level 半径映射、更多 COMSOL 参数组合、alignment-aware training 和更完整的相位覆盖优化。最终材料中将明确区分算法结果、单元仿真结果和结构预览，避免将其表述为真实硬件验证。

## 4. 300 字以内汇报摘要

本项目开展 VCSEL 近场模式与微器件图样的衍射神经网络五分类研究，已完成 baseline、PTQ、QAT 和误差鲁棒性评估。micro device baseline 验证准确率为 0.993；2-level PTQ 为 0.353667，经 QAT 提升至 0.960，说明 QAT 可改善低位相位量化性能。alignment shift 是当前最敏感误差。项目还完成 COMSOL 单元扫描、dense LUT 及 4-level 相位到半径映射，平均映射透射率为 0.946660。现有成果仅为算法仿真、COMSOL 单元仿真和结构映射预览，并非真实硬件实验或最终加工设计。
