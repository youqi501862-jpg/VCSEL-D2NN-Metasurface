"""Check existing D2NN artifacts and write a Chinese engineering status report."""

from __future__ import annotations

import csv
import gc
from dataclasses import dataclass
from pathlib import Path

from project_paths import (
    COMSOL_RESULTS_DIR,
    MICRO_DEVICE_QAT_DIR,
    MICRO_DEVICE_ROBUSTNESS_DIR,
    OUTPUTS_DIR,
    PRESENTATION_ASSETS_DIR,
    REPORTS_DIR,
    ROOT,
    VCSEL_QAT_DIR,
    VCSEL_ROBUSTNESS_DIR,
    ensure_dirs,
)


@dataclass(frozen=True)
class CheckResult:
    module: str
    item: str
    path: Path
    passed: bool
    detail: str


def calculate_score(checks: list[CheckResult]) -> int:
    if not checks:
        return 0
    return round(100 * sum(check.passed for check in checks) / len(checks))


def _exists(module: str, item: str, path: Path) -> CheckResult:
    passed = path.exists()
    return CheckResult(module, item, path, passed, "存在" if passed else "缺失")


def _directory_has_files(module: str, item: str, path: Path, patterns: tuple[str, ...]) -> CheckResult:
    missing = [pattern for pattern in patterns if not any(path.glob(pattern))]
    passed = path.is_dir() and not missing
    detail = "文件齐全" if passed else "缺少：" + ", ".join(missing)
    return CheckResult(module, item, path, passed, detail)


def _readable_csv(module: str, item: str, path: Path) -> CheckResult:
    if not path.is_file():
        return CheckResult(module, item, path, False, "缺失")
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            first = next(reader, None)
            if not reader.fieldnames or first is None:
                raise ValueError("没有表头或数据行")
        return CheckResult(module, item, path, True, f"CSV 可读取，列数 {len(reader.fieldnames)}")
    except Exception as exc:
        return CheckResult(module, item, path, False, f"CSV 无法读取：{exc}")


def _readable_checkpoint(module: str, item: str, path: Path) -> CheckResult:
    if not path.is_file():
        return CheckResult(module, item, path, False, "缺失")
    try:
        import torch

        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            checkpoint = torch.load(path, map_location="cpu")
        detail = f"checkpoint 可读取，类型 {type(checkpoint).__name__}"
        del checkpoint
        gc.collect()
        return CheckResult(module, item, path, True, detail)
    except Exception as exc:
        return CheckResult(module, item, path, False, f"checkpoint 无法读取：{exc}")


def collect_checks() -> list[CheckResult]:
    checks: list[CheckResult] = []

    checks.append(_readable_csv("A. VCSEL near-light", "QAT summary", VCSEL_QAT_DIR / "qat_summary.csv"))
    checks.append(_readable_csv("A. VCSEL near-light", "robustness results", VCSEL_ROBUSTNESS_DIR / "robustness_results.csv"))
    checks.append(
        _directory_has_files(
            "A. VCSEL near-light",
            "robustness curves",
            VCSEL_ROBUSTNESS_DIR / "curves",
            ("accuracy_vs_phase_noise.png", "accuracy_vs_height_noise.png", "accuracy_vs_alignment_shift.png"),
        )
    )
    for level in (2, 4, 8, 16):
        checks.append(
            _readable_checkpoint(
                "A. VCSEL near-light", f"{level}-level QAT checkpoint", VCSEL_QAT_DIR / f"{level}level" / "best.pt"
            )
        )

    dataset_dir = ROOT / "data" / "micro_devices"
    checks.append(
        _directory_has_files("B. micro device", "dataset", dataset_dir, ("train/*/*.png", "val/*/*.png"))
    )
    checks.append(
        _readable_checkpoint("B. micro device", "baseline checkpoint", OUTPUTS_DIR / "micro_device_d2nn_train" / "best.pt")
    )
    checks.append(
        _readable_csv(
            "B. micro device",
            "PTQ quantization results",
            OUTPUTS_DIR / "micro_device_phase_quantization" / "quantization_results.csv",
        )
    )
    checks.append(_readable_csv("B. micro device", "QAT summary", MICRO_DEVICE_QAT_DIR / "qat_summary.csv"))
    checks.append(
        _readable_csv("B. micro device", "robustness results", MICRO_DEVICE_ROBUSTNESS_DIR / "robustness_results.csv")
    )
    checks.append(
        _directory_has_files(
            "B. micro device",
            "inspection results",
            OUTPUTS_DIR / "micro_device_inspection",
            ("classification_metrics.csv", "confusion_matrix.csv", "confusion_matrix.png"),
        )
    )

    for name in (
        "phase_radius_lut.csv",
        "phase_radius_lut_dense.csv",
        "raw_radius_sweep_80_160nm_step5.csv",
    ):
        checks.append(_readable_csv("C. COMSOL", name, COMSOL_RESULTS_DIR / name))
    for name in ("phase_vs_radius_dense.png", "transmittance_vs_radius_dense.png"):
        checks.append(_exists("C. COMSOL", name, COMSOL_RESULTS_DIR / name))

    mapping_files = (
        ("v1 radius_map.npy", COMSOL_RESULTS_DIR / "mapped_micro_device_4level_v1" / "radius_map.npy"),
        ("dense radius_map.npy", COMSOL_RESULTS_DIR / "mapped_micro_device_4level_dense" / "radius_map.npy"),
        ("dense radius_preview.png", COMSOL_RESULTS_DIR / "mapped_micro_device_4level_dense" / "radius_preview.png"),
        (
            "dense phase_radius_mapping_report_cn.md",
            COMSOL_RESULTS_DIR / "mapped_micro_device_4level_dense" / "phase_radius_mapping_report_cn.md",
        ),
    )
    checks.extend(_exists("D. mapping", item, path) for item, path in mapping_files)

    for name in (
        "code_stage_summary_cn.md",
        "final_gap_checklist_cn.md",
        "final_deliverables_file_list_cn.md",
        "midterm_check_form_text_cn.md",
        "next_steps_before_final_submission_cn.md",
    ):
        checks.append(_exists("E. reports", name, REPORTS_DIR / name))
    return checks


def _relative(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def render_report(checks: list[CheckResult]) -> str:
    completed = [check for check in checks if check.passed]
    missing = [check for check in checks if not check.passed]
    lines = [
        "# D2NN 项目工程状态检查报告",
        "",
        "> 本检查只读取已有结果，没有重新训练模型。当前项目定位为算法仿真 + COMSOL 单元仿真 + phase-to-radius 结构映射 preview，不是真实硬件实验，也不是最终加工版设计。",
        "",
    ]
    for module in dict.fromkeys(check.module for check in checks):
        lines.extend([f"## {module}", "", "| 检查项 | 状态 | 路径 | 说明 |", "|---|---|---|---|"])
        for check in (item for item in checks if item.module == module):
            status = "通过" if check.passed else "缺失/异常"
            lines.append(f"| {check.item} | {status} | `{_relative(check.path)}` | {check.detail} |")
        lines.append("")

    lines.extend(["## 汇总", "", f"- 已完成项：{len(completed)} 项。", f"- 缺失或异常项：{len(missing)} 项。"])
    if missing:
        lines.append("- 缺失项：" + "；".join(f"{check.module} / {check.item}" for check in missing) + "。")
    else:
        lines.append("- 缺失项：本次必检清单内无缺失。")
    suggestions = []
    if not any(ROOT.rglob("*.mph")) and not any(ROOT.rglob("*.mphbin")):
        suggestions.append("保存并归档 COMSOL `.mph` 模型及仿真设置")
    if not any(ROOT.rglob("*.pptx")) and not any(ROOT.rglob("*.ppt")):
        suggestions.append("制作并检查正式中期/结题 PPT")
    if not (PRESENTATION_ASSETS_DIR / "comsol_meta_atom_field.png").is_file():
        suggestions.append("补充 COMSOL meta-atom 单元场分布图")
    suggestions.extend(("统一最终提交路径", "在报告和 PPT 中明确结构 preview 与真实硬件的边界"))
    lines.append("- 建议补齐项：" + "；".join(suggestions) + "。")
    lines.extend(
        [
            f"- 当前项目工程完整度评分：**{calculate_score(checks)}%**（必检项通过数 / 必检项总数）。",
            "",
            "该评分衡量现有代码与结果文件的工程归档完整度，不代表真实硬件或最终加工设计的完成度。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    ensure_dirs()
    checks = collect_checks()
    report_path = REPORTS_DIR / "project_status_check_cn.md"
    report_path.write_text(render_report(checks), encoding="utf-8")
    print(f"项目状态报告已生成：{report_path}")
    print(f"通过 {sum(check.passed for check in checks)} / {len(checks)} 项，工程完整度 {calculate_score(checks)}%")
    for check in checks:
        if not check.passed:
            print(f"缺失/异常：{check.module} / {check.item} - {check.detail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
