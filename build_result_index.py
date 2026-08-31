"""Build a deterministic Markdown index of existing D2NN deliverables."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from project_paths import (
    COMSOL_RESULTS_DIR,
    OUTPUTS_DIR,
    PRESENTATION_ASSETS_DIR,
    REPORTS_DIR,
    ROOT,
    ensure_dirs,
)


@dataclass(frozen=True)
class ArtifactInfo:
    file_type: str
    purpose: str
    suitable_for_submission: bool
    suitable_for_ppt: bool


TYPE_LABELS = {
    ".csv": "CSV 数据",
    ".npy": "NumPy 数据",
    ".npz": "NumPy 压缩数据",
    ".pt": "PyTorch checkpoint",
    ".pth": "PyTorch checkpoint",
    ".png": "PNG 图片",
    ".jpg": "JPEG 图片",
    ".jpeg": "JPEG 图片",
    ".md": "Markdown 报告",
    ".txt": "文本文件",
}


def classify_artifact(path: Path) -> ArtifactInfo:
    suffix = path.suffix.lower()
    file_type = TYPE_LABELS.get(suffix, suffix.lstrip(".").upper() + " 文件" if suffix else "无扩展名文件")
    lower = path.as_posix().lower()
    if suffix == ".csv":
        purpose = "保存训练、评估、鲁棒性或 COMSOL 扫描的结构化结果"
    elif suffix in {".pt", ".pth"}:
        purpose = "保存已训练模型 checkpoint，供只读评估或归档"
    elif suffix in {".npy", ".npz"}:
        purpose = "保存 phase map、height map、radius map 或中间数值数组"
    elif suffix in {".png", ".jpg", ".jpeg"}:
        purpose = "展示数据、曲线、混淆矩阵、场图或结构映射预览"
    elif suffix == ".md":
        purpose = "记录实验结果、工程检查、方法说明或阶段总结"
    else:
        purpose = "项目辅助结果文件"

    excluded = any(token in lower for token in ("smoke", "/backups/", "/samples/", "/correct_samples/", "/wrong_samples/"))
    suitable_for_submission = suffix in TYPE_LABELS and not excluded
    ppt_keywords = (
        "presentation_assets",
        "curve",
        "confusion_matrix.png",
        "radius_preview.png",
        "phase_vs_radius",
        "transmittance_vs_radius",
        "preview_grid.png",
        "training_curves.png",
    )
    suitable_for_ppt = suffix in {".png", ".jpg", ".jpeg"} and not excluded and any(
        keyword in lower for keyword in ppt_keywords
    )
    return ArtifactInfo(file_type, purpose, suitable_for_submission, suitable_for_ppt)


def module_name(path: Path) -> str:
    relative = path.relative_to(ROOT)
    if relative.parts[0] == "outputs":
        return "outputs / " + (relative.parts[1] if len(relative.parts) > 1 else "其他")
    if relative.parts[0] == "comsol_results":
        if len(relative.parts) > 2 and relative.parts[1].startswith("mapped_"):
            return "mapping / " + relative.parts[1]
        return "COMSOL results"
    if relative.parts[0] == "reports":
        return "reports"
    if relative.parts[0] == "presentation_assets":
        return "presentation assets"
    return relative.parts[0]


def scan_artifacts() -> list[Path]:
    roots = (OUTPUTS_DIR, COMSOL_RESULTS_DIR, REPORTS_DIR, PRESENTATION_ASSETS_DIR)
    paths: set[Path] = set()
    for directory in roots:
        if directory.is_dir():
            paths.update(path for path in directory.rglob("*") if path.is_file())
    paths.add(REPORTS_DIR / "result_index_cn.md")
    return sorted(paths, key=lambda path: path.as_posix().lower())


def _yes_no(value: bool) -> str:
    return "是" if value else "否"


def render_index(paths: list[Path]) -> str:
    lines = [
        "# D2NN 项目结果索引总表",
        "",
        "> 本索引由脚本扫描生成，只描述已有文件。项目当前属于算法仿真 + COMSOL 单元仿真 + phase-to-radius 结构映射 preview，不是真实硬件实验，也不是最终加工版设计。",
        "",
        f"共索引 `{len(paths)}` 个文件。`是否适合提交` 为工程整理建议，不代表文件已通过最终内容审查。",
        "",
    ]
    modules = sorted({module_name(path) for path in paths})
    for module in modules:
        lines.extend(
            [
                f"## {module}",
                "",
                "| 文件名 | 路径 | 文件类型 | 用途 | 适合提交 | 适合 PPT |",
                "|---|---|---|---|---|---|",
            ]
        )
        for path in (candidate for candidate in paths if module_name(candidate) == module):
            info = classify_artifact(path)
            relative = path.relative_to(ROOT).as_posix()
            lines.append(
                f"| {path.name} | `{relative}` | {info.file_type} | {info.purpose} | "
                f"{_yes_no(info.suitable_for_submission)} | {_yes_no(info.suitable_for_ppt)} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    ensure_dirs()
    paths = scan_artifacts()
    output_path = REPORTS_DIR / "result_index_cn.md"
    output_path.write_text(render_index(paths), encoding="utf-8")
    print(f"结果索引已生成：{output_path}")
    print(f"共索引 {len(paths)} 个文件。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
