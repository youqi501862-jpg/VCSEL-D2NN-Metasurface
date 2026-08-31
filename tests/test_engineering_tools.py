from __future__ import annotations

import csv
import importlib
import math
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np


class ProjectPathsTests(unittest.TestCase):
    def test_environment_override_uses_pathlib_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            old_value = os.environ.get("D2NN_PROJECT_ROOT")
            os.environ["D2NN_PROJECT_ROOT"] = temp_dir
            project_paths = None
            try:
                import project_paths

                module = importlib.reload(project_paths)
                self.assertIsInstance(module.ROOT, Path)
                self.assertEqual(module.ROOT, Path(temp_dir).resolve())
                self.assertEqual(module.REPORTS_DIR, module.ROOT / "reports")
                self.assertEqual(module.PRESENTATION_ASSETS_DIR, module.ROOT / "assets")
                self.assertEqual(module.VCSEL_OUTPUTS_DIR, module.ROOT / "outputs")
            finally:
                if old_value is None:
                    os.environ.pop("D2NN_PROJECT_ROOT", None)
                else:
                    os.environ["D2NN_PROJECT_ROOT"] = old_value
                if project_paths is not None:
                    importlib.reload(project_paths)


class MappingTests(unittest.TestCase):
    def write_lut(self, directory: Path, t_column: str) -> Path:
        path = directory / f"lut_{t_column}.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["radius_nm", t_column, "phase_rad", "phase_wrapped_rad"],
            )
            writer.writeheader()
            writer.writerow(
                {"radius_nm": 100, t_column: 0.9, "phase_rad": 0.1, "phase_wrapped_rad": 0.1}
            )
            writer.writerow(
                {"radius_nm": 150, t_column: 0.95, "phase_rad": 6.2, "phase_wrapped_rad": 6.2}
            )
        return path

    def test_lut_accepts_T_and_transmittance_columns(self) -> None:
        from map_phase_to_radius import read_lut

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for column in ("T", "transmittance"):
                lut = read_lut(self.write_lut(root, column))
                self.assertEqual(lut.transmittance_column, column)
                np.testing.assert_allclose(lut.transmittance, [0.9, 0.95])

    def test_mapping_uses_circular_phase_distance(self) -> None:
        from map_phase_to_radius import LutData, map_phase_array, save_preview

        lut = LutData(
            columns=("radius_nm", "T", "phase_rad", "phase_wrapped_rad"),
            transmittance_column="T",
            radius_nm=np.array([100.0, 150.0]),
            transmittance=np.array([0.9, 0.95]),
            phase_rad=np.array([0.1, 6.2]),
            phase_wrapped_rad=np.array([0.1, 6.2]),
        )
        result = map_phase_array(np.array([[0.02, 6.25, math.pi]]), lut)
        np.testing.assert_array_equal(result.radius_map_nm[0, :2], [100.0, 150.0])
        self.assertLess(result.phase_error_rad[0, 0], 0.1)
        with tempfile.TemporaryDirectory() as temp_dir:
            preview = Path(temp_dir) / "preview.png"
            save_preview(result, preview, "test", 0.8)
            self.assertGreater(preview.stat().st_size, 0)


class StatusTests(unittest.TestCase):
    def test_score_is_percentage_of_passed_required_checks(self) -> None:
        from check_project_status import CheckResult, calculate_score

        checks = [
            CheckResult("A", "one", Path("a"), True, "ok"),
            CheckResult("A", "two", Path("b"), False, "missing"),
            CheckResult("B", "three", Path("c"), True, "ok"),
        ]
        self.assertEqual(calculate_score(checks), 67)


class ReproduceSummaryTests(unittest.TestCase):
    def test_collects_verified_values_from_existing_artifacts(self) -> None:
        from run_reproduce_summary import collect_summary

        root = Path(__file__).resolve().parents[1]
        if not (root / "outputs").is_dir():
            self.skipTest("Release repository excludes outputs; real-artifact test is skipped")
        summary = collect_summary(root)
        self.assertAlmostEqual(summary.micro_baseline_val_acc, 0.993, places=6)
        self.assertAlmostEqual(summary.micro_ptq[2], 0.353667, places=6)
        self.assertAlmostEqual(summary.micro_qat[4], 0.994333, places=6)
        self.assertEqual(summary.dense_lut_points, 17)
        self.assertEqual(summary.dense_mapping_shape, (3, 128, 128))
        np.testing.assert_array_equal(summary.dense_mapping_radii, [95.0, 125.0, 140.0, 155.0])
        self.assertAlmostEqual(summary.mean_mapped_transmittance, 0.946660, places=6)


class ResultIndexTests(unittest.TestCase):
    def test_classification_marks_images_as_ppt_candidates(self) -> None:
        from build_result_index import classify_artifact

        artifact = classify_artifact(Path("presentation_assets") / "phase_vs_radius_dense.png")
        self.assertEqual(artifact.file_type, "PNG 图片")
        self.assertTrue(artifact.suitable_for_submission)
        self.assertTrue(artifact.suitable_for_ppt)


class GithubReleaseLayoutTests(unittest.TestCase):
    def test_public_text_files_do_not_contain_private_absolute_project_path(self) -> None:
        root = Path(__file__).resolve().parents[1]
        offenders = []
        for folder in (root / "scripts", root / "reports", root / "comsol_results"):
            for path in folder.rglob("*"):
                if path.suffix.lower() not in {".py", ".md", ".txt", ".csv"}:
                    continue
                text = path.read_text(encoding="utf-8-sig", errors="ignore")
                private_markers = ("D:" + "\\大创D2NN", "C:" + "\\Users\\Administrator")
                if any(marker in text for marker in private_markers):
                    offenders.append(path.relative_to(root).as_posix())
        self.assertEqual(offenders, [])

    def test_scripts_moved_to_subdirectory_resolve_repository_root(self) -> None:
        root = Path(__file__).resolve().parents[1]
        for name in (
            "process_comsol_dense_lut.py",
            "map_phase_to_radius_v1.py",
            "map_phase_to_radius_dense.py",
        ):
            text = (root / "scripts" / name).read_text(encoding="utf-8")
            self.assertIn("Path(__file__).resolve().parents[1]", text)


if __name__ == "__main__":
    unittest.main()
