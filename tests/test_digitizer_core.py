"""Boundary and packaging checks for headless scientific-plot extraction."""

import os
from importlib.util import find_spec
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from plot_digitizer.axis_calibration import calibration_coefficients, suggest_axis_calibration
from plot_digitizer.digitizer_core import detect_plot_roi, extract_plot_curves


ROOT = Path(__file__).resolve().parents[1]


class DigitizerCoreBoundaryTests(unittest.TestCase):
    def test_core_calibration_and_benchmark_import_without_gui_libraries(self):
        code = """
import importlib.abc
import sys
class NoGui(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('PySide6', 'PyQt5', 'matplotlib'):
            raise AssertionError('GUI dependency imported: ' + fullname)
sys.meta_path.insert(0, NoGui())
from plot_digitizer.digitizer_core import extract_plot_curves
from plot_digitizer.axis_calibration import suggest_axis_calibration
from scripts.benchmark_plot_digitizer import measure_case
import numpy as np
image = np.full((60, 100, 3), 255, np.uint8)
image[30:32, 10:90] = 0
assert len(extract_plot_curves(image, roi=(0, 0, 100, 60))['candidates']) == 1
"""
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                                capture_output=True, text=True, timeout=45)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_thin_large_images_fail_clearly_instead_of_crashing_opencv(self):
        for shape in ((16, 30000, 3), (30000, 16, 3)):
            image = np.full(shape, 255, np.uint8)
            self.assertIsNone(detect_plot_roi(image))
            with self.assertRaisesRegex(ValueError, "too narrow"):
                extract_plot_curves(image)

    def test_invalid_roi_data_has_a_consistent_validation_error(self):
        image = np.full((60, 100, 3), 255, np.uint8)
        for roi in ([0, 0, np.nan, 50], [0, 0, "bad", 50], [[0, 0], [50, 50]], [0, 0, 50]):
            with self.subTest(roi=roi):
                with self.assertRaises(ValueError):
                    extract_plot_curves(image, roi=roi)
                with self.assertRaises(ValueError):
                    suggest_axis_calibration(image, roi)

    def test_pre_cancelled_extraction_never_starts_roi_detection(self):
        image = np.full((60, 100, 3), 255, np.uint8)
        with patch("plot_digitizer.digitizer_core.detect_plot_roi") as detector:
            with self.assertRaises(InterruptedError):
                extract_plot_curves(image, cancelled=lambda: True)
        detector.assert_not_called()

    def test_conflicting_labels_at_the_same_tick_cannot_be_calibrated(self):
        for axis in ("x", "y"):
            points = [{"pixel": [0, 0], "value": 1}, {"pixel": [0, 0], "value": 2},
                      {"pixel": [10, 10], "value": 3}]
            with self.assertRaisesRegex(ValueError, "distinct tick"):
                calibration_coefficients(points, axis)

    @unittest.skipUnless(find_spec("PySide6"), "Desktop smoke needs PySide6")
    def test_desktop_entry_point_imports_from_an_independent_working_directory(self):
        code = """
import os, sys
sys.path[:0] = [sys.argv[1], *sys.argv[2:]]
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
from PySide6.QtWidgets import QApplication
from plot_digitizer.gui import MainWindow
import plot_digitizer_auto
app = QApplication([])
window = MainWindow(standalone=True)
assert window.auto_calibration_button.text()
window.close()
"""
        dependencies = [path for path in sys.path if path and (Path(path) / "numpy").is_dir()]
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, "-I", "-c", code, str(ROOT), *dependencies],
                                    cwd=directory, env=dict(os.environ, QT_QPA_PLATFORM="offscreen"),
                                    capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
