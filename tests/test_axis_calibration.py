import copy
import csv
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
from PySide6.QtCore import QRect, Qt, QTimer
from PySide6.QtGui import QFont
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog

from plot_digitizer.axis_calibration import (
    _axis_geometry, _consistent_axis_pairs, calibration_coefficients,
    parse_axis_number, suggest_axis_calibration,
)
from plot_digitizer.gui import AxisCalibrationDialog, MainWindow


def proposal():
    def axis(points):
        return {"points": [{"pixel": p, "value": v, "text": str(v)} for p, v in points],
                "scale": "linear", "rejected": [], "fit_error_px": 0}
    return {"axes": {"x": axis([([40, 220], 0), ([200, 220], 10), ([360, 220], 20)]),
                     "y": axis([([40, 220], 0), ([40, 120], 50), ([40, 20], 100)])},
            "roi": [42, 22, 360, 218], "engine": "Test OCR", "ready": True, "review_required": True}


class AxisInferenceTests(unittest.TestCase):
    def test_numeric_parser_preserves_sign_and_decimals_and_rejects_non_numeric_text(self):
        for text, value in [("−2.5", -2.5), ("1,000", 1000), ("1,5", 1.5), ("1e-3", .001), ("0", 0)]:
            self.assertEqual(parse_axis_number(text), value)
        for text in ("200O", "Fig. 20", "NaN", "inf", "10 cm", "1e999"):
            self.assertIsNone(parse_axis_number(text))

    def test_axis_pairs_reject_an_ocr_outlier_and_keep_decreasing_wavenumbers(self):
        pairs = [{"pixel": [x, 220], "value": value, "label_distance": 0, "tick_observed": True}
                 for x, value in [(40, 4000), (140, 3000), (240, 2000), (340, 1000), (190, 8900)]]
        result = _consistent_axis_pairs(pairs, "x")
        self.assertEqual(result["scale"], "linear")
        self.assertEqual(len(result["points"]), 4)
        self.assertEqual(result["rejected"][0]["value"], 8900)

    def test_log_axis_requires_multiple_supporting_labels(self):
        pairs = [{"pixel": [40, y], "value": value, "label_distance": 0}
                 for y, value in [(20, 1000), (120, 100), (220, 10), (320, 1)]]
        result = _consistent_axis_pairs(pairs, "y")
        self.assertEqual(result["scale"], "log")
        self.assertEqual(len(result["points"]), 4)
        self.assertEqual(_consistent_axis_pairs(pairs[:2], "y")["scale"], "linear")

    def test_small_magnitude_labels_are_independent_of_their_physical_units(self):
        pairs = [{"pixel": [x, 0], "value": value, "label_distance": 0}
                 for x, value in ((0, 1e-20), (100, 2e-20), (200, 3e-20))]
        result = _consistent_axis_pairs(pairs, "x")
        self.assertEqual(result["scale"], "linear")
        self.assertEqual(len(result["points"]), 3)
        coefficients = calibration_coefficients(pairs, "x")
        np.testing.assert_allclose(np.polyval(coefficients, [0, 100, 200]), [1e-20, 2e-20, 3e-20], rtol=1e-12, atol=0)

    def test_nonuniform_axis_spacing_is_flagged_for_review(self):
        pairs = [{"pixel": [x, 220], "value": value, "label_distance": 0}
                 for x, value in [(0, 0), (100, 1), (200, 2), (300, 3), (400, 5), (500, 7), (600, 9)]]
        result = _consistent_axis_pairs(pairs, "x")
        self.assertTrue(result["scale_conflict"])
        self.assertGreaterEqual(len(result["rejected"]), 2)

    def test_observed_ticks_are_located_on_both_axes(self):
        gray = np.full((280, 440), 255, np.uint8)
        cv2.line(gray, (40, 20), (40, 230), 0, 1)
        cv2.line(gray, (40, 230), (400, 230), 0, 1)
        for x in (100, 200, 300):
            cv2.line(gray, (x, 224), (x, 237), 0, 1)
        for y in (70, 120, 170):
            cv2.line(gray, (33, y), (46, y), 0, 1)
        for axis, expected, coordinate in (("x", [100, 200, 300], 0), ("y", [70, 120, 170], 1)):
            ticks = _axis_geometry(gray, (42, 22, 399, 228), axis)["ticks"]
            for value in expected:
                self.assertLess(min(abs(tick["point"][coordinate] - value) for tick in ticks), 1.1)

    def test_blank_image_does_not_invent_labels_or_call_ocr(self):
        image = np.full((200, 300, 3), 255, np.uint8)
        with patch("plot_digitizer.axis_calibration.read_local_ocr") as ocr:
            result = suggest_axis_calibration(image, (10, 10, 280, 180))
        ocr.assert_not_called()
        self.assertFalse(result["ready"])
        self.assertTrue(all(not result["axes"][axis]["points"] for axis in ("x", "y")))

    def test_invalid_fits_and_cancellation_are_rejected(self):
        for points in ([], [{"pixel": [5, 0], "value": 1}] * 2,
                       [{"pixel": [x, 0], "value": v} for x, v in [(0, 1), (10, 3), (20, 2)]]):
            with self.assertRaises(ValueError):
                calibration_coefficients(points, "x")
        with self.assertRaises(ValueError):
            calibration_coefficients([{"pixel": [0, 0], "value": 0}, {"pixel": [10, 0], "value": 10}], "x", "log")
        with self.assertRaises(InterruptedError):
            suggest_axis_calibration(np.full((100, 100, 3), 255, np.uint8), (5, 5, 95, 95), cancelled=lambda: True)


class AxisCalibrationReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.window = MainWindow(standalone=True)
        self.window.canvas.image = np.full((260, 420, 3), 255, np.uint8)
        self.window.canvas.roi = (42, 22, 360, 218)
        self.window.canvas.fit_x_calib([[0, 0], [400, 0]], [0, 400])
        self.window.canvas.fit_y_calib([[0, 0], [0, 240]], [240, 0])
        self.window.canvas.bulk_add_pixels([(200, 120)])
        self.dialog = None

    def tearDown(self):
        if self.dialog:
            self.dialog.reject()
            self.wait_until(lambda: not self.dialog.worker.isRunning())
            self.dialog.close(); self.dialog.deleteLater()
        self.window.close(); self.window.deleteLater(); QTest.qWait(5)

    def wait_until(self, predicate):
        deadline = time.monotonic() + 5
        while not predicate() and time.monotonic() < deadline:
            QTest.qWait(5)
        self.assertTrue(predicate())

    def create_dialog(self, result=None):
        with patch("plot_digitizer.gui.suggest_axis_calibration", return_value=result or proposal()):
            self.dialog = AxisCalibrationDialog(self.window)
            self.dialog.show()
            self.wait_until(lambda: self.dialog.progress.isHidden())
        return self.dialog

    def test_review_edits_tick_positions_and_values_before_atomic_application(self):
        before = copy.deepcopy(self.window.canvas.datasets)
        dialog = self.create_dialog()
        self.assertEqual(self.window.canvas.datasets, before)
        table = dialog.tables["x"]; table.selectRow(1)
        dialog._set_tick_on_image(SimpleNamespace(inaxes=dialog.axes, button=1, xdata=200, ydata=220))
        dialog._accept_review()
        self.assertEqual(dialog.result(), QDialog.Accepted)
        self.window._apply_reviewed_calibration(dialog.reviewed)
        self.assertEqual(self.window.canvas.x_calib_vals, [0, 10, 20])
        np.testing.assert_allclose(self.window.canvas._px2data(200, 120), [10, 50], atol=1e-9)
        self.assertEqual(self.window.canvas.datasets[0]["pts_px"], before[0]["pts_px"])
        np.testing.assert_allclose(self.window.canvas.datasets[0]["pts_real"], [[10, 50]])

    def test_button_runs_the_review_and_applies_only_after_acceptance(self):
        def create(parent):
            dialog = AxisCalibrationDialog(parent)
            dialog.worker.finished.connect(lambda: QTimer.singleShot(0, dialog._accept_review))
            return dialog
        with patch("plot_digitizer.gui.suggest_axis_calibration", return_value=proposal()), \
                patch("plot_digitizer.gui.AxisCalibrationDialog", side_effect=create):
            self.window.auto_calibration_button.click()
        np.testing.assert_allclose(self.window.canvas._px2data(200, 120), [10, 50], atol=1e-9)

    def test_cancel_preserves_fits_points_and_datasets(self):
        before = (copy.deepcopy(self.window.canvas.datasets), self.window.canvas._calib_ax,
                  self.window.canvas._calib_ay, copy.deepcopy(self.window.canvas.x_calib_pts))
        dialog = self.create_dialog(); dialog.reject()
        self.assertEqual(dialog.result(), QDialog.Rejected)
        self.assertEqual(before, (self.window.canvas.datasets, self.window.canvas._calib_ax,
                                 self.window.canvas._calib_ay, self.window.canvas.x_calib_pts))

    def test_incomplete_or_invalid_review_cannot_apply(self):
        dialog = self.create_dialog()
        for row in range(dialog.tables["y"].rowCount()):
            dialog.tables["y"].item(row, 0).setCheckState(Qt.Unchecked)
        dialog._accept_review()
        self.assertIsNone(dialog.reviewed)
        self.assertTrue(dialog.error.text())
        reviewed = proposal(); reviewed["axes"]["y"]["points"][1]["value"] = float("nan")
        before = copy.deepcopy(self.window.canvas.datasets)
        with self.assertRaises(ValueError):
            self.window._apply_reviewed_calibration(reviewed)
        self.assertEqual(self.window.canvas.datasets, before)
        self.assertAlmostEqual(self.window.canvas._calib_ax, 1)

    def test_semi_log_x_is_applied_without_changing_y_to_log(self):
        reviewed = proposal(); reviewed["axes"]["x"]["scale"] = "log"
        for point, value in zip(reviewed["axes"]["x"]["points"], [1, 10, 100]):
            point["value"] = value
        self.window._apply_reviewed_calibration(reviewed)
        self.assertEqual(self.window.canvas.scale_type, "semilogx")
        np.testing.assert_allclose(self.window.canvas._px2data(200, 120), [10, 50])

    def test_small_log_values_survive_new_points_and_csv_json_exports(self):
        reviewed = proposal()
        reviewed["axes"]["y"]["scale"] = "log"
        for point, value in zip(reviewed["axes"]["y"]["points"], [1e-9, 1e-8, 1e-7]):
            point["value"] = value
        self.window._apply_reviewed_calibration(reviewed)
        canvas = self.window.canvas
        existing = canvas.datasets[0]["pts_real"][0][1]
        canvas.bulk_add_pixels([(200, 120), (200, 220), (200, 20)])
        values = [point[1] for point in canvas.datasets[0]["pts_real"]]
        np.testing.assert_allclose(values, [1e-8, 1e-8, 1e-9, 1e-7], rtol=1e-12, atol=0)
        np.testing.assert_allclose(values[1], existing, rtol=1e-12, atol=0)
        with tempfile.TemporaryDirectory() as directory:
            for suffix, export in (("csv", self.window._export_csv), ("json", self.window._export_json)):
                path = Path(directory) / ("small-values." + suffix)
                with patch("plot_digitizer.gui.QFileDialog.getSaveFileName", return_value=(str(path), "")), \
                        patch("plot_digitizer.gui.QMessageBox.information"):
                    export()
                if suffix == "csv":
                    with path.open(newline="") as stream:
                        exported = [float(row["y"]) for row in csv.DictReader(stream)]
                else:
                    grouped = json.loads(path.read_text())
                    exported = [point["y"] for points in grouped.values() for point in points]
                np.testing.assert_allclose(exported, values, rtol=1e-12, atol=0)

    def test_error_keeps_apply_disabled(self):
        with patch("plot_digitizer.gui.suggest_axis_calibration", side_effect=RuntimeError("OCR unavailable")):
            self.dialog = AxisCalibrationDialog(self.window)
            self.wait_until(lambda: not self.dialog.worker.isRunning())
            QTest.qWait(20)
        self.assertFalse(self.dialog.apply_button.isEnabled())
        self.assertIn("unavailable", self.dialog.error.text())

    def test_manual_calibration_uses_the_same_validation_and_preserves_the_previous_fit(self):
        canvas = self.window.canvas
        before = canvas._calib_ax, canvas._calib_bx, canvas._calib_ay, canvas._calib_by
        with self.assertRaises(ValueError):
            canvas.fit_x_calib([[20, 0], [20, 0], [40, 0]], [0, 1, 2])
        with self.assertRaises(ValueError):
            canvas.fit_y_calib([[0, 0], [0, 120]], [0, float("inf")])
        self.assertEqual(before, (canvas._calib_ax, canvas._calib_bx, canvas._calib_ay, canvas._calib_by))

    def test_invalid_manual_tick_can_be_retried_without_an_uncaught_ui_error(self):
        canvas = self.window.canvas
        for axis in ("x", "y"):
            canvas.mode = "calibrate_" + axis
            setattr(canvas, "_" + axis + "_target", 2)
            setattr(canvas, axis + "_calib_pts", [[20, 20], [20, 20]])
            setattr(canvas, axis + "_calib_vals", [0, 10])
            with patch("plot_digitizer.gui.QMessageBox.warning") as warning:
                self.window._on_calib_click(20, 20)
            warning.assert_called_once()
            self.assertEqual(len(getattr(canvas, axis + "_calib_pts")), 1)
            self.assertEqual(canvas.mode, "calibrate_" + axis)

    def test_review_fits_small_screen_and_large_font_in_both_languages(self):
        font = QFont(self.window.font()); font.setPointSize(14); self.window.setFont(font)
        bounds = QRect(0, 0, 960, 540)
        for language in ("en", "zh"):
            self.window.set_language(language)
            with patch("PySide6.QtGui.QScreen.availableGeometry", return_value=bounds):
                dialog = self.create_dialog()
                QTest.qWait(30)
                self.assertTrue(bounds.contains(dialog.frameGeometry()))
                self.assertEqual(dialog.body_scroll.horizontalScrollBar().maximum(), 0)
                self.assertTrue(dialog.buttons.isVisible())
                self.assertLess(dialog.summary.geometry().bottom(), dialog.preview.geometry().top())
                if language == "zh":
                    self.assertEqual(dialog.apply_button.text(), "应用已检查的校准")
            dialog.close(); dialog.deleteLater(); self.dialog = None; QTest.qWait(5)


if __name__ == "__main__":
    unittest.main()
