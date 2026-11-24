"""Exercise the compact automatic workflow and its manual fallback."""

import copy
import csv
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import xml.etree.ElementTree as ET

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
from PySide6.QtCore import QPoint, QRect, Qt
from PySide6.QtGui import QFont, QPalette
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QFileDialog, QMessageBox, QPushButton

from plot_digitizer.gui import MainWindow, extract_plot_curves
from plot_digitizer.ui_appearance import configure_application_appearance


def plot_image():
    image = np.full((220, 340, 3), 255, np.uint8)
    cv2.line(image, (40, 30), (40, 190), (0, 0, 0), 2)
    cv2.line(image, (40, 190), (300, 190), (0, 0, 0), 2)
    x = np.arange(45, 295)
    points = np.column_stack((x, 110 + 25 * np.sin(x / 22))).astype(np.int32)
    cv2.polylines(image, [points], False, (30, 70, 210), 2)
    return image


def reviewed_calibration():
    return {"axes": {
        "x": {"scale": "linear", "points": [
            {"pixel": [40, 190], "value": 5}, {"pixel": [300, 190], "value": 45}]},
        "y": {"scale": "linear", "points": [
            {"pixel": [40, 190], "value": 0}, {"pixel": [40, 30], "value": 100}]},
    }}


class AutomaticWorkflowUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.app.setStyle("Fusion")
        configure_application_appearance(cls.app)

    def setUp(self):
        self.window = MainWindow(standalone=True)
        self.window.resize(1000, 700)
        self.window.show()
        QTest.qWait(30)

    def tearDown(self):
        self.window.close()
        self.window.deleteLater()
        QTest.qWait(10)

    def prepare_curves(self):
        self.window.canvas.image = plot_image()
        self.window._apply_reviewed_calibration(reviewed_calibration())
        result = extract_plot_curves(self.window.canvas.image)
        self.window._import_autopilot_curves(result, [0])

    def test_default_view_has_four_bottom_actions_and_collapsed_manual_controls(self):
        self.assertEqual([button.text() for button in self.window.workflow_buttons],
                         ["Open Image…", "Auto-calibration…", "Auto Extraction…", "Export…"])
        self.assertEqual(len(self.window.action_bar.findChildren(QPushButton)), 4)
        self.assertFalse(self.window.manual_options_button.isChecked())
        self.assertTrue(self.window.controls_scroll.isHidden())
        self.assertFalse(self.window.x_calib_table.isVisible())
        self.assertFalse(self.window.export_button.isEnabled())
        self.assertTrue(self.window.canvas.isVisible())

    def test_expanding_manual_options_preserves_calibration_and_extracted_data(self):
        self.prepare_curves()
        before = copy.deepcopy(self.window.canvas.datasets)
        coefficients = self.window.canvas._calib_ax, self.window.canvas._calib_ay
        self.window.manual_options_button.click()
        QTest.qWait(20)
        self.assertTrue(self.window.controls_scroll.isVisible())
        self.assertEqual(self.window.manual_options_button.arrowType(), Qt.DownArrow)
        self.window.manual_options_button.click()
        QTest.qWait(20)
        self.assertTrue(self.window.controls_scroll.isHidden())
        self.assertEqual(self.window.manual_options_button.arrowType(), Qt.RightArrow)
        self.assertEqual(self.window.canvas.datasets, before)
        self.assertEqual((self.window.canvas._calib_ax, self.window.canvas._calib_ay), coefficients)

    def test_extraction_opens_linked_review_and_bench_edits_reach_exports(self):
        self.prepare_curves()
        window = self.window
        self.assertIs(window.preview_tabs.currentWidget(), window.linked_review)
        self.assertEqual(window.preview_tabs.tabText(1), "Linked review")
        self.assertEqual(window.linked_review._line.get_color(),
                         window.palette().color(QPalette.Active, QPalette.Highlight).name())
        dataset = window.canvas.datasets[-1]
        pixels = copy.deepcopy(dataset["pts_px"])
        before = np.array(dataset["pts_real"])
        window.linked_review.select_point(window.canvas.active_ds, 17)
        window.manual_options_button.click()
        self.assertEqual(window.preview_tabs.currentIndex(), 0)
        self.assertEqual(window.canvas.review_point, (window.canvas.active_ds, 17))
        bench = window.calibration_bench
        bench.tables["x"].item(1, 3).setText("85")
        self.assertTrue(bench.dirty)
        self.assertFalse(window.export_button.isEnabled())
        self.assertTrue(all(not a.isEnabled() for a in window._file_export_actions))
        np.testing.assert_array_equal(dataset["pts_real"], before)
        window.manual_options_button.click()
        window.manual_options_button.click()
        self.assertEqual(bench.tables["x"].item(1, 3).text(), "85")
        bench.apply_button.click()
        self.assertFalse(bench.dirty)
        self.assertTrue(window.export_button.isEnabled())
        self.assertEqual(dataset["pts_px"], pixels)
        expected = np.column_stack((5 + (np.array(pixels)[:, 0] - 40) * 80 / 260, before[:, 1]))
        np.testing.assert_allclose(dataset["pts_real"], expected)
        self.assertEqual(window.linked_review.selected_point(), (window.canvas.active_ds, 17))
        self.assertAlmostEqual(float(window.table.item(17, 1).text()), expected[17, 0], places=4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corrected.csv"
            with patch.object(QFileDialog, "getSaveFileName", return_value=(str(path), "")), patch.object(QMessageBox, "information"):
                window._export_csv()
            with path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            np.testing.assert_allclose([[float(row["x"]), float(row["y"])] for row in rows], expected)

    def test_dragging_a_tick_changes_only_the_draft_until_applied(self):
        self.prepare_curves()
        window = self.window
        window.manual_options_button.click()
        canvas, bench = window.canvas, window.calibration_bench
        def mouse(x, y):
            return SimpleNamespace(inaxes=canvas.ax, button=1, xdata=x, ydata=y)
        canvas.draw()
        canvas._on_press(mouse(300, 190))
        canvas._on_motion(mouse(290, 188))
        canvas._on_release(mouse(290, 188))
        self.assertEqual(bench.selection(), ("x", 1))
        self.assertEqual(bench.reviewed_axes()["x"]["points"][1]["pixel"], [290, 188])
        self.assertEqual(canvas.x_calib_pts[1], [300, 190])
        self.assertFalse(window.export_button.isEnabled())
        full = canvas.ax.get_xlim()
        window.tick_zoom.click()
        zoomed = canvas.ax.get_xlim()
        self.assertLess(zoomed[1] - zoomed[0], (full[1] - full[0]) / 2)
        self.assertLessEqual(zoomed[0], 290)
        self.assertGreaterEqual(zoomed[1], 290)
        window.tick_zoom.click()
        np.testing.assert_allclose(canvas.ax.get_xlim(), full)
        bench.revert_button.click()
        self.assertTrue(window.export_button.isEnabled())
        bench.pick_button.click()
        canvas._on_press(mouse(292, 189))
        bench.apply_button.click()
        self.assertEqual(canvas.x_calib_pts[1], [292, 189])

    def test_scheme_menu_is_local_and_preserves_source_and_selection(self):
        self.prepare_curves()
        window = self.window
        window.linked_review.select_point(window.canvas.active_ds, 9)
        original = window.canvas.image.copy()
        app_base = self.app.palette().color(QPalette.Base)
        with patch.object(window._scheme_settings, "setValue") as save:
            for key, base in (("graphite", "#18242d"), ("paper", "#ffffff")):
                window.scheme_actions[key].trigger()
                self.assertTrue(window.scheme_actions[key].isChecked())
                self.assertEqual(window.palette().color(QPalette.Base).name(), base)
                save.assert_called_with("appearance/scheme", key)
                self.assertEqual(self.app.palette().color(QPalette.Base), app_base)
                self.assertEqual(window.linked_review.selected_point(), (window.canvas.active_ds, 9))
                np.testing.assert_array_equal(window.canvas.image, original)

    def test_bottom_actions_apply_reviewed_calibration_and_import_detected_curves(self):
        self.window.canvas.image = plot_image()
        calibration = Mock(reviewed=reviewed_calibration())
        calibration.exec.return_value = QDialog.Accepted
        with patch("plot_digitizer.gui.AxisCalibrationDialog", return_value=calibration):
            self.window.auto_calibration_button.click()
        self.assertTrue(self.window.canvas._x_calib_ready)
        self.assertTrue(self.window.canvas._y_calib_ready)
        result = extract_plot_curves(self.window.canvas.image)
        picker = Mock(result_data=result, selected_indices=[0])
        picker.exec.return_value = QDialog.Accepted
        with patch("plot_digitizer.gui.CurvePickerDialog", return_value=picker):
            self.window.autopilot_button.click()
        self.assertGreater(len(self.window.canvas.datasets[-1]["pts_real"]), 200)
        self.assertTrue(self.window.export_button.isEnabled())
        self.assertIn("ready", self.window.workflow_hint.text())
        self.assertTrue(self.window.controls_scroll.isHidden())

    def test_export_menu_writes_all_three_formats(self):
        self.prepare_curves()
        count = sum(len(dataset["pts_real"]) for dataset in self.window.canvas.datasets)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            for action, extension in zip(self.window.export_menu.actions(), ("csv", "json", "xrdml")):
                output = folder / ("trace." + extension)
                with patch.object(QFileDialog, "getSaveFileName", return_value=(str(output), "")), \
                        patch.object(QMessageBox, "information"), patch.object(QMessageBox, "critical") as failure:
                    action.trigger()
                failure.assert_not_called()
                self.assertGreater(output.stat().st_size, 0)
            with (folder / "trace.csv").open(newline="") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), count)
            data = json.loads((folder / "trace.json").read_text())
            self.assertEqual(sum(len(points) for points in data.values()), count)
            self.assertTrue(ET.parse(folder / "trace.xrdml").getroot().tag.endswith("xrdMeasurements"))

    def test_opening_another_image_resets_calibration_export_and_returns_to_image_tab(self):
        self.prepare_curves()
        self.window.preview_tabs.setCurrentIndex(1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plot.png"
            self.assertTrue(cv2.imwrite(str(path), plot_image()))
            with patch.object(QFileDialog, "getOpenFileName", return_value=(str(path), "")):
                self.window.open_image_button.click()
        self.assertEqual(self.window.preview_tabs.currentIndex(), 0)
        self.assertFalse(self.window.canvas._x_calib_ready)
        self.assertFalse(self.window.canvas._y_calib_ready)
        self.assertFalse(self.window.export_button.isEnabled())
        self.assertIsNotNone(self.window.canvas.roi)
        self.assertTrue(self.window.controls_scroll.isHidden())

    def test_manual_image_actions_return_from_the_data_tab(self):
        self.prepare_curves()
        for action in (self.window._start_roi, self.window._start_x_calibration,
                       self.window._start_y_calibration, self.window._start_digitize,
                       self.window._start_reject_box, self.window._start_eyedrop):
            self.window.preview_tabs.setCurrentIndex(1)
            action()
            self.assertEqual(self.window.preview_tabs.currentIndex(), 0)

    def test_clearing_all_points_disables_export(self):
        self.prepare_curves()
        self.window._clear_dataset()
        self.assertFalse(self.window.export_button.isEnabled())

    def test_language_switch_keeps_manual_options_collapsed(self):
        self.window.set_language("zh")
        self.assertEqual(self.window.manual_options_button.text(), "手动选项")
        self.assertEqual(self.window.auto_calibration_button.text(), "自动校准…")
        self.assertEqual(self.window.autopilot_button.text(), "自动提取…")
        self.assertEqual(self.window.export_menu.actions()[0].text(), "导出 CSV…")
        self.assertEqual(self.window.preview_tabs.tabText(1), "数字化数据")
        self.assertTrue(self.window.controls_scroll.isHidden())
        self.window.set_language("en")
        self.assertEqual(self.window.manual_options_button.text(), "Manual options")

    def test_actions_remain_visible_at_small_sizes_and_large_fonts(self):
        font = QFont(self.window.font())
        font.setPointSize(14)
        self.window.setFont(font)
        for language in ("en", "zh"):
            self.window.set_language(language)
            for width in (640, 920):
                for expanded in (False, True):
                    with self.subTest(language=language, width=width, expanded=expanded):
                        self.window.resize(width, 500)
                        self.window.manual_options_button.setChecked(expanded)
                        QTest.qWait(40)
                        # Qt propagates stylesheet font changes and the resulting
                        # layout requests asynchronously. Wait for the footer to
                        # settle while retaining the strict clipping assertions.
                        for _ in range(100):
                            if all(button.width() >= button.minimumSizeHint().width()
                                   and button.height() >= button.minimumSizeHint().height()
                                   and self.window.action_bar.rect().contains(button.geometry())
                                   for button in self.window.workflow_buttons):
                                break
                            QTest.qWait(10)
                        for button in self.window.workflow_buttons:
                            position = button.mapTo(self.window, QPoint(0, 0))
                            self.assertTrue(button.isVisible())
                            self.assertTrue(self.window.rect().contains(QRect(position, button.size())))
                            self.assertGreaterEqual(button.width(), button.minimumSizeHint().width())
                            self.assertGreaterEqual(button.height(), button.minimumSizeHint().height())
                            self.assertTrue(self.window.action_bar.rect().contains(button.geometry()))
                            central=self.window.centralWidget()
                            position=button.mapTo(central, QPoint(0, 0))
                            self.assertTrue(central.rect().contains(QRect(position, button.size())))
                        self.assertGreaterEqual(self.window.manual_options_button.width(),
                                                self.window.manual_options_button.minimumSizeHint().width())
                        manual=self.window.manual_options_button
                        metrics=manual.fontMetrics()
                        self.assertGreaterEqual(manual.width(), metrics.horizontalAdvance(manual.text())
                                                + manual.iconSize().width() + 2 * metrics.horizontalAdvance(" "))
                        if expanded:
                            self.assertEqual(self.window.controls_scroll.horizontalScrollBar().maximum(), 0)


if __name__ == "__main__":
    unittest.main()
