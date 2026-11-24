import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import QEventLoop, Qt, QTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QLineEdit, QWidget

from plot_digitizer.digitizer_calibration_bench import CalibrationBench
from plot_digitizer.gui import AxisCalibrationDialog


def axes():
    return {
        "x": {"scale": "linear", "points": [
            {"pixel": [7.123456789, 80], "value": -10, "text": "−10"},
            {"pixel": [52, 80], "value": 10, "text": "10"}],
            "rejected": [{"pixel": [33, 80], "value": 99, "text": "99"}]},
        "y": {"scale": "log", "points": [
            {"pixel": [7, 15], "value": 1e-5}, {"pixel": [7, 80], "value": 1e-10}]},
    }


class CalibrationBenchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.bench = CalibrationBench(str)
        self.bench.load_axes(axes(), (90, 110))
        self.bench.resize(400, 425)
        self.bench.show()
        QTest.qWait(10)

    def tearDown(self):
        self.bench.close()
        self.bench.deleteLater()
        QTest.qWait(5)

    def test_single_click_value_edits_and_keyboard_moves_one_original_pixel(self):
        table = self.bench.tables["x"]
        selected = table.item(1, 3)
        QTest.mouseClick(table.viewport(), Qt.LeftButton, pos=table.visualItemRect(selected).center())
        editor = QApplication.focusWidget()
        self.assertIsInstance(editor, QLineEdit)
        editor.selectAll()
        QTest.keyClicks(editor, "25")
        self.assertTrue(self.bench.apply_button.isEnabled())
        # Applying from an active cell commits the delegate's text.
        self.assertEqual(self.bench.reviewed_axes()["x"]["points"][1]["value"], 25)
        self.assertTrue(self.bench.dirty)
        self.assertEqual(self.bench.selection(), ("x", 1))
        table.setFocus()
        QTest.keyClick(table, Qt.Key_Right, Qt.AltModifier)
        self.assertEqual(self.bench.reviewed_axes()["x"]["points"][1]["pixel"], [53, 80])
        self.bench.revert()
        self.assertFalse(self.bench.dirty)
        self.assertEqual(self.bench.reviewed_axes()["x"]["points"][0]["pixel"][0], 7.123456789)
        QTest.mouseClick(table.viewport(), Qt.LeftButton, pos=table.visualItemRect(table.item(1, 3)).center())
        editor = QApplication.focusWidget()
        editor.selectAll()
        QTest.keyClicks(editor, "-1.05e1")
        self.assertEqual(editor.text(), "-1.05e1")
        QTest.keyClick(editor, Qt.Key_Escape)
        self.assertFalse(self.bench.dirty)
        self.assertEqual(table.item(1, 3).text(), "10")

    def test_accept_copies_excluded_suggestions_selection_and_clean_baseline(self):
        self.bench.select_tick("x", 2)
        item = self.bench.tables["x"].item(2, 3)
        item.setText("5")
        self.bench.tables["x"].item(2, 0).setCheckState(Qt.Checked)
        self.bench.select_tick("y", 1)
        self.bench.set_pixel(7, 81.5)
        copied = CalibrationBench(str)
        self.addCleanup(copied.deleteLater)
        copied.accept_draft(self.bench)
        self.assertEqual(copied.reviewed_axes(), self.bench.reviewed_axes())
        self.assertEqual(copied.tables["x"].item(2, 4).text(), "99")
        self.assertEqual(copied.selection(), ("y", 1))
        self.assertFalse(copied.dirty)
        copied.plus_button.click()
        copied.accept_draft()
        copied.set_pixel(9, 60)
        copied.revert_button.click()
        self.assertEqual(copied.reviewed_axes()["y"]["points"][1]["pixel"], [7, 82.5])

    def test_invalid_checked_tick_reports_row_and_can_be_excluded_or_removed(self):
        table = self.bench.tables["x"]
        self.bench.add_button.click()
        row = table.currentRow()
        self.assertTrue(self.bench.set_pixel(31, 80))
        self.assertFalse(self.bench.valid)
        self.assertIn(f"X{row+1}", self.bench.error.text())
        self.assertIsNone(self.bench.preview_points()["x"][-1]["value"])
        table.item(row, 3).setText("2")
        self.assertTrue(self.bench.valid)
        table.item(row, 1).setText("111")
        self.assertFalse(self.bench.apply_button.isEnabled())
        with self.assertRaises(ValueError):
            self.bench.reviewed_axes()
        table.item(row, 0).setCheckState(Qt.Unchecked)
        self.assertTrue(self.bench.valid)
        self.bench.remove_button.click()
        self.assertEqual(table.rowCount(), 3)
        self.bench.load_axes({}, None)
        self.assertIsNone(self.bench.selection())
        self.assertFalse(self.bench.valid)
        self.assertFalse(self.bench.isEnabled())

    def test_use_all_controls_only_its_axis_and_tracks_individual_ticks(self):
        table = self.bench.tables["x"]
        use_all = self.bench.use_all_checks["x"]
        self.assertEqual(use_all.checkState(), Qt.PartiallyChecked)
        use_all.click()
        self.assertEqual(use_all.checkState(), Qt.Checked)
        self.assertTrue(all(table.item(row, 0).checkState() == Qt.Checked for row in range(table.rowCount())))
        use_all.click()
        self.assertEqual(use_all.checkState(), Qt.Unchecked)
        self.assertTrue(all(table.item(row, 0).checkState() == Qt.Unchecked for row in range(table.rowCount())))
        table.item(0, 0).setCheckState(Qt.Checked)
        self.assertEqual(use_all.checkState(), Qt.PartiallyChecked)
        self.assertEqual(self.bench.use_all_checks["y"].checkState(), Qt.Checked)
        self.bench.revert()
        self.assertTrue(self.bench.valid)
        self.assertFalse(self.bench.dirty)

    def test_ocr_preview_selects_marker_and_zoomed_placement_uses_source_coordinates(self):
        parent = QWidget()
        parent._tr = str
        parent.canvas = SimpleNamespace(image=np.full((90, 110, 3), 255, np.uint8), roi=(7, 15, 100, 80))
        result = {"axes": axes()}
        with patch("plot_digitizer.gui.suggest_axis_calibration", return_value=result):
            dialog = AxisCalibrationDialog(parent)
            loop = QEventLoop()
            dialog.worker.finished.connect(loop.quit)
            QTimer.singleShot(3000, loop.quit)
            if dialog.worker.isRunning():
                loop.exec()
            QTest.qWait(15)
        try:
            self.assertTrue(dialog.apply_button.isEnabled())
            dialog.bench.select_tick("x", 2)
            self.assertFalse(dialog.bench.preview_points(include_selected_excluded=True)["x"][2]["checked"])
            self.assertEqual(len(dialog.bench.reviewed_axes()["x"]["points"]), 2)
            self.assertTrue(dialog.zoom_tick.isEnabled())
            def click(x, y):
                dialog._set_tick_on_image(SimpleNamespace(inaxes=dialog.axes, button=1, xdata=x, ydata=y))
            click(7, 15)
            self.assertEqual(dialog.bench.selection(), ("y", 0))
            dialog.zoom_tick.setChecked(True)
            left, right = dialog.axes.get_xlim()
            self.assertLess(right-left, 110)
            dialog.bench.pick_button.click()
            click(9.5, 18.25)
            dialog._accept_review()
            self.assertEqual(dialog.reviewed["axes"]["y"]["points"][0]["pixel"], [9.5, 18.25])
        finally:
            dialog.close()
            parent.close()
            parent.deleteLater()
            QTest.qWait(5)


if __name__ == "__main__":
    unittest.main()
