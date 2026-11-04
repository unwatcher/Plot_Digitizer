import copy
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
from PySide6.QtCore import QPoint, QRect, QTimer
from PySide6.QtGui import QFont
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QMessageBox

from plot_digitizer.gui import (
    CurvePickerDialog,
    MainWindow,
    detect_plot_roi,
    extract_plot_curves,
)


def draw_trace(image, baseline, color=(0, 0, 0), amplitude=16, dashed=False):
    xs = np.arange(12, image.shape[1] - 12)
    ys = baseline + amplitude * np.sin(xs / 31)
    points = np.column_stack((xs, np.rint(ys))).astype(np.int32)
    if dashed:
        for start in range(0, len(points), 18):
            cv2.polylines(image, [points[start:start + 10]], False, color, 2, cv2.LINE_AA)
    else:
        cv2.polylines(image, [points], False, color, 2, cv2.LINE_AA)
    return xs, ys


def framed_plot(boxed=False, color=0):
    image = np.full((280, 480, 3), 255, np.uint8)
    ink = (color, color, color)
    cv2.line(image, (40, 25), (40, 245), ink, 2)
    cv2.line(image, (40, 245), (450, 245), ink, 2)
    if boxed:
        cv2.line(image, (40, 25), (450, 25), ink, 2)
        cv2.line(image, (450, 25), (450, 245), ink, 2)
    xs = np.arange(50, 440)
    points = np.column_stack((xs, 145 + 22 * np.sin(xs / 31))).astype(np.int32)
    cv2.polylines(image, [points], False, (35, 65, 210), 2)
    return image


class AutoRoiDetectionTests(unittest.TestCase):
    def test_l_shaped_axes_exclude_frame_and_preserve_plot_interior(self):
        image = framed_plot()
        original = image.copy()
        left, top, right, bottom = detect_plot_roi(image)
        self.assertTrue(41 < left < 48 and 24 < top < 32)
        self.assertTrue(442 < right < 451 and 237 < bottom < 244)
        np.testing.assert_array_equal(image, original)

    def test_boxed_frame_excludes_all_four_borders(self):
        image = framed_plot(boxed=True)
        left, top, right, bottom = detect_plot_roi(image)
        self.assertTrue(41 < left < 48 and 26 < top < 33)
        self.assertTrue(442 < right < 449 and 237 < bottom < 244)
        self.assertTrue(np.all(image[top:bottom, left] == 255))
        self.assertTrue(np.all(image[top, left:right] == 255))
        self.assertTrue(np.all(image[top:bottom, right - 1] == 255))
        self.assertTrue(np.all(image[bottom - 1, left:right] == 255))

    def test_faint_frame_with_short_breaks_is_detected(self):
        image = framed_plot(boxed=True, color=225)
        image[90:92, 38:43] = 255
        image[243:248, 230:233] = 255
        box = detect_plot_roi(image)
        self.assertIsNotNone(box)
        self.assertGreater(box[0], 40)
        self.assertLess(box[2], 450)

    def test_grid_does_not_shrink_roi_to_an_inner_cell(self):
        image = framed_plot(boxed=True)
        for row in (65, 105, 185):
            cv2.line(image, (40, row), (450, row), (160, 160, 160), 1)
        for col in (120, 220, 320, 400):
            cv2.line(image, (col, 25), (col, 245), (160, 160, 160), 1)
        box = detect_plot_roi(image)
        self.assertGreater(box[2] - box[0], 395)
        self.assertGreater(box[3] - box[1], 205)

    def test_large_scan_coordinates_are_returned_in_source_pixels(self):
        image = np.full((1600, 3600, 3), 255, np.uint8)
        cv2.rectangle(image, (240, 120), (3400, 1400), (0, 0, 0), 4)
        box = detect_plot_roi(image)
        self.assertTrue(242 <= box[0] < 265 and 122 <= box[1] < 145)
        self.assertTrue(3375 < box[2] < 3399 and 1375 < box[3] < 1399)

    def test_missing_frame_does_not_invent_an_roi(self):
        image = np.full((280, 480, 3), 255, np.uint8)
        self.assertIsNone(detect_plot_roi(image))
        draw_trace(image, 120)
        cv2.putText(image, "Plot title", (100, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        self.assertIsNone(detect_plot_roi(image))
        self.assertIsNone(detect_plot_roi(image[:8, :8]))

    def test_instrument_table_below_a_boxed_plot_does_not_expand_the_roi(self):
        image = np.full((340, 520, 3), 255, np.uint8)
        cv2.rectangle(image, (40, 25), (480, 250), (0, 0, 0), 1)
        cv2.line(image, (40, 250), (40, 280), (0, 0, 0), 1)
        cv2.line(image, (40, 280), (480, 280), (0, 0, 0), 1)
        cv2.line(image, (40, 300), (480, 300), (0, 0, 0), 1)
        roi = detect_plot_roi(image)
        self.assertIsNotNone(roi)
        self.assertLess(roi[3], 253)
        self.assertGreater(roi[3], 242)


class AutopilotExtractionTests(unittest.TestCase):
    def test_faint_trace_selects_a_threshold_that_retains_observed_pixels(self):
        image = np.full((240, 420, 3), 255, np.uint8)
        xs, ys = draw_trace(image, 135, (229, 229, 229), amplitude=35)
        result = extract_plot_curves(image)
        self.assertEqual(len(result["candidates"]), 1)
        self.assertGreater(result["threshold"], 220)
        points = np.asarray(result["candidates"][0]["points"])
        self.assertGreater(result["candidates"][0]["coverage"], .9)
        self.assertLess(np.median(abs(points[:, 1] - np.interp(points[:, 0], xs, ys))), 2)

    def test_colored_and_neutral_curves_remain_separate(self):
        image = np.full((270, 420, 3), 255, np.uint8)
        expected = []
        for baseline, color in ((50, (210, 30, 35)), (125, (35, 65, 210)), (205, (20, 20, 20))):
            xs, ys = draw_trace(image, baseline, color)
            expected.append((xs, ys))
        original = image.copy()
        result = extract_plot_curves(image)
        self.assertEqual(len(result["candidates"]), 3)
        self.assertEqual([candidate["color"] for candidate in result["candidates"]], ["Red", "Blue", "Black / gray"])
        for candidate, (xs, ys) in zip(result["candidates"], expected):
            points = np.asarray(candidate["points"])
            self.assertGreater(candidate["coverage"], 0.90)
            self.assertLess(np.median(abs(points[:, 1] - np.interp(points[:, 0], xs, ys))), 1.5)
        np.testing.assert_array_equal(image, original)

    def test_stacked_black_curves_select_waterfall_without_manual_bands(self):
        image = np.full((300, 420, 3), 255, np.uint8)
        for baseline in (65, 150, 235):
            draw_trace(image, baseline)
        result = extract_plot_curves(image)
        self.assertEqual(result["preset"], "waterfall")
        self.assertEqual(len(result["candidates"]), 3)
        for candidate, baseline in zip(result["candidates"], (65, 150, 235)):
            self.assertLess(max(abs(y - baseline) for _, y in candidate["points"]), 19)

    def test_sharp_stacked_pxrd_peaks_are_not_mistaken_for_grid_lines(self):
        image = np.full((360, 620, 3), 255, np.uint8)
        xs = np.arange(14, 606)
        expected = []
        for baseline, shift in ((105, 0), (220, 30), (330, -15)):
            ys = baseline + 1.5 * np.sin(xs / 9)
            for center, height, width in ((80 + shift, 45, 2), (210 + shift, 70, 3), (370 + shift, 55, 4), (520 + shift, 35, 2)):
                ys -= height * np.exp(-0.5 * ((xs - center) / width) ** 2)
            expected.append(ys)
            cv2.polylines(image, [np.column_stack((xs, np.rint(ys))).astype(np.int32)], False, (0, 0, 0), 2, cv2.LINE_AA)
        result = extract_plot_curves(image)
        self.assertEqual(result["preset"], "waterfall")
        self.assertEqual(len(result["candidates"]), 3)
        for candidate, expected_y in zip(result["candidates"], expected):
            self.assertGreater(candidate["coverage"], 0.90)
            self.assertAlmostEqual(min(y for _, y in candidate["points"]), min(expected_y), delta=3)

    def test_colored_crossings_still_produce_separate_curves(self):
        image = np.full((200, 420, 3), 255, np.uint8)
        cv2.line(image, (12, 30), (408, 165), (210, 30, 35), 2, cv2.LINE_AA)
        cv2.line(image, (12, 165), (408, 30), (35, 65, 210), 2, cv2.LINE_AA)
        candidates = extract_plot_curves(image)["candidates"]
        self.assertEqual(len(candidates), 2)
        self.assertTrue(all(candidate["coverage"] > 0.90 for candidate in candidates))

    def test_grid_is_removed_and_grid_preset_selected(self):
        image = np.full((240, 420, 3), 255, np.uint8)
        for row in (25, 75, 125, 175, 225):
            cv2.line(image, (0, row), (419, row), (145, 145, 145), 1)
        for col in (60, 140, 220, 300, 380):
            cv2.line(image, (col, 0), (col, 239), (145, 145, 145), 1)
        draw_trace(image, 155, amplitude=35)
        result = extract_plot_curves(image)
        self.assertEqual(result["preset"], "grid")
        self.assertTrue(result["suppress_grid"])
        self.assertEqual(len(result["candidates"]), 1)
        points = np.asarray(result["candidates"][0]["points"])
        self.assertLess(np.median(abs(points[:, 1] - (155 + 35 * np.sin(points[:, 0] / 31)))), 2)

    def test_dashed_curve_keeps_only_observed_columns(self):
        image = np.full((150, 420, 3), 255, np.uint8)
        draw_trace(image, 70, (35, 65, 210), dashed=True)
        result = extract_plot_curves(image)
        self.assertEqual(len(result["candidates"]), 1)
        candidate = result["candidates"][0]
        self.assertGreater(candidate["coverage"], 0.45)
        self.assertLess(candidate["coverage"], 0.90)
        for x, y in candidate["points"]:
            self.assertLess(int(np.min(image[round(y), round(x)])), 240)

    def test_roi_excludes_other_curves_and_preserves_source_coordinates(self):
        image = np.full((240, 420, 3), 255, np.uint8)
        draw_trace(image, 55)
        draw_trace(image, 180)
        result = extract_plot_curves(image, (40, 135, 380, 220))
        self.assertEqual(len(result["candidates"]), 1)
        self.assertFalse(result["inferred_roi"])
        self.assertTrue(all(40 <= x < 380 and 135 <= y < 220 for x, y in result["candidates"][0]["points"]))

    def test_axis_frame_is_inferred_and_not_returned_as_data(self):
        image = np.full((260, 460, 3), 255, np.uint8)
        cv2.line(image, (35, 15), (35, 235), (0, 0, 0), 2)
        cv2.line(image, (35, 235), (440, 235), (0, 0, 0), 2)
        xs = np.arange(45, 430)
        points = np.column_stack((xs, 130 + 22 * np.sin(xs / 31))).astype(np.int32)
        cv2.polylines(image, [points], False, (0, 0, 0), 2)
        result = extract_plot_curves(image)
        self.assertTrue(result["inferred_roi"])
        self.assertEqual(len(result["candidates"]), 1)
        self.assertGreater(result["roi"][0], 35)
        self.assertLess(result["roi"][3], 235)

    def test_blank_and_labels_do_not_become_curves(self):
        image = np.full((180, 360, 3), 255, np.uint8)
        self.assertEqual(extract_plot_curves(image)["candidates"], [])
        cv2.putText(image, "A", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        self.assertEqual(extract_plot_curves(image)["candidates"], [])

    def test_flat_curve_is_retained_without_grid_evidence(self):
        image = np.full((150, 420, 3), 255, np.uint8)
        draw_trace(image, 70, amplitude=0)
        result = extract_plot_curves(image)
        self.assertEqual(result["preset"], "normal")
        self.assertEqual(len(result["candidates"]), 1)

    def test_large_scan_is_bounded_and_maps_back_to_original_pixels(self):
        image = np.full((600, 3600, 3), 255, np.uint8)
        xs, ys = draw_trace(image, 310, (35, 65, 210), amplitude=50)
        result = extract_plot_curves(image, (200, 150, 3400, 500))
        self.assertEqual(len(result["candidates"]), 1)
        points = np.asarray(result["candidates"][0]["points"])
        self.assertLessEqual(len(points), 1800)
        self.assertGreater(points[-1, 0], 3390)
        self.assertLess(np.median(abs(points[:, 1] - np.interp(points[:, 0], xs, ys))), 2)

    def test_invalid_rois_and_cancellation(self):
        image = np.full((180, 360, 3), 255, np.uint8)
        for roi in ((-20, -20, -1, -1), (20, 20, 20, 20), (0, 0, float("nan"), 80)):
            with self.assertRaises(ValueError):
                extract_plot_curves(image, roi)
        with self.assertRaises(InterruptedError):
            extract_plot_curves(image, cancelled=lambda: True)


class AutopilotWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.window = MainWindow(standalone=True)
        image = np.full((240, 420, 3), 255, np.uint8)
        draw_trace(image, 65, (210, 30, 35))
        draw_trace(image, 175, (35, 65, 210))
        self.window.canvas.image = image
        self.window.canvas.fit_x_calib([(0, 0), (420, 0)], [5, 47])
        self.window.canvas.fit_y_calib([(0, 0), (0, 240)], [240, 0])
        self.dialog = None

    def tearDown(self):
        if self.dialog:
            self.dialog.reject()
            self.wait_until(lambda: not self.dialog.worker.isRunning())
            self.dialog.close()
            self.dialog.deleteLater()
        self.window.close()
        self.window.deleteLater()
        QTest.qWait(10)

    def wait_until(self, predicate):
        deadline = time.monotonic() + 15
        while not predicate() and time.monotonic() < deadline:
            QTest.qWait(5)
        self.assertTrue(predicate(), "Curve detection did not finish")

    def picker(self):
        self.dialog = CurvePickerDialog(self.window)
        self.dialog.show()
        self.wait_until(lambda: self.dialog.progress.isHidden())
        self.app.processEvents()
        return self.dialog

    def test_picker_filters_colors_and_selects_curve_by_preview(self):
        dialog = self.picker()
        self.assertEqual(dialog.curves.rowCount(), 2)
        dialog.colors.setCurrentIndex(dialog.colors.findData("Blue"))
        self.assertTrue(dialog.curves.isRowHidden(0))
        self.assertEqual(dialog.curves.currentRow(), 1)
        dialog.preview.draw()
        point = dialog.result_data["candidates"][1]["points"][40]
        x, y = dialog.axes.transData.transform(point)
        dialog._pick_on_preview(SimpleNamespace(inaxes=dialog.axes, x=x, y=y, button=1))
        dialog._use_selected()
        self.assertEqual(dialog.result(), QDialog.Accepted)
        self.assertEqual(dialog.selected_indices, [1])

    def test_import_preserves_existing_data_and_uses_calibration(self):
        self.window.canvas.bulk_add_pixels([(10, 20)])
        before = copy.deepcopy(self.window.canvas.datasets[0])
        result = extract_plot_curves(self.window.canvas.image)
        self.window._import_autopilot_curves(result, [1])
        self.assertEqual(self.window.canvas.datasets[0], before)
        self.assertEqual(len(self.window.canvas.datasets), 2)
        selected = self.window.canvas.datasets[1]
        self.assertIn("Blue", selected["label"])
        self.assertGreater(len(selected["pts_real"]), 350)
        for pixel, real in zip(selected["pts_px"], selected["pts_real"]):
            self.assertAlmostEqual(real[0], 5 + pixel[0] * 0.1, places=5)
            self.assertAlmostEqual(real[1], 240 - pixel[1], places=5)
        self.assertEqual(self.window.combo_ds.currentIndex(), 1)
        self.assertEqual(self.window.combo_ds_auto.currentIndex(), 1)
        self.assertEqual(self.window.table.rowCount(), len(selected["pts_real"]) + 1)

    def test_add_all_respects_color_filter_and_repeated_imports_have_unique_names(self):
        dialog = self.picker()
        dialog.colors.setCurrentIndex(dialog.colors.findData("Blue"))
        dialog._use_all()
        self.assertEqual(dialog.selected_indices, [1])
        self.window._import_autopilot_curves(dialog.result_data, [0, 1])
        self.window._import_autopilot_curves(dialog.result_data, [0, 1])
        labels = [dataset["label"] for dataset in self.window.canvas.datasets]
        self.assertEqual(len(labels), len(set(labels)))
        self.assertEqual(len(labels), 5)

    def test_cancelling_detection_does_not_change_data_roi_or_preset(self):
        before = copy.deepcopy(self.window.canvas.datasets)
        preset = self.window.combo_extraction_preset.currentData()
        self.dialog = CurvePickerDialog(self.window)
        self.dialog.reject()
        self.wait_until(lambda: not self.dialog.worker.isRunning())
        self.app.processEvents()
        self.assertEqual(self.dialog.result(), QDialog.Rejected)
        self.assertEqual(self.window.canvas.datasets, before)
        self.assertIsNone(self.window.canvas.roi)
        self.assertEqual(self.window.combo_extraction_preset.currentData(), preset)

    def test_calibration_is_required_before_autopilot(self):
        self.window.canvas._x_calib_ready = False
        with patch.object(QMessageBox, "warning") as warning, patch("plot_digitizer.gui.CurvePickerDialog") as picker:
            self.window.autopilot_button.click()
        warning.assert_called_once()
        picker.assert_not_called()

    def test_auto_roi_action_works_before_calibration_and_preserves_data(self):
        self.window.canvas.bulk_add_pixels([(60, 80)])
        before = copy.deepcopy(self.window.canvas.datasets)
        self.window.canvas.image = framed_plot()
        self.window.canvas._x_calib_ready = False
        self.window.canvas._y_calib_ready = False
        self.window.auto_roi_button.click()
        self.assertEqual(self.window.canvas.roi, detect_plot_roi(self.window.canvas.image))
        self.assertEqual(self.window.canvas.datasets, before)
        self.assertTrue(self.window.canvas.ax.patches)
        self.assertIn("Check the green box", self.window.status.currentMessage())
        self.assertFalse(self.window.canvas._x_calib_ready)

    def test_auto_roi_failure_preserves_manual_box(self):
        manual = (30, 40, 350, 220)
        self.window.canvas.roi = manual
        self.window.auto_roi_button.click()
        self.assertEqual(self.window.canvas.roi, manual)
        self.assertIn("ROI unchanged", self.window.status.currentMessage())

    def test_auto_roi_can_be_replaced_by_a_manual_box_and_cleared(self):
        canvas = self.window.canvas
        canvas.image = framed_plot()
        self.window.auto_roi_button.click()
        self.window._start_roi()
        canvas._on_press(SimpleNamespace(inaxes=canvas.ax, button=1, xdata=60, ydata=70))
        canvas._on_release(SimpleNamespace(xdata=400, ydata=210))
        self.assertEqual(canvas.roi, (60, 70, 400, 210))
        result = extract_plot_curves(canvas.image, canvas.roi)
        self.assertEqual(result["roi"], canvas.roi)
        self.assertFalse(result["inferred_roi"])
        self.window._clear_roi()
        self.assertIsNone(canvas.roi)

    def test_auto_roi_on_image_load_can_be_disabled(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "framed.png"
            cv2.imwrite(str(path), cv2.cvtColor(framed_plot(), cv2.COLOR_RGB2BGR))
            with patch("plot_digitizer.gui.QFileDialog.getOpenFileName", return_value=(str(path), "")):
                self.window._open_image()
                self.assertIsNotNone(self.window.canvas.roi)
                self.window.chk_auto_roi.setChecked(False)
                self.window._open_image()
                self.assertIsNone(self.window.canvas.roi)

    def test_auto_roi_requires_an_image_and_is_localized(self):
        self.window.canvas.image = None
        with patch.object(QMessageBox, "warning") as warning:
            self.window.auto_roi_button.click()
        warning.assert_called_once()
        self.window.set_language("zh")
        self.assertEqual(self.window.auto_roi_button.text(), "自动检测 ROI")
        self.assertEqual(self.window.chk_auto_roi.text(), "打开图像时自动检测 ROI")

    def test_autopilot_button_completes_the_picker_and_import_flow(self):
        timer = QTimer()
        timer.setInterval(5)

        def choose_when_ready():
            dialog = next((widget for widget in self.window.findChildren(CurvePickerDialog) if widget.use_button.isEnabled()), None)
            if dialog:
                timer.stop()
                dialog.curves.selectRow(1)
                dialog.use_button.click()

        timer.timeout.connect(choose_when_ready)
        timer.start()
        watchdog = QTimer()
        watchdog.setSingleShot(True)
        watchdog.timeout.connect(self.window.close)
        watchdog.start(15000)
        try:
            self.window.autopilot_button.click()
        finally:
            timer.stop()
            watchdog.stop()
        self.assertEqual(len(self.window.canvas.datasets), 2)
        self.assertIn("Blue", self.window.canvas.datasets[1]["label"])

    def test_empty_detection_explains_recovery_and_disables_import(self):
        self.window.canvas.image[:] = 255
        self.dialog = CurvePickerDialog(self.window)
        self.wait_until(lambda: self.dialog.progress.isHidden())
        self.app.processEvents()
        self.assertIn("Draw a tighter ROI", self.dialog.summary.text())
        self.assertFalse(self.dialog.use_button.isEnabled())
        self.assertFalse(self.dialog.all_button.isEnabled())

    def test_worker_error_keeps_import_disabled(self):
        self.window.canvas.roi = (0, 0, 1, 1)
        self.dialog = CurvePickerDialog(self.window)
        self.wait_until(lambda: self.dialog.progress.isHidden())
        self.app.processEvents()
        self.assertIn("Detection failed", self.dialog.summary.text())
        self.assertFalse(self.dialog.use_button.isEnabled())
        self.assertFalse(self.dialog.all_button.isEnabled())

    def test_chinese_picker_uses_localized_questions_and_actions(self):
        self.window.set_language("zh")
        dialog = self.picker()
        self.assertEqual(dialog.colors.itemText(0), "所有颜色")
        self.assertEqual(dialog.use_button.text(), "添加所选曲线")
        self.assertEqual(dialog.curves.item(0, 1).text(), "红色")

    def test_picker_fits_small_screen_and_large_font_in_both_languages(self):
        font = QFont(self.window.font())
        font.setPointSize(14)
        self.window.setFont(font)
        bounds = QRect(0, 0, 960, 540)
        for language in ("en", "zh"):
            self.window.set_language(language)
            with patch("PySide6.QtGui.QScreen.availableGeometry", return_value=bounds):
                dialog = self.picker()
                QTest.qWait(40)
                self.assertTrue(bounds.contains(dialog.frameGeometry()))
                self.assertEqual(dialog.body_scroll.horizontalScrollBar().maximum(), 0)
                self.assertLess(dialog.summary.geometry().bottom(), dialog.colors.geometry().top())
                point = dialog.buttons.mapTo(dialog, QPoint(0, 0))
                self.assertTrue(dialog.rect().contains(QRect(point, dialog.buttons.size())))
            dialog.close()
            dialog.deleteLater()
            self.dialog = None
            QTest.qWait(10)


if __name__ == "__main__":
    unittest.main()
