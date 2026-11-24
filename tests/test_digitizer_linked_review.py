"""A review selection must identify the same measurement in pixels and calibrated space."""

import os
from types import SimpleNamespace
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from plot_digitizer.digitizer_linked_review import LinkedReview


class LinkedReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.widget = LinkedReview(str)
        self.widget.resize(850, 540)
        self.widget.show()
        self.image = np.full((64, 110, 3), 255, dtype=np.uint8)
        self.image[23, 40] = (12, 81, 44)
        self.data = [
            {"label": "Empty", "pts_real": [], "pts_px": []},
            {"label": "Sample", "pts_real": [[.1, 10], [10, 300], [100, 70]],
             "pts_px": [[0, 0], [40, 23], [109, 63]]},
            {"label": "Reference", "pts_real": [[2, 5]], "pts_px": [[4, 18]]},
        ]
        self.events = []
        self.widget.pointSelected.connect(lambda ds, point: self.events.append((ds, point)))

    def tearDown(self):
        self.widget.close()
        self.widget.deleteLater()
        QTest.qWait(5)

    def test_table_and_log_graph_select_the_same_sample_and_center_original_pixels(self):
        self.widget.refresh(self.image, self.data, 0, "loglog")
        self.assertEqual(self.widget.selected_point(), (1, 0))
        self.assertEqual(self.events, [])
        self.assertEqual(self.widget.table.rowCount(), 4)
        self.assertEqual(self.widget.axes.get_xscale(), "log")
        self.assertEqual(self.widget.axes.get_yscale(), "log")
        self.widget.table.setFocus()
        QTest.keyClick(self.widget.table, Qt.Key_Down)
        self.assertEqual(self.widget.selected_point(), (1, 1))
        self.assertEqual(self.widget.source_detail.point, (40, 23))
        self.assertEqual(self.widget.point_number.value(), 2)
        self.assertEqual(self.widget.source_position.text(), "Source pixel (40, 23)")
        crop = self.widget.source_detail.grab().toImage()
        center = crop.pixelColor(crop.width() // 2, crop.height() // 2)
        self.assertEqual(center.getRgb()[:3], (12, 81, 44))
        self.widget.trace_canvas.draw()
        x, y = self.widget.axes.transData.transform((100, 70))
        self.widget._pick_trace(SimpleNamespace(inaxes=self.widget.axes, button=1, x=x, y=y))
        self.assertEqual(self.widget.table.currentRow(), 2)
        self.assertEqual(self.widget.source_detail.point, (109, 63))
        self.widget.table.setCurrentCell(3, 0)
        self.assertEqual(self.widget.dataset_picker.currentText(), "Reference")
        self.assertEqual(self.widget.source_detail.point, (4, 18))
        self.widget.dataset_picker.setCurrentIndex(0)
        self.widget.point_number.setValue(2)
        self.assertEqual(self.events[-1], (1, 1))

    def test_recalibration_deletion_and_reopen_preserve_surviving_pixel_only(self):
        self.widget.refresh(self.image, self.data, 1, "linear")
        self.widget.select_point(1, 1)
        self.data[1]["pts_real"][1] = [17.5, 420]
        self.data[1]["pts_px"].pop(0)
        self.data[1]["pts_real"].pop(0)
        self.events.clear()
        self.widget.refresh(self.image, self.data, 1, "linear")
        self.assertEqual(self.widget.selected_point(), (1, 0))
        self.assertEqual(self.widget.source_detail.point, (40, 23))
        self.assertEqual(self.widget.table.item(0, 1).text(), "17.5")
        self.assertEqual(self.events, [])
        self.widget.refresh(self.image, self.data, 2, "linear")
        self.assertEqual(self.widget.selected_point(), (2, 0))
        self.widget.refresh(self.image.copy(), [], 0, "linear")
        self.assertIsNone(self.widget.selected_point())
        self.assertIsNone(self.widget.source_detail.point)
        self.assertEqual(self.widget.point_number.suffix(), "")
        self.assertEqual(self.widget.table.rowCount(), 0)

    def test_scheme_and_legacy_coordinates_without_source_pixels(self):
        self.widget.refresh(None, [{"label":"Legacy", "pts_real":[[2, 4]]}], 0, "linear")
        self.assertEqual(self.widget.selected_point(), (0, 0))
        self.assertEqual(self.widget.source_position.text(), "Source pixel unavailable")
        palette = self.widget.palette()
        palette.setColor(QPalette.Base, QColor("#18242d"))
        palette.setColor(QPalette.Highlight, QColor("#86e1d4"))
        self.widget.setPalette(palette)
        self.widget.refresh_palette()
        self.assertEqual(self.widget._line.get_color(), "#86e1d4")
        self.assertEqual(self.widget._marker.get_markerfacecolor(), "#18242d")
        self.widget.refresh(self.image, self.data, 1, "linear")
        font = QFont(self.widget.font())
        font.setPointSize(14)
        self.widget.setFont(font)
        self.widget.resize(320, 402)
        QTest.qWait(15)
        self.assertTrue(self.widget.compact_tabs.isVisible())
        self.widget.compact_tabs.setCurrentIndex(1)
        self.widget.select_point(1, 1)
        self.assertTrue(self.widget.table.isVisible())
        self.assertTrue(self.widget.source_detail.isVisible())
        self.assertGreater(self.widget.source_detail.height(), 30)
        self.assertGreater(self.widget.dataset_picker.width(), 80)
        self.widget.resize(900, 540)
        QTest.qWait(15)
        self.assertTrue(self.widget.trace_canvas.isVisible())
        self.assertTrue(self.widget.source_detail.isVisible())
        self.assertEqual(self.widget.selected_point(), (1, 1))

    def test_nonpositive_legacy_log_points_remain_selectable_in_table(self):
        data = [{"label": "Legacy", "pts_real": [[0, 0], [-1, 15]], "pts_px": [[2, 3], [4, 5]]}]
        self.widget.refresh(self.image, data, 0, "loglog")
        self.widget.trace_canvas.draw()
        self.widget.point_number.setValue(2)
        self.assertEqual(self.widget.table.item(1, 1).text(), "-1")
        self.assertEqual(self.widget.source_detail.point, (4, 5))
        self.assertEqual(len(self.widget._marker.get_xdata()), 0)


if __name__ == "__main__":
    unittest.main()
