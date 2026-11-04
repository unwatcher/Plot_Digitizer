"""Local OCR adapter failures, cancellation and coordinate preservation."""

import subprocess
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from plot_digitizer.axis_ocr import _normalize_word, _run_ocr_process, read_local_ocr


TSV = "text\tleft\ttop\twidth\theight\n100\t15\t2\t25\t12\nbad\tnan\t2\t25\t12\n"


class LocalOcrAdapterTests(unittest.TestCase):
    def test_invalid_and_nonfinite_boxes_are_ignored(self):
        valid = {"text": "100", "x": 1, "y": 2, "width": 20, "height": 12}
        self.assertEqual(_normalize_word(valid)["text"], "100")
        for change in ({"x": "nan"}, {"height": 0}, {"width": "invalid"}, {"y": -1}, {"text": ""}):
            self.assertIsNone(_normalize_word(dict(valid, **change)))
        self.assertIsNone(_normalize_word(None))

    def test_windows_failure_falls_back_and_keeps_all_tile_coordinates(self):
        images = {"x": np.full((25, 1800, 3), 255, np.uint8)}
        with patch("plot_digitizer.axis_ocr.sys.platform", "win32"), \
                patch("plot_digitizer.axis_ocr._read_windows_ocr", side_effect=RuntimeError("Unavailable")), \
                patch("plot_digitizer.axis_ocr.shutil.which", return_value="tesseract"), \
                patch("plot_digitizer.axis_ocr._run_ocr_process", return_value=TSV) as run:
            result = read_local_ocr(images)
        self.assertEqual(result["x"]["engine"], "Tesseract")
        self.assertEqual([word["x"] for word in result["x"]["words"]], [15, 755, 1495])
        self.assertEqual(run.call_count, 3)

    def test_windows_cancellation_never_starts_another_engine(self):
        with patch("plot_digitizer.axis_ocr.sys.platform", "win32"), \
                patch("plot_digitizer.axis_ocr._read_windows_ocr", side_effect=InterruptedError("Cancelled")), \
                patch("plot_digitizer.axis_ocr.shutil.which") as search:
            with self.assertRaises(InterruptedError):
                read_local_ocr({"x": np.full((25, 80, 3), 255, np.uint8)})
        search.assert_not_called()

    def test_no_installed_engine_reports_a_clear_error(self):
        with patch("plot_digitizer.axis_ocr.sys.platform", "linux"), \
                patch("plot_digitizer.axis_ocr.shutil.which", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "needs local OCR"):
                read_local_ocr({"x": np.full((25, 80, 3), 255, np.uint8)})

    def test_timeout_kills_and_reaps_the_ocr_process(self):
        process = MagicMock()
        process.poll.return_value = None
        with patch("plot_digitizer.axis_ocr.subprocess.Popen", return_value=process), \
                patch("plot_digitizer.axis_ocr.time.monotonic", side_effect=[0, 100]):
            with self.assertRaisesRegex(RuntimeError, "timed out"):
                _run_ocr_process(["unused"], lambda: False)
        process.kill.assert_called_once()
        process.communicate.assert_called_once()

    def test_cancellation_during_ocr_kills_and_reaps_the_process(self):
        process = MagicMock()
        process.poll.return_value = None
        process.communicate.side_effect = [subprocess.TimeoutExpired("unused", .1), (b"", b"")]
        with patch("plot_digitizer.axis_ocr.subprocess.Popen", return_value=process):
            cancelled = MagicMock(side_effect=[False, False, True])
            with self.assertRaises(InterruptedError):
                _run_ocr_process(["unused"], cancelled)
        process.kill.assert_called_once()
        self.assertEqual(process.communicate.call_count, 2)

    def test_invalid_images_and_pre_cancelled_requests_do_not_launch_a_process(self):
        with patch("plot_digitizer.axis_ocr.subprocess.Popen") as process:
            with self.assertRaises(ValueError):
                read_local_ocr({"x": np.empty((0, 10, 3), np.uint8)})
            with self.assertRaises(InterruptedError):
                read_local_ocr({"x": np.full((25, 80, 3), 255, np.uint8)}, cancelled=lambda: True)
        process.assert_not_called()


if __name__ == "__main__":
    unittest.main()
