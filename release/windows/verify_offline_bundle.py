"""Exercise the packaged interpreter, dependency imports, GUI and extraction."""

import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
DEPS = ROOT / "runtime_deps/py314"
sys.path[:0] = [str(DEPS), str(ROOT)]
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from plot_digitizer.qt_runtime import configure_qt_runtime
configure_qt_runtime(application_root=ROOT)

import cv2
import matplotlib
import numpy as np
import PySide6
from PySide6.QtCore import QTimer
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import QApplication

from plot_digitizer.gui import MainWindow
from plot_digitizer.digitizer_core import extract_plot_curves
from plot_digitizer.axis_calibration import calibration_coefficients


def main():
    metadata = json.loads((ROOT / "release-info.json").read_text(encoding="utf8"))
    expected_python = ROOT / "python314/python.exe"
    if not expected_python.samefile(sys.executable):
        raise RuntimeError("Verification must use the bundled Python interpreter.")
    if sys.version.split()[0] != metadata["python"]["version"]:
        raise RuntimeError("The bundled Python version does not match its manifest.")
    for module in (cv2, matplotlib, np, PySide6):
        if not Path(module.__file__).resolve().is_relative_to(DEPS.resolve()):
            raise RuntimeError("A dependency was imported outside the packaged runtime.")
    app = QApplication.instance() or QApplication([])
    if not QFontDatabase.families():
        raise RuntimeError("No Qt fonts are available.")
    window = MainWindow(standalone=True)
    window.show()
    app.processEvents()
    if window.reference_group.isVisible() or not window.auto_calibration_button.text():
        raise RuntimeError("The standalone window is incomplete.")
    image = cv2.cvtColor(cv2.imread(str(ROOT / "examples/patent_plots/us8198452b2-fig1.png")), cv2.COLOR_BGR2RGB)
    result = extract_plot_curves(image)
    if len(result["candidates"]) != 1 or result["preset"] != "normal":
        raise RuntimeError("Packaged PXRD extraction failed.")
    coefficients = calibration_coefficients([
        {"pixel": [0, 0], "value": 1e-20}, {"pixel": [100, 0], "value": 2e-20}], "x")
    if not np.isclose(np.polyval(coefficients, 100), 2e-20, rtol=1e-12, atol=0):
        raise RuntimeError("Small scientific values lost calibration precision.")
    QTimer.singleShot(20, window.close)
    QTimer.singleShot(40, app.quit)
    app.exec()
    print(json.dumps({"verified": True, "python": sys.version.split()[0], "numpy": np.__version__,
                      "opencv": cv2.__version__, "qt": PySide6.__version__,
                      "curve_count": len(result["candidates"]), "runtime_isolated": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
