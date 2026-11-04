#!/usr/bin/env python3
"""
PlotDigitizer GUI — PySide6 + Matplotlib + OpenCV

Adapted from Plot_Digitizer by unwatcher (BSD 3-Clause License):
https://github.com/unwatcher/Plot_Digitizer

The CrystalTrace catalog integration adds direct, project-specific reference import.
pip install pyside6 matplotlib numpy opencv-python

Calibration: X and Y axes calibrated SEPARATELY via independent linear fits.
Extraction: ROI crop prevents picking up axes/labels/borders.
Dark-curve mode: connectivity filter removes label blobs; topmost pixel of curve component.
Color mode:      connectivity filter + topmost pixel; both avoid label/text contamination.
Autopilot:       detect individual traces, recommend a preset, and preview before import.
"""

import os, sys, csv, json, math
from pathlib import Path
import numpy as np
import cv2

try:
    from .qt_runtime import configure_qt_runtime
    from .ui_layout import AdaptiveDialog, AdaptivePanes, fit_to_screen
    from .ui_appearance import apply_window_appearance, configure_application_appearance
    from .axis_calibration import suggest_axis_calibration, calibration_coefficients
    from .digitizer_core import (
        AUTO_DARK_THRESHOLDS, EXTRACTION_PRESETS, detect_plot_roi,
        extract_by_color, extract_dark_curve, extract_dark_curve_auto, extract_blobs,
        extract_plot_curves, extraction_preset, prepare_reference_points,
        reject_points_in_box, subsample, suppress_grid_line_signals,
    )
except ImportError:  # standalone archive: qt_runtime.py sits beside this file
    from qt_runtime import configure_qt_runtime
    from ui_layout import AdaptiveDialog, AdaptivePanes, fit_to_screen
    from ui_appearance import apply_window_appearance, configure_application_appearance
    from axis_calibration import suggest_axis_calibration, calibration_coefficients
    from digitizer_core import (
        AUTO_DARK_THRESHOLDS, EXTRACTION_PRESETS, detect_plot_roi,
        extract_by_color, extract_dark_curve, extract_dark_curve_auto, extract_blobs,
        extract_plot_curves, extraction_preset, prepare_reference_points,
        reject_points_in_box, subsample, suppress_grid_line_signals,
    )

configure_qt_runtime()

from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QLineEdit, QTableWidget, QDialog,
    QTableWidgetItem, QGroupBox, QSplitter, QComboBox, QStatusBar,
    QMessageBox, QHeaderView, QAbstractItemView, QSizePolicy,
    QSlider, QSpinBox, QDoubleSpinBox, QTabWidget, QScrollArea,
    QColorDialog, QProgressBar, QCheckBox, QTextBrowser, QDialogButtonBox,
)
from PySide6.QtCore import QEvent, Qt, QTimer, QThread, Signal
from PySide6.QtGui import QColor

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.image as mpimg
import matplotlib.patches as mpatches


# ══════════════════════════════════════════════════════════════════════════════
#  User guidance and XRDML export
# ══════════════════════════════════════════════════════════════════════════════

def extraction_instructions_html(language="en"):
    """Return printable instructions for the extraction presets."""
    if str(language).strip().lower() == "zh":
        return """
        <h1>PXRD 专利图提取说明</h1>
        <h2>通用步骤</h2>
        <ol>
          <li>打开图像后会自动检测 ROI。检查绿色框；可关闭“打开图像时自动检测 ROI”，也可点击“自动检测 ROI”重试，或手动绘制框调整范围。</li>
          <li>分别校准 X 与 Y 轴。</li>
          <li>点击“检测所有曲线后选择”。自动驾驶会选择预设并检测曲线；在预览中点击目标曲线，也可按颜色筛选，再添加所选曲线或所有显示的曲线。每条曲线会存入一个新数据集。</li>
          <li>提取后检查彩色数据点；使用“框选拒绝点”拖框删除错误点。</li>
          <li>按 Ctrl+Z 可恢复最近一次框选删除，然后导出或加入参考数据库。</li>
        </ol>
        <p>自动驾驶支持浅色背景上的彩色、暗色和堆叠曲线，并会尝试识别坐标轴边框或带刻度的 X 轴。它会选择暗色阈值、网格抑制和向上峰 / 向下吸收谷的取点方向。添加后，选定的阈值和取点方式会显示在手动设置中。橙色区域表示可能接触或未解析的曲线，需检查曲线身份及遗漏；X 覆盖率不代表精度。若区域识别不准，请先绘制 ROI。散点图请使用斑点模式。</p>
        <h2>手动预设参数</h2>
        <table border="1" cellspacing="0" cellpadding="5">
          <tr><th>预设</th><th>阈值 / 取点</th><th>连通区域过滤</th><th>网格 / Y-band / 最大点数</th></tr>
          <tr><td>普通扫描</td><td>自动 (滑块 160)；步长 1；run 2；topmost</td><td>开启；面积 0.003；宽度 0.05；宽高比 8；Top N 5</td><td>网格关闭；Y-band 关闭；3000</td></tr>
          <tr><td>红外 / 向下吸收谷</td><td>自动 (滑块 160)；步长 1；run 2；bottommost</td><td>开启；面积 0.003；宽度 0.05；宽高比 8；Top N 5</td><td>网格关闭；Y-band 关闭；3000</td></tr>
          <tr><td>网格 / 专利扫描</td><td>自动 (滑块 128)；步长 1；run 2；topmost</td><td>开启；面积 0.003；宽度 0.05；宽高比 8；Top N 5</td><td>强力网格；Y-band 关闭；5000</td></tr>
          <tr><td>瀑布 / 堆叠曲线</td><td>自动 (滑块 160)；步长 1；run 2；topmost</td><td>开启；面积 0.001；宽度 0.015；宽高比 12；Top N 12</td><td>网格关闭；Y-band 开启；5000</td></tr>
        </table>
        <h2>瀑布图</h2>
        <p>悬停图像读取像素 Y 值，把 Y min 与 Y max 缩小到一条曲线。每条曲线新建一个数据集并重复提取。上方或中间曲线使用 topmost；最下方曲线受其它线干扰时可使用 bottommost。若瀑布图也有网格，再手动开启强力网格抑制。</p>
        <h2>框选删除</h2>
        <p>框选只影响当前数据集。拖动矩形后，矩形内的像素点及其校准坐标会成对删除。Ctrl+Z 可一次恢复最近删除的整组点。</p>
        """
    return """
    <h1>PXRD Patent-Plot Extraction Instructions</h1>
    <h2>Common workflow</h2>
    <ol>
      <li>Open the image to detect the ROI automatically. Inspect the green box. Toggle <b>Auto-detect ROI on image load</b> to disable this, use <b>Auto-detect ROI</b> to retry, or draw a box manually to adjust the area.</li>
      <li>Calibrate X and Y independently from visible tick positions.</li>
      <li>Click <b>Find all curves, then choose</b>. Autopilot chooses a preset and detects curve candidates. Click a curve in the preview or filter by color, then add the selected curve or all shown curves. Each is added as a new dataset.</li>
      <li>Extract and inspect the colored points. Use <b>Reject Points in Box</b> to drag around false points.</li>
      <li>Press Ctrl+Z to restore the most recent box rejection, then export or add the trace as a reference.</li>
    </ol>
    <p>Autopilot supports colored, dark, and stacked line plots on light backgrounds and locates the axis frame or a ticked X-axis. It selects the dark threshold, grid suppression and upward-peak / downward-dip sampling direction. After adding curves, the selected threshold and pick method appear in the manual settings. Orange bands mark touching or unresolved strokes: check curve identity and missing curves. X coverage is not accuracy. Draw an ROI first if the inferred area is inaccurate. Use Blob / Scatter for scatter plots.</p>
    <h2>Manual preset parameters</h2>
    <table border="1" cellspacing="0" cellpadding="5">
      <tr><th>Preset</th><th>Threshold / sampling</th><th>Component filter</th><th>Grid / Y-band / max points</th></tr>
      <tr><td>Normal scan</td><td>Auto (slider 160); step 1; run 2; topmost</td><td>On; area 0.003; width 0.05; aspect 8; Top N 5</td><td>Grid off; Y-band off; 3000</td></tr>
      <tr><td>IR / downward dips</td><td>Auto (slider 160); step 1; run 2; bottommost</td><td>On; area 0.003; width 0.05; aspect 8; Top N 5</td><td>Grid off; Y-band off; 3000</td></tr>
      <tr><td>Grid / scanned patent</td><td>Auto (slider 128); step 1; run 2; topmost</td><td>On; area 0.003; width 0.05; aspect 8; Top N 5</td><td>Strong grid; Y-band off; 5000</td></tr>
      <tr><td>Waterfall / stacked traces</td><td>Auto (slider 160); step 1; run 2; topmost</td><td>On; area 0.001; width 0.015; aspect 12; Top N 12</td><td>Grid off; Y-band on; 5000</td></tr>
    </table>
    <h2>Waterfall scans</h2>
    <p>Hover over the image to read pixel Y values, then narrow Y min and Y max around one trace. Create a new dataset for each trace and repeat extraction. Use topmost for upper or middle traces; bottommost can help with the lowest trace. If the waterfall also has a grid, enable Strong suppression manually.</p>
    <h2>Box rejection</h2>
    <p>Only the active dataset is edited. Dragging a rectangle removes every point inside it, keeping pixel and calibrated coordinates paired. Ctrl+Z restores the entire most recently rejected group.</p>
    """


def export_xrdml(path, two_theta_arr, intensity_arr, dataset_label="Digitized",
                 wavelength_ka1=1.5405980, wavelength_ka2=1.5444260,
                 wavelength_kb=1.3922500,
                 generator_voltage=45.0, generator_current=40.0,
                 counting_time=1.0, y_scale=1.0, y_baseline=0.0,
                 comment="Digitized by PlotDigitizer"):
    """Write HighScore-compatible XRDML v2.2 matching real Empyrean output.
    counting_time: HighScore divides raw counts by this to display cps.
      Set to 1.0 if Y axis is already in cps (default).
      Set to actual dwell time T if Y axis is in raw counts.
    y_scale: multiplied into baseline-corrected intensities before writing.
    y_baseline: subtracted before scaling. Default export preserves the CSV
      intensity scale apart from integer count rounding.
    """
    from datetime import datetime
    import xml.etree.ElementTree as ET
    from xml.dom import minidom

    t = np.asarray(two_theta_arr, dtype=float)
    y = np.asarray(intensity_arr, dtype=float)
    order = np.argsort(t); t = t[order]; y = y[order]
    _, uniq = np.unique(t, return_index=True)
    t = t[uniq]; y = y[uniq]
    n = len(t)
    if n < 2:
        raise ValueError("Need at least 2 data points to write XRDML.")
    start2t = float(t[0])
    end2t   = float(t[-1])
    y = (y - float(y_baseline)) * float(y_scale)
    y_int = np.maximum(0, np.round(y).astype(int))

    NS  = "http://www.xrdml.com/XRDMeasurement/2.2"
    XSI = "http://www.w3.org/2001/XMLSchema-instance"
    LOC = NS + " " + NS + "/XRDMeasurement.xsd"
    now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S+00:00")

    def el(parent, tag, text=None, **attrs):
        e = ET.SubElement(parent, tag, **attrs)
        if text is not None:
            e.text = str(text)
        return e

    root = ET.Element("xrdMeasurements",
                      xmlns=NS,
                      **{"xmlns:xsi": XSI, "xsi:schemaLocation": LOC, "status": "Completed"})

    rc = el(root, "comment")
    el(rc, "entry", "Digitized dataset: " + dataset_label)
    el(rc, "entry", "Created by PlotDigitizer")

    sample = el(root, "sample", type="To be analyzed")
    el(sample, "id",         dataset_label[:60])
    el(sample, "name",       dataset_label[:60])
    el(sample, "preparedBy", "PlotDigitizer")

    meas = el(root, "xrdMeasurement",
              measurementType="Scan", status="Completed", sampleMode="Reflection")
    mc = el(meas, "comment"); el(mc, "entry", comment)

    wl = el(meas, "usedWavelength", intended="K-Alpha 1")
    el(wl, "kAlpha1", "%.7f" % wavelength_ka1, unit="Angstrom")
    el(wl, "kAlpha2", "%.7f" % wavelength_ka2, unit="Angstrom")
    el(wl, "kBeta",   "%.7f" % wavelength_kb,  unit="Angstrom")
    el(wl, "ratioKAlpha2KAlpha1", "0.5000")

    ibp = el(meas, "incidentBeamPath")
    el(ibp, "radius", "240.00", unit="mm")
    xrt = el(ibp, "xRayTube", id="0", name="Generic Cu tube")
    el(xrt, "tension",       "%.1f" % generator_voltage, unit="kV")
    el(xrt, "current",       "%.1f" % generator_current, unit="mA")
    el(xrt, "anodeMaterial", "Cu")
    foc = el(xrt, "focus", type="Line")
    el(foc, "length", "12.0", unit="mm")
    el(foc, "width",  "0.4",  unit="mm")
    el(foc, "takeOffAngle", "7.6", unit="deg")

    dbp = el(meas, "diffractedBeamPath")
    el(dbp, "radius", "240.00", unit="mm")
    # Declare a point-detector so HighScore skips PIXcel multi-channel
    # normalisation (which inflates digitized baselines by ~5x)
    det = el(dbp, "detector", id="0", name="Digitized (point)",
             **{"xsi:type": "pointDetectorType"})
    phd = el(det, "phd", unit="%")
    el(phd, "lowerLevel", "10.0")
    el(phd, "upperLevel", "90.0")

    scan = el(meas, "scan",
              appendNumber="0", mode="Continuous", scanAxis="Gonio", status="Completed")
    hdr = el(scan, "header")
    el(hdr, "startTimeStamp", now)
    el(hdr, "endTimeStamp",   now)
    author = el(hdr, "author"); el(author, "name", "PlotDigitizer")
    source = el(hdr, "source")
    el(source, "applicationSoftware",       "PlotDigitizer", version="1.0")
    el(source, "instrumentControlSoftware", "N/A",           version="N/A")
    el(source, "instrumentID", "0")

    dp = el(scan, "dataPoints")
    pos2t = el(dp, "positions", axis="2Theta", unit="deg")
    el(pos2t, "startPosition", "%.8f" % start2t)
    el(pos2t, "endPosition",   "%.8f" % end2t)
    posOm = el(dp, "positions", axis="Omega", unit="deg")
    el(posOm, "startPosition", "%.8f" % (start2t / 2.0))
    el(posOm, "endPosition",   "%.8f" % (end2t   / 2.0))

    ct_val = float(counting_time)
    el(dp, "commonCountingTime", "%.4f" % ct_val, unit="seconds")
    el(dp, "counts", " ".join(str(v) for v in y_int), unit="counts")

    raw    = ET.tostring(root, encoding="unicode", xml_declaration=False)
    pretty = minidom.parseString(raw).toprettyxml(indent="\t", encoding=None)
    lines  = pretty.split("\n")
    lines[0] = '<?xml version="1.0"?>'
    result = "\n".join(l for l in lines if l.strip() != "")
    with open(path, "w", encoding="utf-8-sig") as fh:
        fh.write(result + "\n")



# ══════════════════════════════════════════════════════════════════════════════
#  Canvas
# ══════════════════════════════════════════════════════════════════════════════

class PlotCanvas(FigureCanvas):
    calibration_click = Signal(float, float)
    datapoint_click   = Signal(float, float)
    roi_updated       = Signal(float, float, float, float)   # x0,y0,x1,y1
    points_rejected   = Signal(int, float, float, float, float)
    DS_COLORS = ["#1677b8", "#d97706", "#16803c", "#c0265e",
                 "#7c3fb2", "#b88a00", "#d14b25", "#087f8c"]

    def __init__(self, parent=None):
        self.fig = Figure(facecolor="#f4f5f7")
        self.ax  = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._layout_timer = QTimer(self)
        self._layout_timer.setSingleShot(True)
        self._layout_timer.timeout.connect(self._fit_resized_axes)
        self.image = None
        self.mode  = "none"

        # Separated axis calibration
        self._x_target = 3; self._y_target = 3
        self._x_calib_ready = False; self._y_calib_ready = False
        self._calib_ax=1.0; self._calib_bx=0.0
        self._calib_ay=1.0; self._calib_by=0.0
        self.x_calib_pts=[]; self.x_calib_vals=[]
        self.y_calib_pts=[]; self.y_calib_vals=[]
        self.scale_type = "linear"

        # ROI
        self.roi = None          # (x0,y0,x1,y1) in pixel coords or None
        self._roi_click1 = None  # first corner during ROI drawing
        self._reject_click1 = None
        self._drag_rect_artist = None

        self.datasets = [{"label":"Dataset 1","marker":"o","pts_px":[],"pts_real":[]}]
        self.active_ds = 0
        self._last_rejection = None
        self.last_undo_kind = ""
        self.last_undo_count = 0
        self.mpl_connect("button_press_event",  self._on_press)
        self.mpl_connect("motion_notify_event", self._on_motion)
        self.mpl_connect("button_release_event",self._on_release)
        self._style()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "_layout_timer"):
            self._layout_timer.start(0)

    def _fit_resized_axes(self):
        self.fig.tight_layout(pad=0.6)
        self.draw_idle()

    def _style(self):
        self.ax.set_facecolor("#ffffff")
        self.ax.tick_params(colors="#4b5563")
        for sp in self.ax.spines.values(): sp.set_color("#9ca3af")
        self.fig.tight_layout(pad=0.3)

    def load_image(self, path):
        raw = mpimg.imread(path)
        if raw.dtype != np.uint8:
            raw = (np.clip(raw,0,1)*255).astype(np.uint8)
        if raw.ndim==2: raw=np.stack([raw]*3,axis=-1)
        elif raw.shape[2]==4: raw=raw[:,:,:3]
        self.image = raw
        self.roi = None; self._roi_click1 = None; self._reject_click1 = None
        self.x_calib_pts=[]; self.x_calib_vals=[]
        self.y_calib_pts=[]; self.y_calib_vals=[]
        self._x_calib_ready=False; self._y_calib_ready=False
        for ds in self.datasets: ds["pts_px"]=[]; ds["pts_real"]=[]
        self.clear_rejection_undo()
        self._redraw(preserve_view=False)

    def fit_x_calib(self, pts_px, vals_real):
        if len(pts_px) != len(vals_real):
            raise ValueError("Each calibration point needs a numeric value.")
        points = [{"pixel": point, "value": value} for point, value in zip(pts_px, vals_real)]
        scale = "log" if self.scale_type in ("semilogx", "loglog") else "linear"
        self._calib_ax, self._calib_bx = calibration_coefficients(points, "x", scale)
        self._x_calib_ready = True

    def fit_y_calib(self, pts_px, vals_real):
        if len(pts_px) != len(vals_real):
            raise ValueError("Each calibration point needs a numeric value.")
        points = [{"pixel": point, "value": value} for point, value in zip(pts_px, vals_real)]
        scale = "log" if self.scale_type in ("semilogy", "loglog") else "linear"
        self._calib_ay, self._calib_by = calibration_coefficients(points, "y", scale)
        self._y_calib_ready = True

    def _px2data(self, px, py):
        if not (self._x_calib_ready and self._y_calib_ready): return None,None
        xr = self._calib_ax*px + self._calib_bx
        yr = self._calib_ay*py + self._calib_by
        x = 10**xr if self.scale_type in ("semilogx", "loglog") else xr
        y = 10**yr if self.scale_type in ("semilogy","loglog") else yr
        # Scientific axes can legitimately contain values far below 1e-6.
        # Keep numeric precision here; apply display formatting only in the UI.
        return float(x), float(y)

    def remove_last_point(self):
        ds = self.datasets[self.active_ds]
        if (
            self._last_rejection is not None
            and self._last_rejection["dataset"] is ds
        ):
            removed = self._last_rejection["removed"]
            for index, pixel, real in removed:
                insert_at = min(index, len(ds["pts_px"]))
                ds["pts_px"].insert(insert_at, pixel)
                ds["pts_real"].insert(insert_at, real)
            self._last_rejection = None
            self.last_undo_kind = "box_rejection"
            self.last_undo_count = len(removed)
            self._redraw()
            return True
        self._last_rejection = None
        if ds["pts_px"]:
            ds["pts_px"].pop(); ds["pts_real"].pop()
            self.last_undo_kind = "point"
            self.last_undo_count = 1
            self._redraw()
            return True
        return False

    def clear_rejection_undo(self):
        self._last_rejection = None
        self.last_undo_kind = ""
        self.last_undo_count = 0

    def reject_active_points_in_box(self, box):
        ds = self.datasets[self.active_ds]
        kept_px, kept_real, removed = reject_points_in_box(
            ds["pts_px"], ds["pts_real"], box
        )
        if not removed:
            return 0
        ds["pts_px"] = kept_px
        ds["pts_real"] = kept_real
        self._last_rejection = {"dataset": ds, "removed": removed}
        self.last_undo_kind = ""
        self.last_undo_count = 0
        self._redraw()
        return len(removed)

    def bulk_add_pixels(self, pts_px):
        ds = self.datasets[self.active_ds]; added=0
        for px,py in pts_px:
            x,y = self._px2data(px,py)
            if x is not None:
                ds["pts_px"].append([px,py]); ds["pts_real"].append([x,y]); added+=1
        if added:
            self.clear_rejection_undo()
            self._redraw()
        return added

    def _on_press(self, event):
        if event.inaxes != self.ax or self.image is None or event.button != 1: return
        px,py = event.xdata, event.ydata
        if self.mode == "roi":
            self._roi_click1 = (px,py)
            self._begin_box_drag(px, py, "#15803d")
        elif self.mode == "reject":
            self._reject_click1 = (px,py)
            self._begin_box_drag(px, py, "#b91c1c")
        elif self.mode == "calibrate_x" and len(self.x_calib_pts) < self._x_target:
            self.x_calib_pts.append([px,py]); self.calibration_click.emit(px,py); self._redraw()
        elif self.mode == "calibrate_y" and len(self.y_calib_pts) < self._y_target:
            self.y_calib_pts.append([px,py]); self.calibration_click.emit(px,py); self._redraw()
        elif self.mode == "digitize":
            x,y = self._px2data(px,py)
            if x is None: return
            ds = self.datasets[self.active_ds]
            self.clear_rejection_undo()
            ds["pts_px"].append([px,py]); ds["pts_real"].append([x,y])
            self.datapoint_click.emit(x,y); self._redraw()

    def _begin_box_drag(self, px, py, color):
        if self._drag_rect_artist is not None:
            self._drag_rect_artist.remove()
        self._drag_rect_artist = mpatches.Rectangle(
            (px, py), 0, 0,
            linewidth=1.8,
            edgecolor=color,
            facecolor=color,
            alpha=0.16,
            linestyle="--",
            zorder=20,
        )
        self.ax.add_patch(self._drag_rect_artist)
        self.draw_idle()

    def _on_motion(self, event):
        if (
            self.mode not in ("roi", "reject")
            or self._drag_rect_artist is None
            or event.inaxes != self.ax
            or event.xdata is None
            or event.ydata is None
        ):
            return
        click1 = self._roi_click1 if self.mode == "roi" else self._reject_click1
        if click1 is None:
            return
        x0, y0 = click1
        self._drag_rect_artist.set_xy((min(x0, event.xdata), min(y0, event.ydata)))
        self._drag_rect_artist.set_width(abs(event.xdata - x0))
        self._drag_rect_artist.set_height(abs(event.ydata - y0))
        self.draw_idle()

    def _on_release(self, event):
        if self.mode not in ("roi", "reject"): return
        if self.image is None: return
        click1 = self._roi_click1 if self.mode == "roi" else self._reject_click1
        if click1 is None: return
        x0,y0 = click1
        if event.xdata is None or event.ydata is None:
            x1, y1 = self.ax.transData.inverted().transform((event.x, event.y))
        else:
            x1, y1 = event.xdata, event.ydata
        height, width = self.image.shape[:2]
        x1 = max(0.0, min(float(width), float(x1)))
        y1 = max(0.0, min(float(height), float(y1)))
        box = (min(x0,x1), min(y0,y1), max(x0,x1), max(y0,y1))
        mode = self.mode
        self.mode = "none"
        if mode == "roi":
            self.roi = box
            self._roi_click1 = None
            self.roi_updated.emit(*self.roi)
            self._redraw()
        else:
            self._reject_click1 = None
            count = self.reject_active_points_in_box(box)
            if not count:
                self._redraw()
            self.points_rejected.emit(count, *box)

    def _redraw(self, preserve_view=True):
        view = None
        if preserve_view and self.image is not None and self.ax.images:
            view = (self.ax.get_xlim(), self.ax.get_ylim())
        self._drag_rect_artist = None
        self.ax.cla(); self._style()
        if self.image is None: self.draw(); return
        h,w = self.image.shape[:2]
        self.ax.imshow(self.image, extent=[0,w,h,0], aspect="auto", interpolation="bilinear")
        self.ax.set_xlim(0,w); self.ax.set_ylim(h,0)
        # ROI rectangle
        if self.roi is not None:
            rx0,ry0,rx1,ry1 = self.roi
            rect = mpatches.Rectangle((rx0,ry0), rx1-rx0, ry1-ry0,
                linewidth=1.5, edgecolor="#15803d", facecolor="none",
                linestyle="--", zorder=8)
            self.ax.add_patch(rect)
            self.ax.annotate("ROI",xy=(rx0,ry0),xytext=(4,4),
                             textcoords="offset points",color="#15803d",fontsize=8,zorder=9)
        # X calib points — orange ▼
        for i,pt in enumerate(self.x_calib_pts):
            self.ax.plot(pt[0],pt[1],"v",color="#d97706",markersize=11,
                         markeredgecolor="white",markeredgewidth=0.8,zorder=10)
            self.ax.annotate(f"X{i+1}",xy=(pt[0],pt[1]),xytext=(5,4),
                             textcoords="offset points",color="#b45309",fontsize=8,fontweight="bold",zorder=11)
        # Y calib points — cyan ▶
        for i,pt in enumerate(self.y_calib_pts):
            self.ax.plot(pt[0],pt[1],">",color="#0891b2",markersize=11,
                         markeredgecolor="white",markeredgewidth=0.8,zorder=10)
            self.ax.annotate(f"Y{i+1}",xy=(pt[0],pt[1]),xytext=(5,4),
                             textcoords="offset points",color="#0e7490",fontsize=8,fontweight="bold",zorder=11)
        # Dataset points
        for idx,ds in enumerate(self.datasets):
            if not ds["pts_px"]: continue
            xpx=[p[0] for p in ds["pts_px"]]; ypx=[p[1] for p in ds["pts_px"]]
            col=self.DS_COLORS[idx%len(self.DS_COLORS)]
            self.ax.scatter(xpx,ypx,color=col,marker=ds["marker"],
                            s=20,edgecolors="white",linewidths=0.3,zorder=9,label=ds["label"])
        if any(ds["pts_px"] for ds in self.datasets):
            self.ax.legend(loc="lower right", facecolor="white",
                           edgecolor="#9ca3af", labelcolor="#1f2937", fontsize=8)
        # Y=0 guide line — verify calibration zero is correctly placed
        if self._y_calib_ready and self.image is not None:
            if abs(self._calib_ay) > 1e-12:
                py_zero = -self._calib_by / self._calib_ay
                img_h = self.image.shape[0]
                if 0 <= py_zero <= img_h:
                    self.ax.axhline(py_zero, color="#15803d", linewidth=1.0,
                                    linestyle="--", alpha=0.75, zorder=8)
                    self.ax.text(4, py_zero - 4, "Y=0 (calib)",
                                color="#15803d", fontsize=8, zorder=9)
        if view is not None:
            self.ax.set_xlim(*view[0])
            self.ax.set_ylim(*view[1])
        self.fig.tight_layout(pad=0.3); self.draw()


# ══════════════════════════════════════════════════════════════════════════════
#  Color swatch
# ══════════════════════════════════════════════════════════════════════════════

class ColorSwatch(QPushButton):
    def __init__(self, rgb=(220,50,50)):
        super().__init__()
        self.setAccessibleName("Curve color")
        self.set_color(rgb)
        self._update_size()

    def _update_size(self):
        side = max(32, self.fontMetrics().height() + 14)
        self.setFixedSize(side, side)

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() == QEvent.FontChange:
            self._update_size()

    def set_color(self, rgb):
        self._rgb = rgb
        self.setStyleSheet(
            f"QPushButton {{ background: rgb{rgb}; border: 2px solid palette(text); "
            "border-radius: 4px; padding: 0; min-height: 0; } "
            "QPushButton:focus { border: 3px dashed palette(highlight); }"
        )

    def get_rgb(self):
        return self._rgb


# ══════════════════════════════════════════════════════════════════════════════
#  Main Window
# ══════════════════════════════════════════════════════════════════════════════

ZH_LABELS = {
    "Autopilot": "自动驾驶",
    "Find all curves, then choose": "检测所有曲线后选择",
    "Open an image and calibrate the axes, then let Autopilot choose a preset and find the curves. Optional: draw an ROI first.": "打开图像并校准坐标轴，然后让自动驾驶选择预设并检测曲线。也可先绘制 ROI。",
    "Autopilot — choose curves": "自动驾驶 — 选择曲线",
    "Detecting curves and choosing a preset…": "正在检测曲线并选择预设…",
    "What color is the desired curve?": "目标曲线是什么颜色？",
    "All colors": "所有颜色",
    "Click a curve on the preview or choose a row below.": "在预览中点击曲线，或选择下方对应行。",
    "Curve": "曲线",
    "Color": "颜色",
    "X coverage": "X 覆盖率",
    "Cancel": "取消",
    "Add selected curve": "添加所选曲线",
    "Add all shown curves": "添加所有显示的曲线",
    "Detection failed: ": "检测失败：",
    "No curves found. Draw a tighter ROI, then retry Autopilot or use Color Pick in Auto Extract.": "未找到曲线。请缩小 ROI 后重试，或使用自动提取中的颜色选取。",
    "{count} curves detected · Preset: {preset}. Check the highlighted points before adding. Touching curves of the same color may need manual extraction.": "检测到 {count} 条曲线 · 预设：{preset}。添加前请检查高亮点。同色且相接的曲线可能需要手动提取。",
    "Curve {number}": "曲线 {number}",
    "Stopping curve detection…": "正在停止曲线检测…",
    "Black / gray": "黑色 / 灰色",
    "Red": "红色",
    "Orange": "橙色",
    "Yellow": "黄色",
    "Green": "绿色",
    "Cyan": "青色",
    "Blue": "蓝色",
    "Purple": "紫色",
    "Pink": "粉色",
    "Autopilot added {count} curves as new datasets · Preset: {preset}": "自动驾驶已添加 {count} 条曲线为新数据集 · 预设：{preset}",
    "Could not add curves": "无法添加曲线",
    "Controls": "操作",
    "Image & Data": "图像与数据",
    "&File": "文件(&F)",
    "&Edit": "编辑(&E)",
    "&Help": "帮助(&H)",
    "&Language": "语言(&L)",
    "Open Image…": "打开图像…",
    "Export CSV…": "导出 CSV…",
    "Export JSON…": "导出 JSON…",
    "Export XRDML…": "导出 XRDML…",
    "Quit": "退出",
    "Undo Last Point": "撤销上一个点",
    "Clear Active Dataset": "清空当前数据集",
    "Reset Calibration": "重置校准",
    "Clear ROI": "清除 ROI",
    "About": "关于",
    "English": "English",
    "Chinese": "中文",
    "📂  Open Image…": "📂  打开图像…",
    "🟩  Draw ROI Box on Image": "🟩  在图像上绘制 ROI 框",
    "Auto-detect ROI": "自动检测 ROI",
    "Auto-detect ROI on image load": "打开图像时自动检测 ROI",
    "Detect the plot area automatically, or draw a box inside the axes to exclude labels, borders, and margins.": "自动检测绘图区域，或在坐标轴内侧绘制框，以排除标签、边框和边距。",
    "Auto ROI set. Check the green box; use Draw ROI to adjust it.": "自动 ROI 已设置。请检查绿色框，可通过绘制 ROI 调整。",
    "No clear plot frame found. ROI unchanged; draw a box manually.": "未找到清晰的绘图边框。ROI 保持不变，请手动绘制。",
    "✕  Clear ROI": "✕  清除 ROI",
    "🖱  Click X Ticks on Image": "🖱  点击图像上的 X 轴刻度",
    "🖱  Click Y Ticks on Image": "🖱  点击图像上的 Y 轴刻度",
    "↺  Reset Calibration": "↺  重置校准",
    "＋  New Dataset": "＋  新建数据集",
    "✖  Delete Dataset": "✖  删除数据集",
    "▭  Reject Points in Box": "▭  框选拒绝点",
    "Reject Points in Box": "框选拒绝点",
    "🖱  Start Digitizing": "🖱  开始数字化",
    "↩  Undo Last  (Ctrl+Z)": "↩  撤销上一个点  (Ctrl+Z)",
    "🗑  Clear Dataset": "🗑  清空数据集",
    "⚡  Extract Dark Curve": "⚡  提取暗色曲线",
    "🎯 Eyedrop": "🎯 取色器",
    "⚡  Extract by Color": "⚡  按颜色提取",
    "⚡  Detect Blobs": "⚡  检测斑点",
    "💾  Export CSV": "💾  导出 CSV",
    "💾  Export JSON": "💾  导出 JSON",
    "📡  Export XRDML": "📡  导出 XRDML",
    "🎯  Auto-detect baseline from data": "🎯  从数据自动检测基线",
    "📏  Y-band (restrict rows for waterfall curves)": "📏  Y 带 (限制瀑布图曲线行)",
    "🔗  Connectivity filter  (removes labels/text)": "🔗  连通性过滤 (移除标签/文字)",
    "Suppress long grid / axis lines": "抑制长网格线 / 坐标轴线",
    "Grid suppression:": "网格线抑制:",
    "Standard (straight lines)": "标准 (直线)",
    "Strong (wavy / scanned grids)": "强力 (波浪 / 扫描网格)",
    "Strong mode detects long line bands in noisy patent scans and removes their signals before tracing.": "强力模式可检测专利扫描图中的长线带，并在跟踪曲线前移除其信号。",
    "Extraction Presets": "提取预设",
    "Extraction preset:": "提取预设:",
    "Normal scan": "普通扫描",
    "Auto-select darkness threshold (128–235)": "自动选择暗色阈值 (128–235)",
    "Grid / scanned patent": "网格 / 专利扫描",
    "Waterfall / stacked traces": "瀑布 / 堆叠曲线",
    "IR / downward dips": "红外 / 向下吸收谷",
    "Single transmittance trace with downward absorption dips.": "带有向下吸收谷的单条透射率曲线。",
    "{count} curve candidates · Preset: {preset} · Dark threshold: {threshold}.": "{count} 条候选曲线 · 预设：{preset} · 暗色阈值：{threshold}。",
    "Review orange bands: curves may touch, merge or remain undetected.": "请检查橙色区域：曲线可能接触、合并或未被完整检测。",
    "Check the highlighted points before adding; X coverage is not accuracy.": "添加前请检查高亮数据点；X 覆盖率不代表提取精度。",
    "Curve count changes with contrast; check for missing curves.": "曲线数量随对比度变化；请检查是否遗漏曲线。",
    "Separation": "分离情况",
    "Needs review": "需要检查",
    "Partial trace": "部分曲线",
    "No overlap found": "未发现重叠",
    "Clean single trace without a long grid.": "无长网格的干净单条曲线。",
    "Single trace with long straight, broken, or wavy grid lines.": "带有长直线、断线或波浪网格的单条曲线。",
    "One trace at a time using a manually adjusted pixel Y-band.": "使用手动调整的像素 Y-band 逐条提取曲线。",
    "Apply Preset": "应用预设",
    "Instructions / Print…": "说明 / 打印…",
    "Extraction Presets & Instructions…": "提取预设与说明…",
    "Print…": "打印…",
    "Close": "关闭",
    "⚙  XRDML options (wavelength, source)": "⚙  XRDML 选项 (波长、光源)",
}

ZH_LABELS.update({
    "PlotDigitizer — Separated Axis Calibration + ROI": "PlotDigitizer — 独立坐标轴校准 + ROI",
    "Image": "图像",
    "Axis Scale": "坐标轴比例",
    "Linear": "线性",
    "Semi-log Y": "半对数 Y",
    "Semi-log X": "半对数 X",
    "Log-Log": "双对数",
    "Logarithmic": "对数",
    "Suggest axis calibration…": "建议坐标轴校准…",
    "Review automatic axis calibration": "检查自动坐标轴校准",
    "Reading numeric labels and matching axis ticks locally…": "正在本机识别数值标签并匹配坐标轴刻度…",
    "Review each value and tick. Edit cells, uncheck rejected labels, or select a row and click its tick on the image.": "请检查数值及对应刻度。可编辑单元格、取消不合适的标签，或选中一行后点击图中对应刻度。",
    "Add calibration point": "添加校准点",
    "Use": "使用",
    "Pixel X": "像素 X",
    "Pixel Y": "像素 Y",
    "Value": "数值",
    "Read label": "识别标签",
    "Apply reviewed calibration": "应用已检查的校准",
    "Axis detection failed. Existing calibration is unchanged.": "坐标轴检测失败。现有校准保持不变。",
    "Suggested {x} X and {y} Y points. Confirm the scale, numeric labels and exact tick positions before applying. Missing labels are not inferred.": "建议 {x} 个 X 点和 {y} 个 Y 点。应用前请确认坐标尺度、数值和确切的刻度位置。不会推断缺失的标签。",
    "Enter numeric pixel coordinates and a value for every checked row.": "请为每个勾选行输入数值型像素坐标及刻度数值。",
    "Calibration points must be finite and inside the image.": "校准点必须是有限数值且位于图像内。",
    "Choose at most 15 calibration points per axis.": "每个轴最多选择 15 个校准点。",
    "Stopping axis detection…": "正在停止坐标轴检测…",
    "X calibrated ✓": "X 已校准 ✓",
    "Y calibrated ✓": "Y 已校准 ✓",
    "Reviewed axis calibration applied.": "已应用检查后的坐标轴校准。",
    "Calibration Error": "校准错误",
    "Some labels disagree with one axis scale. Inspect unchecked rows; broken or nonuniform axes need separate plot regions.": "部分标签无法匹配同一坐标尺度。请检查未勾选行；断轴或非均匀坐标需按区域分别校准。",
    "Step 0 — Plot Area (ROI)  [recommended]": "步骤 0 — 绘图区域 (ROI)  [推荐]",
    "Restrict extraction to the plot area only.\nDraw a box just inside the axis lines to\nexclude tick labels, borders, and margins.": "仅在绘图区域内提取。\n请在坐标轴内侧绘制框，\n以排除刻度标签、边框和边距。",
    "No ROI — full image used": "未设置 ROI — 使用整张图像",
    "Step 1 — Calibrate Axes": "步骤 1 — 校准坐标轴",
    "X and Y are calibrated independently.\nNo need to use the origin — click any\nvisible tick marks on each axis.": "X 和 Y 轴将分别校准。\n不需要使用原点 — 点击每个轴上\n任意可见的刻度即可。",
    "X Axis  (orange ▼ markers)": "X 轴  (橙色 ▼ 标记)",
    "# X points:": "X 点数量:",
    "Label": "标签",
    "Real X": "真实 X",
    "○ ○ ○  awaiting X points": "○ ○ ○  等待 X 点",
    "Y Axis  (cyan ▶ markers)": "Y 轴  (青色 ▶ 标记)",
    "# Y points:": "Y 点数量:",
    "Real Y": "真实 Y",
    "○ ○ ○  awaiting Y points": "○ ○ ○  等待 Y 点",
    "⬤ X: not set     ⬤ Y: not set": "⬤ X: 未设置     ⬤ Y: 未设置",
    "⬤ X: not set": "⬤ X: 未设置",
    "Dataset:": "数据集:",
    "Dataset 1": "数据集 1",
    "✋ Manual": "✋ 手动",
    "Into dataset:": "写入数据集:",
    "Mode 1 — Dark Curve  ★ (black on white)": "模式 1 — 暗色曲线  ★ (白底黑线)",
    "Best for black/dark lines on light background.\n(e.g. patent figures, scanned PXRD)": "适合浅色背景上的黑色/深色线条。\n(例如专利图、扫描 PXRD)",
    "Darkness threshold (0=black, 255=white):": "暗度阈值 (0=黑，255=白):",
    "Column step (1=densest):": "列步长 (1=最密):",
    "Min run length (noise filter, no-conn only):": "最小连续长度 (噪声过滤，仅无连通性时):",
    "Pick:": "选取:",
    "topmost  (upper / single curve)": "最上方  (上方/单条曲线)",
    "bottommost  (lower curve in waterfall)": "最下方  (瀑布图下方曲线)",
    "center  (thick strokes)": "中心  (粗线条)",
    "Y-band (pixel rows in image)": "Y 带 (图像像素行)",
    "Limit extraction to rows Y_min...Y_max.\nHover image to read pixel Y values.\n  topmost+Y-band   -> upper/mid curve\n  bottommost+Y-band -> lower/mid curve": "将提取限制在 Y_min...Y_max 行。\n悬停图像可读取像素 Y 值。\n  最上方+Y 带   -> 上方/中间曲线\n  最下方+Y 带 -> 下方/中间曲线",
    "Y min (px):": "Y 最小值 (px):",
    "Y max (px):": "Y 最大值 (px):",
    "Connectivity parameters": "连通性参数",
    "Min component area (×total):": "最小连通区域面积 (×总面积):",
    "Min width (×image width):": "最小宽度 (×图像宽度):",
    "Max aspect ratio (w/h) to reject:": "拒绝的最大宽高比 (w/h):",
    "Keep top N components:": "保留前 N 个连通区域:",
    "Mode 2 — Color Pick  (colored curves)": "模式 2 — 颜色选取  (彩色曲线)",
    "Hue tolerance:": "色相容差:",
    "Column step:": "列步长:",
    "Smoothing band (px):": "平滑带宽 (px):",
    "Min blob area (px²):": "最小斑点面积 (px²):",
    "center  (thick/dotted)": "中心  (粗线/点线)",
    "Limit extraction to rows Y_min...Y_max.\nHover over the image to read pixel Y values.": "将提取限制在 Y_min...Y_max 行。\n悬停图像可读取像素 Y 值。",
    "Max aspect (w/h) to reject:": "拒绝的最大宽高比 (w/h):",
    "Mode 3 — Blob / Scatter": "模式 3 — 斑点 / 散点",
    "Detects circular scatter-dot markers.": "检测圆形散点标记。",
    "Min area:": "最小面积:",
    "Max:": "最大:",
    "Min circularity:": "最小圆度:",
    "Max points:": "最大点数:",
    "⚡ Auto Extract": "⚡ 自动提取",
    "Export": "导出",
    "Export active dataset as PANalytical XRDML\nfor direct import into HighScore / HighScore Plus": "将当前数据集导出为 PANalytical XRDML\n可直接导入 HighScore / HighScore Plus",
    "XRDML Parameters": "XRDML 参数",
    "Ka1 (Å):": "Ka1 (Å):",
    "Ka2 (Å):": "Ka2 (Å):",
    "Voltage (kV):": "电压 (kV):",
    "Current (mA):": "电流 (mA):",
    "Counting time (s):": "计数时间 (s):",
    "Set to 1.0 if your Y axis is already in CPS (default).\nSet to actual scan dwell time (s) if Y axis is raw counts.": "如果 Y 轴已经是 CPS，请设为 1.0 (默认)。\n如果 Y 轴是原始计数，请设为实际扫描驻留时间 (s)。",
    "Y-scale factor:": "Y 缩放系数:",
    "Multiplied into all intensities before writing.\nIf HighScore shows 5x too high, enter 0.2 (=1/5).": "写入前乘到所有强度值上。\n如果 HighScore 显示高出 5 倍，请输入 0.2 (=1/5)。",
    "Y baseline offset (subtract):": "Y 基线偏移 (减去):",
    "Subtracted from ALL data values before writing.\nUse this when the digitized background is too high\ndue to Y=0 calibration point being placed on the\naxis line instead of the true data zero.\nEnter the inflated background value here to zero it.": "写入前从所有数据值中减去。\n当数字化背景过高时使用，\n通常是因为 Y=0 校准点落在坐标轴线上，\n而不是真实的数据零点。\n在此输入抬高的背景值以将其归零。",
    "Sets the Y baseline offset to the 5th-percentile\nintensity of the active dataset — a robust estimate\nof the background floor.": "将 Y 基线偏移设为当前数据集强度的第 5 百分位，\n这是对背景底部的稳健估计。",
    "Digitized Data": "数字化数据",
    "Dataset": "数据集",
    "X": "X",
    "Y": "Y",
    "Ready — File > Open Image to begin": "就绪 — 请从 文件 > 打开图像 开始",
    "✓ X calibrated     ✓ Y calibrated — ready!": "✓ X 已校准     ✓ Y 已校准 — 准备就绪!",
    "ROI — click and drag to draw the plot area box, then release": "ROI — 单击并拖动绘制绘图区域框，然后松开",
    "ROI cleared": "ROI 已清除",
    "Calibration reset": "校准已重置",
    "Last point removed": "已移除上一个点",
    "Open Plot Image": "打开绘图图像",
    "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.gif);;All (*)": "图像 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.gif);;所有文件 (*)",
    "Scale: ": "坐标轴比例: ",
})


class CurveDetectionWorker(QThread):
    result_ready = Signal(object)
    failed = Signal(str)

    def __init__(self, image, roi, parent=None):
        super().__init__(parent)
        self.image = image.copy()
        self.roi = roi

    def run(self):
        try:
            result = extract_plot_curves(self.image, self.roi, self.isInterruptionRequested)
            if not self.isInterruptionRequested():
                self.result_ready.emit(result)
        except InterruptedError:
            pass
        except Exception as error:
            self.failed.emit(str(error))


class AxisCalibrationWorker(QThread):
    result_ready = Signal(object)
    failed = Signal(str)

    def __init__(self, image, roi, parent=None):
        super().__init__(parent)
        self.image, self.roi = image, roi

    def run(self):
        try:
            roi = self.roi or detect_plot_roi(self.image, cancelled=self.isInterruptionRequested)
            if roi is None:
                raise ValueError("Detect or draw the plot ROI before reading axis labels.")
            result = suggest_axis_calibration(self.image, roi, self.isInterruptionRequested)
            if not self.isInterruptionRequested():
                self.result_ready.emit(result)
        except InterruptedError:
            pass
        except Exception as error:
            self.failed.emit(str(error))


class AxisCalibrationDialog(AdaptiveDialog):
    """Edit local OCR/tick proposals on the image before changing calibration."""

    def __init__(self, parent):
        super().__init__(parent)
        self._tr = parent._tr
        self.image = parent.canvas.image
        self.result_data = None
        self.reviewed = None
        self._closing = False
        self._updating = False
        self.setWindowTitle(self._tr("Review automatic axis calibration"))
        self.resize(1000, 820)
        layout = QVBoxLayout(self)
        self.summary = QLabel(self._tr("Reading numeric labels and matching axis ticks locally…"))
        self.summary.setWordWrap(True)
        self.summary.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        layout.addWidget(self.summary)
        self.figure = Figure(facecolor="#f4f5f7")
        self.preview = FigureCanvas(self.figure)
        self.preview.setMinimumSize(240, 190)
        self.axes = self.figure.add_subplot(111)
        self.preview.mpl_connect("button_press_event", self._set_tick_on_image)
        layout.addWidget(self.preview, 1)
        hint = QLabel(self._tr("Review each value and tick. Edit cells, uncheck rejected labels, or select a row and click its tick on the image."))
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.tabs = QTabWidget()
        self.tables, self.scales = {}, {}
        for axis in ("x", "y"):
            page = QWidget(); body = QVBoxLayout(page)
            row = QHBoxLayout()
            row.addWidget(QLabel(self._tr("Axis Scale")))
            scale = QComboBox(); scale.addItem(self._tr("Linear"), "linear"); scale.addItem(self._tr("Logarithmic"), "log")
            row.addWidget(scale)
            add = QPushButton(self._tr("Add calibration point"))
            add.clicked.connect(lambda _checked=False, key=axis: self._add_point(key))
            row.addWidget(add); body.addLayout(row)
            table = QTableWidget(0, 5)
            table.setHorizontalHeaderLabels([self._tr(text) for text in ("Use", "Pixel X", "Pixel Y", "Value", "Read label")])
            table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            table.setSelectionBehavior(QAbstractItemView.SelectRows)
            table.setMinimumHeight(115); table.setMaximumHeight(180)
            table.itemChanged.connect(lambda _item: self._preview_points())
            table.currentCellChanged.connect(lambda *_args: self._preview_points())
            body.addWidget(table)
            self.tables[axis], self.scales[axis] = table, scale
            self.tabs.addTab(page, axis.upper())
        self.tabs.currentChanged.connect(lambda _index: self._preview_points())
        self.tabs.setEnabled(False)
        layout.addWidget(self.tabs)
        self.error = QLabel(); self.error.setWordWrap(True); layout.addWidget(self.error)
        self.progress = QProgressBar(); self.progress.setRange(0, 0); layout.addWidget(self.progress)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Cancel)
        self.buttons.button(QDialogButtonBox.Cancel).setText(self._tr("Cancel"))
        self.apply_button = self.buttons.addButton(self._tr("Apply reviewed calibration"), QDialogButtonBox.ActionRole)
        self.apply_button.setObjectName("primary"); self.apply_button.setEnabled(False)
        self.apply_button.clicked.connect(self._accept_review)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self._preview_points()
        apply_window_appearance(self)
        self.worker = AxisCalibrationWorker(self.image, parent.canvas.roi, self)
        self.worker.result_ready.connect(self._receive_result)
        self.worker.failed.connect(lambda message: self.error.setText(message))
        self.worker.finished.connect(self._detection_finished)
        self.worker.start()
        fit_to_screen(self)

    def _receive_result(self, result):
        self.result_data = result

    def _fit_summary(self):
        if hasattr(self, "summary"):
            bounds = self.summary.fontMetrics().boundingRect(
                0, 0, max(240, self.width() - 48), 10000, Qt.TextWordWrap, self.summary.text())
            self.summary.setFixedHeight(bounds.height() + 8)
            # Refresh the height explicitly: Qt's cached height-for-width can
            # retain the one-line loading height inside the scrolling body.
            policy = self.summary.sizePolicy()
            policy.setHeightForWidth(False)
            self.summary.setSizePolicy(policy)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit_summary()

    def _detection_finished(self):
        self.progress.hide()
        if self._closing:
            super().reject(); return
        if self.result_data is None:
            self.summary.setText(self._tr("Axis detection failed. Existing calibration is unchanged."))
            return
        self.tabs.setEnabled(True)
        self._updating = True
        for axis in ("x", "y"):
            result = self.result_data["axes"][axis]
            self.scales[axis].setCurrentIndex(self.scales[axis].findData(result["scale"]))
            accepted = result["points"]
            selected = (set(np.linspace(0, len(accepted) - 1, min(15, len(accepted)), dtype=int))
                        if accepted and not result.get("scale_conflict") else set())
            for index, point in enumerate(accepted + result.get("rejected", [])):
                self._append_row(axis, point, index in selected)
        self._updating = False
        nx, ny = (len(self.result_data["axes"][axis]["points"]) for axis in ("x", "y"))
        summary = self._tr("Suggested {x} X and {y} Y points. Confirm the scale, numeric labels and exact tick positions before applying. Missing labels are not inferred.").format(x=nx, y=ny)
        if any(self.result_data["axes"][axis].get("scale_conflict") for axis in ("x", "y")):
            summary += "\n" + self._tr("Some labels disagree with one axis scale. Inspect unchecked rows; broken or nonuniform axes need separate plot regions.")
        self.summary.setText(summary)
        self._fit_summary()
        self.apply_button.setEnabled(True)
        self._preview_points()

    def _append_row(self, axis, point, checked=True):
        table = self.tables[axis]; row = table.rowCount(); table.insertRow(row)
        use = QTableWidgetItem()
        use.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsUserCheckable)
        use.setCheckState(Qt.Checked if checked else Qt.Unchecked); table.setItem(row, 0, use)
        x, y = point["pixel"]
        for column, value in enumerate((f"{x:.2f}", f"{y:.2f}", str(point["value"]), point.get("text", "")), 1):
            item = QTableWidgetItem(value)
            if column == 4:
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            table.setItem(row, column, item)

    def _add_point(self, axis):
        self._updating = True
        self._append_row(axis, {"pixel": [0, 0], "value": "", "text": ""})
        self.tables[axis].selectRow(self.tables[axis].rowCount() - 1)
        self._updating = False
        self._preview_points()

    def _points(self, axis):
        table = self.tables[axis]; points = []
        for row in range(table.rowCount()):
            if table.item(row, 0).checkState() != Qt.Checked:
                continue
            try:
                x, y, value = (float(table.item(row, col).text().strip()) for col in (1, 2, 3))
            except (ValueError, AttributeError):
                raise ValueError(self._tr("Enter numeric pixel coordinates and a value for every checked row."))
            if not np.all(np.isfinite([x, y, value])) or not (0 <= x < self.image.shape[1] and 0 <= y < self.image.shape[0]):
                raise ValueError(self._tr("Calibration points must be finite and inside the image."))
            points.append({"pixel": [x, y], "value": value})
        return points

    def _preview_points(self):
        if self._updating:
            return
        self.axes.clear(); self.axes.imshow(self.image); self.axes.set_axis_off()
        for axis, color in (("x", "#e17915"), ("y", "#009bb5")):
            table = self.tables[axis]
            for row in range(table.rowCount()):
                try:
                    if table.item(row, 0).checkState() != Qt.Checked:
                        continue
                    x, y = (float(table.item(row, col).text()) for col in (1, 2))
                    text = table.item(row, 3).text()
                except (ValueError, AttributeError):
                    continue
                if not np.all(np.isfinite([x, y])):
                    continue
                selected = table.currentRow() == row and self.tabs.currentIndex() == (0 if axis == "x" else 1)
                self.axes.scatter([x], [y], color=color, marker="+", s=130 if selected else 65, linewidths=2, zorder=4)
                self.axes.annotate(f"{axis.upper()}{row + 1}: {text}", (x, y), xytext=(4, -14 if axis == "x" else 4),
                                   textcoords="offset points", fontsize=8, color=color,
                                   bbox={"facecolor": "white", "alpha": .75, "edgecolor": "none", "pad": 1})
        self.figure.tight_layout(pad=.4); self.preview.draw_idle()

    def _set_tick_on_image(self, event):
        if event.inaxes != self.axes or event.button != 1:
            return
        axis = "x" if self.tabs.currentIndex() == 0 else "y"
        table = self.tables[axis]; row = table.currentRow()
        if row < 0:
            return
        self._updating = True
        table.item(row, 1).setText(f"{event.xdata:.2f}")
        table.item(row, 2).setText(f"{event.ydata:.2f}")
        self._updating = False; self._preview_points()

    def _accept_review(self):
        try:
            axes = {}
            for axis in ("x", "y"):
                points = self._points(axis)
                if len(points) > 15:
                    raise ValueError(self._tr("Choose at most 15 calibration points per axis."))
                scale = self.scales[axis].currentData()
                calibration_coefficients(points, axis, scale)
                axes[axis] = {"points": points, "scale": scale}
            self.reviewed = {"axes": axes}
        except ValueError as error:
            self.error.setText(str(error)); return
        self.accept()

    def reject(self):
        if self.worker.isRunning():
            self._closing = True; self.worker.requestInterruption(); self.buttons.setEnabled(False)
            self.summary.setText(self._tr("Stopping axis detection…"))
        else:
            super().reject()


class CurvePickerDialog(AdaptiveDialog):
    """Background detection followed by a reversible, visual curve choice."""

    def __init__(self, parent):
        super().__init__(parent)
        self._tr = parent._tr
        self.image = parent.canvas.image
        self.result_data = None
        self.selected_indices = []
        self._closing = False
        self.setWindowTitle(self._tr("Autopilot — choose curves"))
        self.resize(960, 760)
        layout = QVBoxLayout(self)
        self.summary = QLabel(self._tr("Detecting curves and choosing a preset…"))
        self.summary.setWordWrap(True)
        self.summary.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        layout.addWidget(self.summary)
        layout.addWidget(QLabel(self._tr("What color is the desired curve?")))
        self.colors = QComboBox()
        self.colors.addItem(self._tr("All colors"), None)
        self.colors.setEnabled(False)
        self.colors.currentIndexChanged.connect(self._filter_curves)
        layout.addWidget(self.colors)
        self.figure = Figure(facecolor="#f4f5f7")
        self.preview = FigureCanvas(self.figure)
        self.preview.setMinimumSize(240, 180)
        self.axes = self.figure.add_subplot(111)
        self.axes.imshow(self.image)
        self.axes.set_axis_off()
        self.figure.tight_layout(pad=0.3)
        self.preview.mpl_connect("button_press_event", self._pick_on_preview)
        layout.addWidget(self.preview, 1)
        hint = QLabel(self._tr("Click a curve on the preview or choose a row below."))
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.curves = QTableWidget(0, 4)
        self.curves.setHorizontalHeaderLabels([self._tr("Curve"), self._tr("Color"), self._tr("X coverage"), self._tr("Separation")])
        self.curves.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.curves.verticalHeader().setVisible(False)
        self.curves.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.curves.setSelectionMode(QAbstractItemView.SingleSelection)
        self.curves.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.curves.setMaximumHeight(150)
        self.curves.currentCellChanged.connect(lambda *_args: self._preview_curves())
        layout.addWidget(self.curves)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        layout.addWidget(self.progress)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Cancel)
        self.buttons.button(QDialogButtonBox.Cancel).setText(self._tr("Cancel"))
        self.use_button = self.buttons.addButton(self._tr("Add selected curve"), QDialogButtonBox.ActionRole)
        self.all_button = self.buttons.addButton(self._tr("Add all shown curves"), QDialogButtonBox.ActionRole)
        self.use_button.setObjectName("primary")
        self.use_button.setEnabled(False)
        self.all_button.setEnabled(False)
        self.use_button.clicked.connect(self._use_selected)
        self.all_button.clicked.connect(self._use_all)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        apply_window_appearance(self)
        self.worker = CurveDetectionWorker(self.image, parent.canvas.roi, self)
        self.worker.result_ready.connect(self._receive_result)
        self.worker.failed.connect(self._show_error)
        self.worker.finished.connect(self._detection_finished)
        self.worker.start()
        fit_to_screen(self)

    def _receive_result(self, result):
        self.result_data = result

    def _fit_summary(self):
        if hasattr(self, "summary"):
            bounds = self.summary.fontMetrics().boundingRect(
                0, 0, max(240, self.width() - 48), 10000, Qt.TextWordWrap, self.summary.text())
            self.summary.setFixedHeight(bounds.height() + 8)
            policy = self.summary.sizePolicy()
            policy.setHeightForWidth(False)
            self.summary.setSizePolicy(policy)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit_summary()

    def _show_error(self, error):
        self.summary.setText(self._tr("Detection failed: ") + error)

    def _detection_finished(self):
        self.progress.hide()
        if self._closing:
            super().reject()
            return
        if self.result_data is None:
            return
        candidates = self.result_data["candidates"]
        if not candidates:
            self.summary.setText(self._tr("No curves found. Draw a tighter ROI, then retry Autopilot or use Color Pick in Auto Extract."))
            return
        preset = self._tr(EXTRACTION_PRESETS[self.result_data["preset"]]["label"])
        summary = self._tr("{count} curve candidates · Preset: {preset} · Dark threshold: {threshold}.").format(
            count=len(candidates), preset=preset, threshold=self.result_data["threshold"])
        if self.result_data.get("overlap_regions"):
            summary += "\n" + self._tr("Review orange bands: curves may touch, merge or remain undetected.")
        if self.result_data.get("threshold_sensitive"):
            summary += "\n" + self._tr("Curve count changes with contrast; check for missing curves.")
        summary += "\n" + self._tr("Check the highlighted points before adding; X coverage is not accuracy.")
        self.summary.setText(summary)
        self._fit_summary()
        for color in dict.fromkeys(candidate["color"] for candidate in candidates):
            self.colors.addItem(self._tr(color), color)
        self.colors.setEnabled(True)
        self.curves.setRowCount(len(candidates))
        for row, candidate in enumerate(candidates):
            item = QTableWidgetItem(self._tr("Curve {number}").format(number=row + 1))
            item.setData(Qt.UserRole, row)
            self.curves.setItem(row, 0, item)
            color = QTableWidgetItem(self._tr(candidate["color"]))
            color.setForeground(QColor(*candidate["rgb"]).darker(140))
            self.curves.setItem(row, 1, color)
            self.curves.setItem(row, 2, QTableWidgetItem(f"{candidate['coverage']:.0%}"))
            separation = {"review": "Needs review", "limited": "Partial trace", "supported": "No overlap found"}
            self.curves.setItem(row, 3, QTableWidgetItem(self._tr(separation[candidate.get("separation", "limited")])))
        self.curves.selectRow(0)
        self.use_button.setEnabled(True)
        self.all_button.setEnabled(True)
        self._preview_curves()

    def _filter_curves(self):
        if self.result_data is None:
            return
        color = self.colors.currentData()
        for row, candidate in enumerate(self.result_data["candidates"]):
            self.curves.setRowHidden(row, color is not None and color != candidate["color"])
        shown = [row for row in range(self.curves.rowCount()) if not self.curves.isRowHidden(row)]
        if shown:
            self.curves.selectRow(shown[0])
        self._preview_curves()

    def _preview_curves(self):
        if not self.result_data:
            return
        self.axes.clear()
        self.axes.imshow(self.image)
        x0, y0, x1, y1 = self.result_data["roi"]
        self.axes.set_xlim(x0, x1)
        self.axes.set_ylim(y1, y0)
        self.axes.set_axis_off()
        for region in self.result_data.get("overlap_regions", []):
            pad = max(4, (x1 - x0) * 0.005)
            self.axes.axvspan(max(x0, region["x0"] - pad), min(x1, region["x1"] + pad),
                             facecolor="#e69f00", alpha=0.12, zorder=1)
        for index, candidate in enumerate(self.result_data["candidates"]):
            if self.curves.isRowHidden(index):
                continue
            points = np.asarray(candidate["points"])
            selected = index == self.curves.currentRow()
            color = PlotCanvas.DS_COLORS[index % len(PlotCanvas.DS_COLORS)]
            self.axes.scatter(points[:, 0], points[:, 1], s=9 if selected else 2,
                              color=color, alpha=1 if selected else 0.30, zorder=5 if selected else 3)
            self.axes.annotate(str(index + 1), points[len(points) // 8], color="white",
                               bbox={"facecolor": color, "edgecolor": "white", "pad": 2}, zorder=6)
        self.figure.tight_layout(pad=0.3)
        self.preview.draw_idle()

    def _pick_on_preview(self, event):
        if event.inaxes != self.axes or self.result_data is None or event.button != 1:
            return
        nearest = None
        for row, candidate in enumerate(self.result_data["candidates"]):
            if self.curves.isRowHidden(row):
                continue
            points = self.axes.transData.transform(np.asarray(candidate["points"]))
            distance = float(np.min(np.linalg.norm(points - (event.x, event.y), axis=1)))
            if nearest is None or distance < nearest[0]:
                nearest = (distance, row)
        if nearest is not None and nearest[0] <= 18:
            self.curves.selectRow(nearest[1])

    def _use_selected(self):
        row = self.curves.currentRow()
        if row >= 0 and not self.curves.isRowHidden(row):
            self.selected_indices = [row]
            self.accept()

    def _use_all(self):
        self.selected_indices = [row for row in range(self.curves.rowCount()) if not self.curves.isRowHidden(row)]
        if self.selected_indices:
            self.accept()

    def reject(self):
        if self.worker.isRunning():
            self._closing = True
            self.worker.requestInterruption()
            self.summary.setText(self._tr("Stopping curve detection…"))
            self.buttons.setEnabled(False)
        else:
            super().reject()


class MainWindow(QMainWindow):
    def __init__(
        self,
        dpi_scale=1.0,
        parent=None,
        reference_projects=(),
        initial_project="",
        add_reference_callback=None,
        language="en",
        standalone=False,
    ):
        super().__init__(parent)
        self.setWindowFlag(Qt.Window, True)
        # Keep widget geometry in Qt logical pixels.  Qt's high-DPI support
        # scales those pixels once; multiplying them by the screen DPI here
        # double-scales the standalone window on mixed-DPI machines.
        self._dpi_scale = 1.0
        self._image_path = ""
        self._reference_projects = sorted({
            str(project).strip() for project in reference_projects if str(project).strip()
        })
        self._initial_project = str(initial_project).strip()
        self._add_reference_callback = add_reference_callback
        self._standalone = bool(standalone)
        self.language = "zh" if str(language).strip().lower() == "zh" else "en"
        self._translatable_actions = []
        self._translatable_controls = []
        self._translatable_combos = []
        self._translatable_tabs = []
        self._translatable_table_headers = []
        self._translatable_tooltips = []
        self.setWindowTitle("PlotDigitizer — Separated Axis Calibration + ROI")
        available = self.screen().availableGeometry()
        width = min(1600, round(available.width() * 0.94))
        height = min(1050, round(available.height() * 0.90))
        self.setMinimumSize(640, 400)
        self.resize(width, height)
        self._eyedrop_active = False
        self._build_ui(); self._build_menu()
        self._register_translatable_controls()
        self.adaptive_panes.modeChanged.connect(lambda _compact: self._apply_language())
        self._apply_language()
        apply_window_appearance(self)
        self._update_control_metrics()
        self._update_status(self._tr("Ready — File > Open Image to begin"))

    def _build_menu(self):
        mb=self.menuBar()
        self.menu_file=fm=mb.addMenu("&File")
        for lbl,sc,fn in [("Open Image…","Ctrl+O",self._open_image),
                           ("Export CSV…","Ctrl+S",self._export_csv),
                           ("Export JSON…",None,self._export_json),
                           ("Export XRDML…",None,self._export_xrdml),
                           ("Quit","Ctrl+Q",self.close)]:
            a=QAction(lbl,self)
            a.setProperty("i18n_key", lbl)
            if sc: a.setShortcut(sc)
            a.triggered.connect(fn); fm.addAction(a)
            self._translatable_actions.append(a)
            if lbl in ("Open Image…","Export JSON…"): fm.addSeparator()
        self.menu_edit=em=mb.addMenu("&Edit")
        for lbl,sc,fn in [("Undo Last Point","Ctrl+Z",self._undo_point),
                           ("Reject Points in Box","R",self._start_reject_box),
                           ("Clear Active Dataset",None,self._clear_dataset),
                           ("Reset Calibration",None,self._reset_calibration),
                           ("Clear ROI",None,self._clear_roi)]:
            a=QAction(lbl,self)
            a.setProperty("i18n_key", lbl)
            if sc: a.setShortcut(sc)
            a.triggered.connect(fn); em.addAction(a)
            self._translatable_actions.append(a)
        self.menu_help=hm=mb.addMenu("&Help")
        a=QAction("Extraction Presets & Instructions…",self)
        a.setProperty("i18n_key","Extraction Presets & Instructions…")
        a.triggered.connect(self._show_extraction_instructions); hm.addAction(a)
        self._translatable_actions.append(a); hm.addSeparator()
        a=QAction("About",self); a.setProperty("i18n_key","About")
        a.triggered.connect(self._about); hm.addAction(a)
        self._translatable_actions.append(a)

    def _tr(self, text):
        return ZH_LABELS.get(text, text) if self.language == "zh" else text

    def set_language(self, language):
        self.language = "zh" if str(language).strip().lower() == "zh" else "en"
        self._apply_language()
        self._update_status(self._tr("Ready — File > Open Image to begin"))

    def _register_translatable_controls(self):
        seen = set()
        for cls in (QPushButton, QCheckBox, QLabel):
            for widget in self.findChildren(cls):
                key = widget.text()
                if key not in ZH_LABELS or id(widget) in seen:
                    continue
                widget.setProperty("i18n_key", key)
                self._translatable_controls.append(widget)
                seen.add(id(widget))
                self._register_tooltip(widget)
        for widget in self.findChildren(QGroupBox):
            key = widget.title()
            if key in ZH_LABELS and id(widget) not in seen:
                widget.setProperty("i18n_key", key)
                self._translatable_controls.append(widget)
                seen.add(id(widget))
            self._register_tooltip(widget)
        dataset_combos = {id(getattr(self, "combo_ds", None)),
                          id(getattr(self, "combo_ds_auto", None))}
        for combo in self.findChildren(QComboBox):
            if id(combo) in dataset_combos:
                continue
            keys = []
            for idx in range(combo.count()):
                key = combo.itemText(idx)
                keys.append(key)
            if any(key in ZH_LABELS for key in keys):
                combo.setProperty("i18n_keys", keys)
                self._translatable_combos.append(combo)
            self._register_tooltip(combo)
        for tabs in self.findChildren(QTabWidget):
            keys = [tabs.tabText(idx) for idx in range(tabs.count())]
            if any(key in ZH_LABELS for key in keys):
                tabs.setProperty("i18n_keys", keys)
                self._translatable_tabs.append(tabs)
        for table in self.findChildren(QTableWidget):
            keys = []
            for idx in range(table.columnCount()):
                item = table.horizontalHeaderItem(idx)
                keys.append(item.text() if item else "")
            if any(key in ZH_LABELS for key in keys):
                table.setProperty("i18n_header_keys", keys)
                self._translatable_table_headers.append(table)
        for widget in self.findChildren(QWidget):
            self._register_tooltip(widget)

    def _register_tooltip(self, widget):
        key = widget.toolTip()
        if key and key in ZH_LABELS and id(widget) not in {id(w) for w in self._translatable_tooltips}:
            widget.setProperty("i18n_tooltip_key", key)
            self._translatable_tooltips.append(widget)

    def _apply_language(self):
        if hasattr(self, "adaptive_panes") and self.adaptive_panes.compact:
            for index, title in enumerate(self.adaptive_panes.titles):
                self.adaptive_panes.tabs.setTabText(index, self._tr(title))
        if hasattr(self, "menu_file"):
            self.menu_file.setTitle(self._tr("&File"))
            self.menu_edit.setTitle(self._tr("&Edit"))
            self.menu_help.setTitle(self._tr("&Help"))
        self.setWindowTitle(self._tr("PlotDigitizer — Separated Axis Calibration + ROI"))
        for action in self._translatable_actions:
            key = action.property("i18n_key")
            if key:
                action.setText(self._tr(key))
        for widget in self._translatable_controls:
            key = widget.property("i18n_key")
            if not key:
                continue
            if isinstance(widget, QGroupBox):
                widget.setTitle(self._tr(key))
            else:
                widget.setText(self._tr(key))
        for combo in self._translatable_combos:
            keys = combo.property("i18n_keys") or []
            current = combo.currentIndex()
            combo.blockSignals(True)
            for idx, key in enumerate(keys):
                combo.setItemText(idx, self._tr(key))
            combo.setCurrentIndex(current)
            combo.blockSignals(False)
        for tabs in self._translatable_tabs:
            keys = tabs.property("i18n_keys") or []
            for idx, key in enumerate(keys):
                tabs.setTabText(idx, self._tr(key))
        for table in self._translatable_table_headers:
            keys = table.property("i18n_header_keys") or []
            for idx, key in enumerate(keys):
                if key:
                    item = table.horizontalHeaderItem(idx)
                    if item:
                        item.setText(self._tr(key))
        for widget in self._translatable_tooltips:
            key = widget.property("i18n_tooltip_key")
            if key:
                widget.setToolTip(self._tr(key))
        self._sync_dynamic_labels()
        self._update_control_metrics()

    def _sync_dynamic_labels(self):
        if not hasattr(self, "canvas"):
            return
        if self.canvas.roi is None:
            self.lbl_roi.setText(self._tr("No ROI — full image used"))
        else:
            x0,y0,x1,y1 = self.canvas.roi
            self.lbl_roi.setText(f"ROI: ({x0:.0f},{y0:.0f}) → ({x1:.0f},{y1:.0f})")
        self._update_calib_status()
        self._update_preset_summary()

    def _build_ui(self):
        central=QWidget(); self.setCentralWidget(central)
        root=QHBoxLayout(central); root.setContentsMargins(0,0,0,0); root.setSpacing(0)
        splitter=QSplitter(Qt.Horizontal); root.addWidget(splitter)

        scroll=QScrollArea(); scroll.setWidgetResizable(True)
        panel_w = 420
        scroll.setMinimumWidth(panel_w)
        self.controls_scroll = scroll
        left=QWidget(); scroll.setWidget(left)
        lv=QVBoxLayout(left); lv.setContentsMargins(8,8,8,8); lv.setSpacing(8)

        # Open image
        g=QGroupBox("Image"); gv=QVBoxLayout(g)
        b=QPushButton("📂  Open Image…"); b.setObjectName("primary")
        b.clicked.connect(self._open_image); gv.addWidget(b); lv.addWidget(g)

        g=QGroupBox("Autopilot"); gv=QVBoxLayout(g)
        hint=QLabel("Open an image and calibrate the axes, then let Autopilot choose a preset and find the curves. Optional: draw an ROI first.")
        hint.setWordWrap(True); gv.addWidget(hint)
        self.autopilot_button=QPushButton("Find all curves, then choose")
        self.autopilot_button.setObjectName("primary")
        self.autopilot_button.clicked.connect(self._run_autopilot)
        gv.addWidget(self.autopilot_button); lv.addWidget(g)

        # Scale
        g=QGroupBox("Axis Scale"); gv=QVBoxLayout(g)
        self.combo_scale=QComboBox(); self.combo_scale.addItems(["Linear","Semi-log Y","Log-Log","Semi-log X"])
        self.combo_scale.currentIndexChanged.connect(self._scale_changed)
        gv.addWidget(self.combo_scale); lv.addWidget(g)

        # ── ROI ───────────────────────────────────────────────────────────────
        g=QGroupBox("Step 0 — Plot Area (ROI)  [recommended]"); gv=QVBoxLayout(g)
        hint=QLabel("Detect the plot area automatically, or draw a box inside the axes to exclude labels, borders, and margins.")
        hint.setWordWrap(True); gv.addWidget(hint)
        self.chk_auto_roi=QCheckBox("Auto-detect ROI on image load")
        self.chk_auto_roi.setChecked(True); gv.addWidget(self.chk_auto_roi)
        self.auto_roi_button=QPushButton("Auto-detect ROI")
        self.auto_roi_button.setObjectName("roi")
        self.auto_roi_button.clicked.connect(self._detect_roi); gv.addWidget(self.auto_roi_button)
        b=QPushButton("🟩  Draw ROI Box on Image"); b.setObjectName("roi")
        b.clicked.connect(self._start_roi); gv.addWidget(b)
        row=QHBoxLayout()
        b2=QPushButton("✕  Clear ROI"); b2.setObjectName("danger"); b2.clicked.connect(self._clear_roi); row.addWidget(b2)
        gv.addLayout(row)
        self.lbl_roi=QLabel("No ROI — full image used"); self.lbl_roi.setAlignment(Qt.AlignCenter)
        gv.addWidget(self.lbl_roi); lv.addWidget(g)

        # ── Calibration ───────────────────────────────────────────────────────
        g=QGroupBox("Step 1 — Calibrate Axes"); gv=QVBoxLayout(g)
        gv.addWidget(QLabel(
            "X and Y are calibrated independently.\n"
            "No need to use the origin — click any\n"
            "visible tick marks on each axis."))
        self.auto_calibration_button = QPushButton("Suggest axis calibration…")
        self.auto_calibration_button.setObjectName("primary")
        self.auto_calibration_button.clicked.connect(self._run_auto_calibration)
        gv.addWidget(self.auto_calibration_button)

        # X axis
        xg=QGroupBox("X Axis  (orange ▼ markers)"); xgv=QVBoxLayout(xg)
        nxr=QHBoxLayout(); nxr.addWidget(QLabel("# X points:"))
        self.spin_nx=QSpinBox(); self.spin_nx.setRange(2,15); self.spin_nx.setValue(3)
        self.spin_nx.valueChanged.connect(self._rebuild_x_table)
        nxr.addWidget(self.spin_nx); xgv.addLayout(nxr)
        self.x_calib_table=QTableWidget(3,2)
        self.x_calib_table.setHorizontalHeaderLabels(["Label","Real X"])
        self.x_calib_table.horizontalHeader().setSectionResizeMode(0,QHeaderView.ResizeToContents)
        self.x_calib_table.horizontalHeader().setSectionResizeMode(1,QHeaderView.Stretch)
        self.x_calib_table.verticalHeader().setVisible(False)
        self.x_calib_table.verticalHeader().setDefaultSectionSize(26)
        self.x_calib_table.setMinimumHeight(110)
        self.x_calib_table.setMaximumHeight(int(300 * self._dpi_scale))
        xgv.addWidget(self.x_calib_table)
        bx=QPushButton("🖱  Click X Ticks on Image"); bx.setObjectName("xbtn")
        bx.clicked.connect(self._start_x_calibration); xgv.addWidget(bx)
        self.lbl_x_calib=QLabel("○ ○ ○  awaiting X points")
        self.lbl_x_calib.setAlignment(Qt.AlignCenter); xgv.addWidget(self.lbl_x_calib)
        gv.addWidget(xg)

        # Y axis
        yg=QGroupBox("Y Axis  (cyan ▶ markers)"); ygv=QVBoxLayout(yg)
        nyr=QHBoxLayout(); nyr.addWidget(QLabel("# Y points:"))
        self.spin_ny=QSpinBox(); self.spin_ny.setRange(2,15); self.spin_ny.setValue(3)
        self.spin_ny.valueChanged.connect(self._rebuild_y_table)
        nyr.addWidget(self.spin_ny); ygv.addLayout(nyr)
        self.y_calib_table=QTableWidget(3,2)
        self.y_calib_table.setHorizontalHeaderLabels(["Label","Real Y"])
        self.y_calib_table.horizontalHeader().setSectionResizeMode(0,QHeaderView.ResizeToContents)
        self.y_calib_table.horizontalHeader().setSectionResizeMode(1,QHeaderView.Stretch)
        self.y_calib_table.verticalHeader().setVisible(False)
        self.y_calib_table.verticalHeader().setDefaultSectionSize(26)
        self.y_calib_table.setMinimumHeight(110)
        self.y_calib_table.setMaximumHeight(int(300 * self._dpi_scale))
        ygv.addWidget(self.y_calib_table)
        by_=QPushButton("🖱  Click Y Ticks on Image"); by_.setObjectName("ybtn")
        by_.clicked.connect(self._start_y_calibration); ygv.addWidget(by_)
        self.lbl_y_calib=QLabel("○ ○ ○  awaiting Y points")
        self.lbl_y_calib.setAlignment(Qt.AlignCenter); ygv.addWidget(self.lbl_y_calib)
        gv.addWidget(yg)

        row=QHBoxLayout()
        b2=QPushButton("↺  Reset Calibration"); b2.setObjectName("danger")
        b2.clicked.connect(self._reset_calibration); row.addWidget(b2); gv.addLayout(row)
        self.lbl_calib=QLabel("⬤ X: not set     ⬤ Y: not set")
        self.lbl_calib.setAlignment(Qt.AlignCenter); gv.addWidget(self.lbl_calib)
        lv.addWidget(g)

        # Populate tables after widgets exist
        self._rebuild_x_table(3); self._rebuild_y_table(3)

        # ── Tabs ─────────────────────────────────────────────────────────────
        tabs=QTabWidget(); lv.addWidget(tabs)

        # Manual tab
        mw=QWidget(); mv=QVBoxLayout(mw); mv.setSpacing(6)
        row=QHBoxLayout(); row.addWidget(QLabel("Dataset:"))
        self.combo_ds=QComboBox(); self.combo_ds.addItem("Dataset 1")
        self.combo_ds.currentIndexChanged.connect(self._ds_changed); row.addWidget(self.combo_ds); mv.addLayout(row)
        b=QPushButton("＋  New Dataset"); b.clicked.connect(self._add_dataset); mv.addWidget(b)
        b=QPushButton("✖  Delete Dataset"); b.setObjectName("danger"); b.clicked.connect(self._delete_dataset); mv.addWidget(b)
        b=QPushButton("🖱  Start Digitizing"); b.setObjectName("primary"); b.clicked.connect(self._start_digitize); mv.addWidget(b)
        b=QPushButton("▭  Reject Points in Box"); b.setObjectName("danger"); b.clicked.connect(self._start_reject_box); mv.addWidget(b)
        b=QPushButton("↩  Undo Last  (Ctrl+Z)"); b.clicked.connect(self._undo_point); mv.addWidget(b)
        b=QPushButton("🗑  Clear Dataset"); b.setObjectName("danger"); b.clicked.connect(self._clear_dataset); mv.addWidget(b)
        mv.addStretch(); tabs.addTab(mw,"✋ Manual")

        # Auto tab
        aw=QWidget(); av=QVBoxLayout(aw); av.setSpacing(8)
        row=QHBoxLayout(); row.addWidget(QLabel("Into dataset:"))
        self.combo_ds_auto=QComboBox(); self.combo_ds_auto.addItem("Dataset 1")
        self.combo_ds_auto.currentIndexChanged.connect(self._ds_changed); row.addWidget(self.combo_ds_auto); av.addLayout(row)
        b=QPushButton("＋  New Dataset"); b.clicked.connect(self._add_dataset); av.addWidget(b)
        b=QPushButton("✖  Delete Dataset"); b.setObjectName("danger"); b.clicked.connect(self._delete_dataset); av.addWidget(b)
        b=QPushButton("▭  Reject Points in Box"); b.setObjectName("danger"); b.clicked.connect(self._start_reject_box); av.addWidget(b)

        pg=QGroupBox("Extraction Presets"); pgv=QVBoxLayout(pg)
        prow=QHBoxLayout(); prow.addWidget(QLabel("Extraction preset:"))
        self.combo_extraction_preset=QComboBox()
        for preset_key in ("normal", "grid", "waterfall"):
            self.combo_extraction_preset.addItem(
                EXTRACTION_PRESETS[preset_key]["label"], preset_key
            )
        self.combo_extraction_preset.setCurrentIndex(
            self.combo_extraction_preset.findData("grid")
        )
        self.combo_extraction_preset.currentIndexChanged.connect(
            lambda _index: self._update_preset_summary()
        )
        prow.addWidget(self.combo_extraction_preset, 1); pgv.addLayout(prow)
        self.lbl_preset_summary=QLabel()
        self.lbl_preset_summary.setWordWrap(True); pgv.addWidget(self.lbl_preset_summary)
        prow=QHBoxLayout()
        b=QPushButton("Apply Preset"); b.setObjectName("primary")
        b.clicked.connect(lambda _checked=False: self._apply_extraction_preset()); prow.addWidget(b)
        b=QPushButton("Instructions / Print…")
        b.clicked.connect(self._show_extraction_instructions); prow.addWidget(b)
        pgv.addLayout(prow); av.addWidget(pg)

        # Dark curve mode — prominent for black-on-white plots
        dg=QGroupBox("Mode 1 — Dark Curve  ★ (black on white)"); dgv=QVBoxLayout(dg)
        dgv.addWidget(QLabel("Best for black/dark lines on light background.\n(e.g. patent figures, scanned PXRD)"))
        dgv.addWidget(QLabel("Darkness threshold (0=black, 255=white):"))
        self.slider_dark=QSlider(Qt.Horizontal); self.slider_dark.setRange(10,240); self.slider_dark.setValue(128)
        self.lbl_dark=QLabel("128"); self.slider_dark.valueChanged.connect(lambda v: self.lbl_dark.setText(str(v)))
        dr=QHBoxLayout(); dr.addWidget(self.slider_dark); dr.addWidget(self.lbl_dark); dgv.addLayout(dr)
        self.chk_dark_auto_threshold=QCheckBox("Auto-select darkness threshold (128–235)")
        self.chk_dark_auto_threshold.setChecked(True)
        self.chk_dark_auto_threshold.toggled.connect(
            lambda checked: self.slider_dark.setEnabled(not checked)
        )
        self.slider_dark.setEnabled(False)
        dgv.addWidget(self.chk_dark_auto_threshold)
        dgv.addWidget(QLabel("Column step (1=densest):"))
        self.spin_step=QSpinBox(); self.spin_step.setRange(1,20); self.spin_step.setValue(1); dgv.addWidget(self.spin_step)
        dgv.addWidget(QLabel("Min run length (noise filter, no-conn only):"))
        self.spin_run=QSpinBox(); self.spin_run.setRange(1,20); self.spin_run.setValue(2); dgv.addWidget(self.spin_run)
        row=QHBoxLayout(); row.addWidget(QLabel("Pick:"))
        self.combo_dark_pick=QComboBox()
        self.combo_dark_pick.addItems(["topmost  (upper / single curve)",
                                       "bottommost  (lower curve in waterfall)",
                                       "center  (thick strokes)"])
        row.addWidget(self.combo_dark_pick); dgv.addLayout(row)
        # Y-band (vertical restriction)
        self.chk_dark_yband=QCheckBox("📏  Y-band (restrict rows for waterfall curves)")
        self.chk_dark_yband.setChecked(False); dgv.addWidget(self.chk_dark_yband)
        yband_box_d=QGroupBox("Y-band (pixel rows in image)"); yband_v_d=QVBoxLayout(yband_box_d)
        yband_box_d.setVisible(False)
        yband_v_d.addWidget(QLabel(
            "Limit extraction to rows Y_min...Y_max.\n"
            "Hover image to read pixel Y values.\n"
            "  topmost+Y-band   -> upper/mid curve\n"
            "  bottommost+Y-band -> lower/mid curve"))
        r_yb=QHBoxLayout()
        r_yb.addWidget(QLabel("Y min (px):")); self.spin_dark_ymin=QSpinBox()
        self.spin_dark_ymin.setRange(0,99999); self.spin_dark_ymin.setValue(0); r_yb.addWidget(self.spin_dark_ymin)
        r_yb.addWidget(QLabel("Y max (px):")); self.spin_dark_ymax=QSpinBox()
        self.spin_dark_ymax.setRange(0,99999); self.spin_dark_ymax.setValue(500); r_yb.addWidget(self.spin_dark_ymax)
        yband_v_d.addLayout(r_yb); dgv.addWidget(yband_box_d)
        self.chk_dark_yband.toggled.connect(yband_box_d.setVisible)

        # Connectivity filter controls
        self.chk_dark_conn=QCheckBox("🔗  Connectivity filter  (removes labels/text)"); self.chk_dark_conn.setChecked(True)
        dgv.addWidget(self.chk_dark_conn)
        self.chk_dark_grid=QCheckBox("Suppress long grid / axis lines")
        self.chk_dark_grid.setChecked(True); dgv.addWidget(self.chk_dark_grid)
        grid_row=QHBoxLayout(); grid_row.addWidget(QLabel("Grid suppression:"))
        self.combo_dark_grid_strength=QComboBox()
        self.combo_dark_grid_strength.addItem("Standard (straight lines)", "standard")
        self.combo_dark_grid_strength.addItem("Strong (wavy / scanned grids)", "strong")
        self.combo_dark_grid_strength.setCurrentIndex(1)
        self.combo_dark_grid_strength.setToolTip(
            "Strong mode detects long line bands in noisy patent scans and removes their signals before tracing."
        )
        grid_row.addWidget(self.combo_dark_grid_strength, 1); dgv.addLayout(grid_row)
        self.chk_dark_grid.toggled.connect(self.combo_dark_grid_strength.setEnabled)
        conn_box=QGroupBox("Connectivity parameters"); conn_v=QVBoxLayout(conn_box)
        conn_box.setVisible(True)
        r1=QHBoxLayout(); r1.addWidget(QLabel("Min component area (×total):"))
        self.spin_dark_maf=QDoubleSpinBox(); self.spin_dark_maf.setRange(0.0001,0.2); self.spin_dark_maf.setDecimals(4); self.spin_dark_maf.setSingleStep(0.001); self.spin_dark_maf.setValue(0.003); r1.addWidget(self.spin_dark_maf); conn_v.addLayout(r1)
        r2=QHBoxLayout(); r2.addWidget(QLabel("Min width (×image width):"))
        self.spin_dark_mwf=QDoubleSpinBox(); self.spin_dark_mwf.setRange(0.001,0.5); self.spin_dark_mwf.setDecimals(3); self.spin_dark_mwf.setSingleStep(0.01); self.spin_dark_mwf.setValue(0.05); r2.addWidget(self.spin_dark_mwf); conn_v.addLayout(r2)
        r3=QHBoxLayout(); r3.addWidget(QLabel("Max aspect ratio (w/h) to reject:"))
        self.spin_dark_asp=QDoubleSpinBox(); self.spin_dark_asp.setRange(1.0,20.0); self.spin_dark_asp.setDecimals(1); self.spin_dark_asp.setSingleStep(0.5); self.spin_dark_asp.setValue(8.0); r3.addWidget(self.spin_dark_asp); conn_v.addLayout(r3)
        r4=QHBoxLayout(); r4.addWidget(QLabel("Keep top N components:"))
        self.spin_dark_topn=QSpinBox(); self.spin_dark_topn.setRange(1,20); self.spin_dark_topn.setValue(5); r4.addWidget(self.spin_dark_topn); conn_v.addLayout(r4)
        dgv.addWidget(conn_box)
        self.chk_dark_conn.toggled.connect(conn_box.setVisible)
        b=QPushButton("⚡  Extract Dark Curve"); b.setObjectName("extract"); b.clicked.connect(self._run_dark); dgv.addWidget(b)
        av.addWidget(dg)

        # Color mode
        cg=QGroupBox("Mode 2 — Color Pick  (colored curves)"); cgv=QVBoxLayout(cg)
        crow=QHBoxLayout()
        self.color_swatch=ColorSwatch((50,50,200)); self.color_swatch.clicked.connect(self._pick_color_dialog)
        self.btn_eyedrop=QPushButton("🎯 Eyedrop"); self.btn_eyedrop.clicked.connect(self._start_eyedrop)
        crow.addWidget(self.color_swatch); crow.addWidget(self.btn_eyedrop); cgv.addLayout(crow)
        cgv.addWidget(QLabel("Hue tolerance:"))
        self.slider_tol=QSlider(Qt.Horizontal); self.slider_tol.setRange(5,60); self.slider_tol.setValue(25)
        self.lbl_tol=QLabel("25"); self.slider_tol.valueChanged.connect(lambda v: self.lbl_tol.setText(str(v)))
        tr=QHBoxLayout(); tr.addWidget(self.slider_tol); tr.addWidget(self.lbl_tol); cgv.addLayout(tr)
        cgv.addWidget(QLabel("Column step:"))
        self.spin_col_step_color=QSpinBox(); self.spin_col_step_color.setRange(1,50); self.spin_col_step_color.setValue(1); cgv.addWidget(self.spin_col_step_color)
        cgv.addWidget(QLabel("Smoothing band (px):"))
        self.spin_smooth=QSpinBox(); self.spin_smooth.setRange(1,31); self.spin_smooth.setValue(3); cgv.addWidget(self.spin_smooth)
        cgv.addWidget(QLabel("Min blob area (px²):"))
        self.spin_min_area=QSpinBox(); self.spin_min_area.setRange(1,5000); self.spin_min_area.setValue(10); cgv.addWidget(self.spin_min_area)
        row=QHBoxLayout(); row.addWidget(QLabel("Pick:"))
        self.combo_color_pick=QComboBox()
        self.combo_color_pick.addItems(["topmost  (upper / single curve)",
                                        "bottommost  (lower curve in waterfall)",
                                        "center  (thick/dotted)"])
        row.addWidget(self.combo_color_pick); cgv.addLayout(row)
        # Y-band
        self.chk_color_yband=QCheckBox("📏  Y-band (restrict rows for waterfall curves)")
        self.chk_color_yband.setChecked(False); cgv.addWidget(self.chk_color_yband)
        yband_box_c=QGroupBox("Y-band (pixel rows in image)"); yband_v_c=QVBoxLayout(yband_box_c)
        yband_box_c.setVisible(False)
        yband_v_c.addWidget(QLabel(
            "Limit extraction to rows Y_min...Y_max.\n"
            "Hover over the image to read pixel Y values."))
        r_ybc=QHBoxLayout()
        r_ybc.addWidget(QLabel("Y min (px):")); self.spin_color_ymin=QSpinBox()
        self.spin_color_ymin.setRange(0,99999); self.spin_color_ymin.setValue(0); r_ybc.addWidget(self.spin_color_ymin)
        r_ybc.addWidget(QLabel("Y max (px):")); self.spin_color_ymax=QSpinBox()
        self.spin_color_ymax.setRange(0,99999); self.spin_color_ymax.setValue(500); r_ybc.addWidget(self.spin_color_ymax)
        yband_v_c.addLayout(r_ybc); cgv.addWidget(yband_box_c)
        self.chk_color_yband.toggled.connect(yband_box_c.setVisible)

        self.chk_color_conn=QCheckBox("🔗  Connectivity filter  (removes labels/text)"); self.chk_color_conn.setChecked(True)
        cgv.addWidget(self.chk_color_conn)
        conn_box_c=QGroupBox("Connectivity parameters"); conn_vc=QVBoxLayout(conn_box_c)
        r1c=QHBoxLayout(); r1c.addWidget(QLabel("Min component area (×total):"))
        self.spin_color_maf=QDoubleSpinBox(); self.spin_color_maf.setRange(0.0001,0.2); self.spin_color_maf.setDecimals(4); self.spin_color_maf.setSingleStep(0.001); self.spin_color_maf.setValue(0.005); r1c.addWidget(self.spin_color_maf); conn_vc.addLayout(r1c)
        r2c=QHBoxLayout(); r2c.addWidget(QLabel("Min width (×image width):"))
        self.spin_color_mwf=QDoubleSpinBox(); self.spin_color_mwf.setRange(0.001,0.5); self.spin_color_mwf.setDecimals(3); self.spin_color_mwf.setSingleStep(0.01); self.spin_color_mwf.setValue(0.05); r2c.addWidget(self.spin_color_mwf); conn_vc.addLayout(r2c)
        r3c=QHBoxLayout(); r3c.addWidget(QLabel("Max aspect (w/h) to reject:"))
        self.spin_color_asp=QDoubleSpinBox(); self.spin_color_asp.setRange(1.0,20.0); self.spin_color_asp.setDecimals(1); self.spin_color_asp.setSingleStep(0.5); self.spin_color_asp.setValue(8.0); r3c.addWidget(self.spin_color_asp); conn_vc.addLayout(r3c)
        r4c=QHBoxLayout(); r4c.addWidget(QLabel("Keep top N components:"))
        self.spin_color_topn=QSpinBox(); self.spin_color_topn.setRange(1,20); self.spin_color_topn.setValue(5); r4c.addWidget(self.spin_color_topn); conn_vc.addLayout(r4c)
        cgv.addWidget(conn_box_c)
        self.chk_color_conn.toggled.connect(conn_box_c.setVisible)
        b=QPushButton("⚡  Extract by Color"); b.setObjectName("extract"); b.clicked.connect(self._run_color); cgv.addWidget(b)
        av.addWidget(cg)

        # Blob mode
        bg=QGroupBox("Mode 3 — Blob / Scatter"); bgv=QVBoxLayout(bg)
        bgv.addWidget(QLabel("Detects circular scatter-dot markers."))
        row=QHBoxLayout()
        row.addWidget(QLabel("Min area:")); self.spin_bmin=QSpinBox(); self.spin_bmin.setRange(1,1000); self.spin_bmin.setValue(15); row.addWidget(self.spin_bmin)
        row.addWidget(QLabel("Max:")); self.spin_bmax=QSpinBox(); self.spin_bmax.setRange(100,50000); self.spin_bmax.setValue(3000); row.addWidget(self.spin_bmax); bgv.addLayout(row)
        bgv.addWidget(QLabel("Min circularity:"))
        self.spin_circ=QDoubleSpinBox(); self.spin_circ.setRange(0.1,1.0); self.spin_circ.setSingleStep(0.05); self.spin_circ.setValue(0.4); bgv.addWidget(self.spin_circ)
        b=QPushButton("⚡  Detect Blobs"); b.setObjectName("extract"); b.clicked.connect(self._run_blob); bgv.addWidget(b)
        av.addWidget(bg)

        row=QHBoxLayout(); row.addWidget(QLabel("Max points:"))
        self.spin_maxpts=QSpinBox(); self.spin_maxpts.setRange(10,100000); self.spin_maxpts.setValue(3000); row.addWidget(self.spin_maxpts); av.addLayout(row)
        self.progress=QProgressBar(); self.progress.setVisible(False); av.addWidget(self.progress)
        av.addStretch(); tabs.addTab(aw,"⚡ Auto Extract")

        # Export
        g=QGroupBox("Export"); gv=QVBoxLayout(g)
        b=QPushButton("💾  Export CSV");    b.clicked.connect(self._export_csv);   gv.addWidget(b)
        b=QPushButton("💾  Export JSON");   b.clicked.connect(self._export_json);  gv.addWidget(b)
        bx=QPushButton("📡  Export XRDML"); bx.setObjectName("primary")
        bx.setToolTip("Export active dataset as PANalytical XRDML\nfor direct import into HighScore / HighScore Plus")
        bx.clicked.connect(self._export_xrdml); gv.addWidget(bx)
        self.chk_xrdml_opts=QCheckBox("⚙  XRDML options (wavelength, source)"); gv.addWidget(self.chk_xrdml_opts)
        xrdml_box=QGroupBox("XRDML Parameters"); xrdml_v=QVBoxLayout(xrdml_box); xrdml_box.setVisible(False)
        r=QHBoxLayout(); r.addWidget(QLabel("Ka1 (Å):"))
        self.spin_ka1=QDoubleSpinBox(); self.spin_ka1.setDecimals(7); self.spin_ka1.setRange(0.1,10); self.spin_ka1.setSingleStep(0.0001); self.spin_ka1.setValue(1.5405980); r.addWidget(self.spin_ka1); xrdml_v.addLayout(r)
        r=QHBoxLayout(); r.addWidget(QLabel("Ka2 (Å):"))
        self.spin_ka2=QDoubleSpinBox(); self.spin_ka2.setDecimals(7); self.spin_ka2.setRange(0.1,10); self.spin_ka2.setSingleStep(0.0001); self.spin_ka2.setValue(1.5444260); r.addWidget(self.spin_ka2); xrdml_v.addLayout(r)
        r=QHBoxLayout(); r.addWidget(QLabel("Voltage (kV):"))
        self.spin_kv=QDoubleSpinBox(); self.spin_kv.setDecimals(1); self.spin_kv.setRange(1,100); self.spin_kv.setValue(45.0); r.addWidget(self.spin_kv); xrdml_v.addLayout(r)
        r=QHBoxLayout(); r.addWidget(QLabel("Current (mA):"))
        self.spin_ma=QDoubleSpinBox(); self.spin_ma.setDecimals(1); self.spin_ma.setRange(1,100); self.spin_ma.setValue(40.0); r.addWidget(self.spin_ma); xrdml_v.addLayout(r)
        # Counting time
        r=QHBoxLayout(); r.addWidget(QLabel("Counting time (s):"))
        self.spin_ct=QDoubleSpinBox(); self.spin_ct.setDecimals(3)
        self.spin_ct.setRange(0.001,100000); self.spin_ct.setValue(1.0)
        self.spin_ct.setToolTip(
            "Set to 1.0 if your Y axis is already in CPS (default).\n"
            "Set to actual scan dwell time (s) if Y axis is raw counts.")
        r.addWidget(self.spin_ct); xrdml_v.addLayout(r)
        # Y-scale factor
        r=QHBoxLayout(); r.addWidget(QLabel("Y-scale factor:"))
        self.spin_yscale=QDoubleSpinBox(); self.spin_yscale.setDecimals(4)
        self.spin_yscale.setRange(0.0001,10000); self.spin_yscale.setValue(1.0)
        self.spin_yscale.setToolTip(
            "Multiplied into all intensities before writing.\n"
            "If HighScore shows 5x too high, enter 0.2 (=1/5).")
        r.addWidget(self.spin_yscale); xrdml_v.addLayout(r)
        # Y baseline offset
        r=QHBoxLayout(); r.addWidget(QLabel("Y baseline offset (subtract):"))
        self.spin_yoffset=QDoubleSpinBox(); self.spin_yoffset.setDecimals(2)
        self.spin_yoffset.setRange(-1e7, 1e7); self.spin_yoffset.setValue(0.0)
        self.spin_yoffset.setToolTip(
            "Subtracted from ALL data values before writing.\n"
            "Use this when the digitized background is too high\n"
            "due to Y=0 calibration point being placed on the\n"
            "axis line instead of the true data zero.\n"
            "Enter the inflated background value here to zero it.")
        r.addWidget(self.spin_yoffset); xrdml_v.addLayout(r)
        # Auto-detect baseline button
        btn_bl=QPushButton("🎯  Auto-detect baseline from data")
        btn_bl.setToolTip(
            "Sets the Y baseline offset to the 5th-percentile\n"
            "intensity of the active dataset — a robust estimate\n"
            "of the background floor.")
        btn_bl.clicked.connect(self._auto_baseline); xrdml_v.addWidget(btn_bl)
        gv.addWidget(xrdml_box); self.chk_xrdml_opts.toggled.connect(xrdml_box.setVisible)
        lv.addWidget(g)

        self.reference_group=QGroupBox("Add to CrystalTrace Reference Database")
        refv=QVBoxLayout(self.reference_group)
        refv.addWidget(QLabel(
            "Adds the active digitized dataset as a project-specific phase reference."))
        refrow=QHBoxLayout(); refrow.addWidget(QLabel("Project:"))
        self.reference_project_combo=QComboBox()
        self.reference_project_combo.setMinimumContentsLength(18)
        self.reference_project_combo.addItems(self._reference_projects)
        if self._initial_project in self._reference_projects:
            self.reference_project_combo.setCurrentText(self._initial_project)
        refrow.addWidget(self.reference_project_combo, 1); refv.addLayout(refrow)
        phaserow=QHBoxLayout(); phaserow.addWidget(QLabel("Phase:"))
        self.reference_phase_edit=QLineEdit()
        self.reference_phase_edit.setPlaceholderText("Form A, Form B, alpha, beta...")
        phaserow.addWidget(self.reference_phase_edit, 1); refv.addLayout(phaserow)
        self.add_reference_button=QPushButton("Add Active Dataset as Reference")
        self.add_reference_button.setObjectName("primary")
        self.add_reference_button.setEnabled(
            self._add_reference_callback is not None and bool(self._reference_projects))
        self.add_reference_button.clicked.connect(self._add_active_dataset_as_reference)
        refv.addWidget(self.add_reference_button)
        lv.addWidget(self.reference_group)
        # The catalog supplies the callback and project context.  The separate
        # PlotDigitizer release keeps the same extraction UI but omits this
        # CrystalTrace-specific integration panel.
        self.reference_group.setVisible(not self._standalone)
        lv.addStretch(); splitter.addWidget(scroll)

        # Right panel
        right=QWidget(); rv=QVBoxLayout(right); rv.setContentsMargins(6,6,6,6); rv.setSpacing(6)
        self.canvas=PlotCanvas(self)
        self.canvas.calibration_click.connect(self._on_calib_click)
        self.canvas.datapoint_click.connect(self._on_data_click)
        self.canvas.roi_updated.connect(self._on_roi_updated)
        self.canvas.points_rejected.connect(self._on_points_rejected)
        self.canvas.mpl_connect("button_press_event", self._eyedrop_click)
        self.nav=NavigationToolbar(self.canvas,self)
        rv.addWidget(self.nav); rv.addWidget(self.canvas,stretch=3)
        tg=QGroupBox("Digitized Data"); tv=QVBoxLayout(tg)
        self.table=QTableWidget(0,3)
        self.table.setHorizontalHeaderLabels(["Dataset","X","Y"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setMaximumHeight(175); tv.addWidget(self.table)
        rv.addWidget(tg,stretch=1); splitter.addWidget(right)
        splitter.setStretchFactor(0,0); splitter.setStretchFactor(1,1)
        # Defer setSizes until after show() so geometry is resolved
        from PySide6.QtCore import QTimer
        QTimer.singleShot(0, lambda: splitter.setSizes(
            [int(420 * self._dpi_scale),
             splitter.width() - int(420 * self._dpi_scale)]))
        self.pane_splitter = splitter
        self.adaptive_panes = AdaptivePanes(
            self, root, splitter, [scroll, right], ["Controls", "Image & Data"], 1050,
        )
        for label in left.findChildren(QLabel):
            if "\n" in label.text() or len(label.text()) > 60:
                label.setWordWrap(True)
        self.adaptive_panes.update_layout()
        if self.adaptive_panes.compact:
            self.adaptive_panes.tabs.setCurrentIndex(1)
        self.status=QStatusBar(); self.setStatusBar(self.status)
        self._apply_extraction_preset(announce=False)

    def _update_control_metrics(self):
        if not hasattr(self, "controls_scroll"):
            return
        panel = self.controls_scroll.widget()
        title_width = max((group.fontMetrics().horizontalAdvance(group.title())
                           for group in panel.findChildren(QGroupBox)), default=0)
        for table in (self.x_calib_table, self.y_calib_table):
            row_height = max(28, table.fontMetrics().lineSpacing() + 12)
            table.setMinimumHeight(table.horizontalHeader().sizeHint().height() + 3 * row_height + 8)
            table.setMaximumHeight(max(300, 7 * row_height))
        panel.layout().activate()
        self.controls_scroll.setMinimumWidth(max(360, title_width + 64, panel.minimumSizeHint().width() + 24))
        self.adaptive_panes.update_layout()

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() == QEvent.FontChange:
            QTimer.singleShot(0, self._update_control_metrics)

    def showEvent(self, event):
        super().showEvent(event)
        fit_to_screen(self)
        handle = self.windowHandle()
        if handle and not getattr(self, "_screen_connected", False):
            handle.screenChanged.connect(lambda _screen: fit_to_screen(self))
            self._screen_connected = True

    # ── Table builders ─────────────────────────────────────────────────────

    def _rebuild_x_table(self, n=None):
        if n is None: n=self.spin_nx.value()
        self.x_calib_table.setRowCount(n)
        defs=["5","10","20","30","40","50","15","25","35","45"]
        for i in range(n):
            lbl=QTableWidgetItem(f"X{i+1}"); lbl.setFlags(Qt.ItemIsEnabled)
            self.x_calib_table.setItem(i,0,lbl)
            if self.x_calib_table.item(i,1) is None:
                self.x_calib_table.setItem(i,1,QTableWidgetItem(defs[i] if i<len(defs) else ""))
        if hasattr(self,"canvas"): self.canvas._x_target=n

    def _rebuild_y_table(self, n=None):
        if n is None: n=self.spin_ny.value()
        self.y_calib_table.setRowCount(n)
        defs=["0","20","40","60","80","100","25","50","75","10"]
        for i in range(n):
            lbl=QTableWidgetItem(f"Y{i+1}"); lbl.setFlags(Qt.ItemIsEnabled)
            self.y_calib_table.setItem(i,0,lbl)
            if self.y_calib_table.item(i,1) is None:
                self.y_calib_table.setItem(i,1,QTableWidgetItem(defs[i] if i<len(defs) else ""))
        if hasattr(self,"canvas"): self.canvas._y_target=n

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _update_status(self,msg): self.status.showMessage(msg)

    def _update_preset_summary(self):
        if not hasattr(self, "combo_extraction_preset"):
            return
        preset = extraction_preset(self.combo_extraction_preset.currentData())
        self.lbl_preset_summary.setText(self._tr(preset["summary"]))

    def _apply_extraction_preset(self, announce=True):
        preset_key = self.combo_extraction_preset.currentData()
        preset = extraction_preset(preset_key)
        self.chk_dark_auto_threshold.setChecked(preset["auto_threshold"])
        self.slider_dark.setValue(preset["threshold"])
        self.spin_step.setValue(preset["col_step"])
        self.spin_run.setValue(preset["min_run"])
        self.combo_dark_pick.setCurrentIndex(
            {"topmost": 0, "bottommost": 1, "center": 2}[preset["pick"]]
        )
        self.chk_dark_yband.setChecked(preset["y_band"])
        self.chk_dark_conn.setChecked(preset["connectivity"])
        self.spin_dark_maf.setValue(preset["min_area_frac"])
        self.spin_dark_mwf.setValue(preset["min_width_frac"])
        self.spin_dark_asp.setValue(preset["max_aspect"])
        self.spin_dark_topn.setValue(preset["top_n"])
        self.chk_dark_grid.setChecked(preset["suppress_grid"])
        strength_index = self.combo_dark_grid_strength.findData(
            preset["grid_strength"]
        )
        if strength_index >= 0:
            self.combo_dark_grid_strength.setCurrentIndex(strength_index)
        self.spin_maxpts.setValue(preset["max_points"])

        if preset["y_band"]:
            if self.canvas.roi is not None:
                y_min, y_max = self.canvas.roi[1], self.canvas.roi[3]
            elif self.canvas.image is not None:
                y_min, y_max = 0, self.canvas.image.shape[0]
            else:
                y_min, y_max = 0, 500
            self.spin_dark_ymin.setValue(max(0, round(y_min)))
            self.spin_dark_ymax.setValue(max(1, round(y_max)))

        self._update_preset_summary()
        if announce:
            self._update_status(
                f"Applied extraction preset: {self._tr(preset['label'])}"
            )

    def _show_extraction_instructions(self):
        dialog = QDialog(self)
        dialog.setWindowTitle(self._tr("Extraction Presets & Instructions…"))
        dialog.resize(
            min(820, round(self.screen().availableGeometry().width() * 0.75)),
            min(720, round(self.screen().availableGeometry().height() * 0.80)),
        )
        layout = QVBoxLayout(dialog)
        browser = QTextBrowser(dialog)
        browser.setHtml(extraction_instructions_html(self.language))
        layout.addWidget(browser, 1)
        buttons = QHBoxLayout(); buttons.addStretch()
        print_button = QPushButton(self._tr("Print…"))
        print_button.clicked.connect(
            lambda _checked=False: self._print_extraction_instructions(browser, dialog)
        )
        buttons.addWidget(print_button)
        close_button = QPushButton(self._tr("Close"))
        close_button.clicked.connect(dialog.accept); buttons.addWidget(close_button)
        layout.addLayout(buttons)
        dialog.exec()

    def _print_extraction_instructions(self, browser, parent):
        try:
            from PySide6.QtPrintSupport import QPrintDialog, QPrinter

            printer = QPrinter(QPrinter.HighResolution)
            printer.setDocName("PXRD Patent-Plot Extraction Instructions")
            print_dialog = QPrintDialog(printer, parent)
            if print_dialog.exec() == QDialog.Accepted:
                browser.document().print_(printer)
                self._update_status("Extraction instructions sent to printer.")
        except Exception as error:
            QMessageBox.critical(
                self,
                "Printing failed",
                f"The extraction instructions could not be printed.\n\n{error}",
            )

    def _update_calib_status(self):
        x="✓ X" if self.canvas._x_calib_ready else self._tr("⬤ X: not set")
        y="✓ Y" if self.canvas._y_calib_ready else "⬤ Y: not set"
        if self.language == "zh" and not self.canvas._y_calib_ready:
            y = "⬤ Y: 未设置"
        if self.canvas._x_calib_ready and self.canvas._y_calib_ready:
            self.lbl_calib.setText(self._tr("✓ X calibrated     ✓ Y calibrated — ready!"))
        else:
            self.lbl_calib.setText(f"{x}     {y}")

    def _refresh_table(self):
        self.table.setRowCount(0)
        for ds in self.canvas.datasets:
            for x,y in ds["pts_real"]:
                r=self.table.rowCount(); self.table.insertRow(r)
                self.table.setItem(r,0,QTableWidgetItem(ds["label"]))
                self.table.setItem(r,1,QTableWidgetItem(f"{x:.6g}"))
                self.table.setItem(r,2,QTableWidgetItem(f"{y:.6g}"))
        self.table.scrollToBottom()

    def _ok(self):
        if self.canvas.image is None:
            QMessageBox.warning(self,"No Image","Load an image first."); return False
        if not (self.canvas._x_calib_ready and self.canvas._y_calib_ready):
            QMessageBox.warning(self,"Not Calibrated",
                "Calibrate both X and Y axes first."); return False
        return True

    # ── ROI ─────────────────────────────────────────────────────────────────

    def _detect_roi(self):
        if self.canvas.image is None:
            QMessageBox.warning(self,"No Image","Load an image first.")
            return
        box = detect_plot_roi(self.canvas.image)
        if box is None:
            self._update_status(self._tr("No clear plot frame found. ROI unchanged; draw a box manually."))
            return
        self.canvas.mode = "none"
        self._eyedrop_active = False
        self.canvas._roi_click1 = None
        self.canvas._reject_click1 = None
        self.canvas.roi = box
        self.canvas.roi_updated.emit(*box)
        self.canvas._redraw()
        self._update_status(self._tr("Auto ROI set. Check the green box; use Draw ROI to adjust it."))

    def _start_roi(self):
        if self.canvas.image is None:
            QMessageBox.warning(self,"No Image","Load an image first."); return
        self.canvas.mode="roi"
        self._update_status(self._tr("ROI — click and drag to draw the plot area box, then release"))

    def _clear_roi(self):
        self.canvas.roi=None; self.canvas._roi_click1=None
        self.lbl_roi.setText(self._tr("No ROI — full image used"))
        self.canvas._redraw(); self._update_status(self._tr("ROI cleared"))

    def _on_roi_updated(self,x0,y0,x1,y1):
        if self.language == "zh":
            self.lbl_roi.setText(f"ROI: ({x0:.0f},{y0:.0f}) → ({x1:.0f},{y1:.0f})")
            self._update_status(f"ROI 已设置: ({x0:.0f},{y0:.0f})→({x1:.0f},{y1:.0f})  — 可开始提取")
        else:
            self.lbl_roi.setText(f"ROI: ({x0:.0f},{y0:.0f}) → ({x1:.0f},{y1:.0f})")
            self._update_status(f"ROI set: ({x0:.0f},{y0:.0f})→({x1:.0f},{y1:.0f})  — ready to extract")

    # ── Calibration ──────────────────────────────────────────────────────────

    def _run_auto_calibration(self):
        if self.canvas.image is None:
            QMessageBox.warning(self, self._tr("No Image"), self._tr("Load an image first.")); return
        dialog = AxisCalibrationDialog(self)
        try:
            if dialog.exec() == QDialog.Accepted and dialog.reviewed:
                try:
                    self._apply_reviewed_calibration(dialog.reviewed)
                except (ValueError, OverflowError) as error:
                    QMessageBox.warning(self, self._tr("Calibration Error"), str(error))
        finally:
            dialog.deleteLater()

    def _apply_reviewed_calibration(self, reviewed):
        # Validate both fits and all recalculated datasets before changing any
        # state. Cancellation/error leaves the existing calibration intact.
        axes = reviewed["axes"]
        coefficients = {axis: calibration_coefficients(axes[axis]["points"], axis, axes[axis]["scale"])
                        for axis in ("x", "y")}
        if self.canvas.image is None:
            raise ValueError("Load an image first.")
        height, width = self.canvas.image.shape[:2]
        for axis in ("x", "y"):
            if len(axes[axis]["points"]) > 15:
                raise ValueError("Choose at most 15 calibration points per axis.")
            for point in axes[axis]["points"]:
                px, py = point["pixel"]
                if not np.all(np.isfinite([px, py])) or not (0 <= px < width and 0 <= py < height):
                    raise ValueError("Calibration points must be finite and inside the image.")
        recalculated = []
        for dataset in self.canvas.datasets:
            real = []
            for px, py in dataset["pts_px"]:
                x = coefficients["x"][0] * px + coefficients["x"][1]
                y = coefficients["y"][0] * py + coefficients["y"][1]
                x = 10.0 ** x if axes["x"]["scale"] == "log" else x
                y = 10.0 ** y if axes["y"]["scale"] == "log" else y
                if not np.all(np.isfinite([x, y])):
                    raise ValueError("Calibration produced non-finite curve coordinates.")
                real.append([x, y])
            recalculated.append(real)
        mode = {("linear", "linear"): 0, ("linear", "log"): 1,
                ("log", "log"): 2, ("log", "linear"): 3}[(axes["x"]["scale"], axes["y"]["scale"])]
        self.combo_scale.setCurrentIndex(mode)
        self.canvas.scale_type = ("linear", "semilogy", "loglog", "semilogx")[mode]
        for axis, spin, table in (("x", self.spin_nx, self.x_calib_table), ("y", self.spin_ny, self.y_calib_table)):
            points = axes[axis]["points"]
            spin.setValue(len(points))
            setattr(self.canvas, axis + "_calib_pts", [list(point["pixel"]) for point in points])
            setattr(self.canvas, axis + "_calib_vals", [point["value"] for point in points])
            for row, point in enumerate(points):
                table.item(row, 1).setText(f'{point["value"]:g}')
        self.canvas._calib_ax, self.canvas._calib_bx = coefficients["x"]
        self.canvas._calib_ay, self.canvas._calib_by = coefficients["y"]
        self.canvas._x_calib_ready = self.canvas._y_calib_ready = True
        self.canvas.mode = "none"
        for dataset, real in zip(self.canvas.datasets, recalculated):
            dataset["pts_real"] = real
        self.canvas.clear_rejection_undo()
        self.lbl_x_calib.setText(self._tr("X calibrated ✓"))
        self.lbl_y_calib.setText(self._tr("Y calibrated ✓"))
        self._update_calib_status(); self.canvas._redraw(); self._refresh_table()
        self._update_status(self._tr("Reviewed axis calibration applied."))

    def _start_x_calibration(self):
        n=self.spin_nx.value(); vals=[]
        for i in range(n):
            xi=self.x_calib_table.item(i,1)
            try: vals.append(float(xi.text().strip()))
            except: QMessageBox.warning(self,"Input Error",f"X{i+1}: enter a number."); return
        self.canvas._x_target=n; self.canvas.x_calib_pts=[]; self.canvas.x_calib_vals=vals
        self.canvas.mode="calibrate_x"
        self.lbl_x_calib.setText("  ".join(["○"]*n)+f"  click X1")
        self._update_status(f"X CALIBRATE — click {n} X-axis tick positions on image")
        self.canvas._redraw()

    def _start_y_calibration(self):
        n=self.spin_ny.value(); vals=[]
        for i in range(n):
            yi=self.y_calib_table.item(i,1)
            try: vals.append(float(yi.text().strip()))
            except: QMessageBox.warning(self,"Input Error",f"Y{i+1}: enter a number."); return
        self.canvas._y_target=n; self.canvas.y_calib_pts=[]; self.canvas.y_calib_vals=vals
        self.canvas.mode="calibrate_y"
        self.lbl_y_calib.setText("  ".join(["○"]*n)+f"  click Y1")
        self._update_status(f"Y CALIBRATE — click {n} Y-axis tick positions on image")
        self.canvas._redraw()

    def _on_calib_click(self, px, py):
        if self.canvas.mode=="calibrate_x":
            nd=len(self.canvas.x_calib_pts); nt=self.canvas._x_target
            dots="  ".join("●" if i<nd else "○" for i in range(nt))
            if nd<nt:
                self.lbl_x_calib.setText(f"{dots}  {nd}/{nt} — click X{nd+1}")
            else:
                try:
                    self.canvas.fit_x_calib(self.canvas.x_calib_pts,self.canvas.x_calib_vals)
                except ValueError as error:
                    self.canvas.x_calib_pts.pop()
                    self.lbl_x_calib.setText(f"{nd-1}/{nt} — click X{nd}")
                    QMessageBox.warning(self, self._tr("Calibration incomplete"), str(error))
                    return
                self.canvas.mode="none"
                self.lbl_x_calib.setText(f"{dots}  X calibrated ✓")
                self._update_calib_status()
                self._update_status(f"X calibrated ✓ ({nt} pts, least-squares)")
        elif self.canvas.mode=="calibrate_y":
            nd=len(self.canvas.y_calib_pts); nt=self.canvas._y_target
            dots="  ".join("●" if i<nd else "○" for i in range(nt))
            if nd<nt:
                self.lbl_y_calib.setText(f"{dots}  {nd}/{nt} — click Y{nd+1}")
            else:
                try:
                    self.canvas.fit_y_calib(self.canvas.y_calib_pts,self.canvas.y_calib_vals)
                except ValueError as error:
                    self.canvas.y_calib_pts.pop()
                    self.lbl_y_calib.setText(f"{nd-1}/{nt} — click Y{nd}")
                    QMessageBox.warning(self, self._tr("Calibration incomplete"), str(error))
                    return
                self.canvas.mode="none"
                self.lbl_y_calib.setText(f"{dots}  Y calibrated ✓")
                self._update_calib_status()
                self._update_status(f"Y calibrated ✓ ({nt} pts, least-squares)")

    def _reset_calibration(self):
        self.canvas.x_calib_pts=[]; self.canvas.x_calib_vals=[]
        self.canvas.y_calib_pts=[]; self.canvas.y_calib_vals=[]
        self.canvas._x_calib_ready=False; self.canvas._y_calib_ready=False
        self.canvas.mode="none"
        self.lbl_x_calib.setText(self._tr("○ ○ ○  awaiting X points"))
        self.lbl_y_calib.setText(self._tr("○ ○ ○  awaiting Y points"))
        self.lbl_calib.setText(self._tr("⬤ X: not set     ⬤ Y: not set"))
        self.canvas._redraw(); self._update_status(self._tr("Calibration reset"))

    # ── Image / scale ────────────────────────────────────────────────────────

    def _open_image(self):
        path,_=QFileDialog.getOpenFileName(self,self._tr("Open Plot Image"),"",
            self._tr("Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.gif);;All (*)"))
        if path:
            self._image_path = path
            self.canvas.load_image(path)
            self.lbl_calib.setText(self._tr("⬤ X: not set     ⬤ Y: not set"))
            self.lbl_x_calib.setText(self._tr("○ ○ ○  awaiting X points"))
            self.lbl_y_calib.setText(self._tr("○ ○ ○  awaiting Y points"))
            self.lbl_roi.setText(self._tr("No ROI — full image used"))
            self.table.setRowCount(0)
            self._update_status(f"Loaded: {path}")
            if self.chk_auto_roi.isChecked():
                self._detect_roi()

    def _scale_changed(self,idx):
        self.canvas.scale_type=("linear","semilogy","loglog","semilogx")[idx]
        self._update_status(self._tr("Scale: ")+self.combo_scale.currentText())

    # ── Manual digitize ──────────────────────────────────────────────────────

    def _start_digitize(self):
        if not self._ok(): return
        self.canvas.mode="digitize"
        self._update_status(f"DIGITIZE | {self.canvas.datasets[self.canvas.active_ds]['label']}")

    def _start_reject_box(self):
        if self.canvas.image is None:
            QMessageBox.warning(self,"No Image","Load an image first."); return
        dataset = self.canvas.datasets[self.canvas.active_ds]
        if not dataset["pts_px"]:
            QMessageBox.information(
                self,
                "No Points",
                "The active dataset has no points to reject.",
            )
            return
        self.canvas.mode = "reject"
        self.canvas._reject_click1 = None
        self._update_status(
            f"REJECT POINTS | {dataset['label']} — drag a box around unwanted points; Ctrl+Z restores them"
        )

    def _on_points_rejected(self, count, x0, y0, x1, y1):
        self._refresh_table()
        dataset = self.canvas.datasets[self.canvas.active_ds]
        if count:
            self._update_status(
                f"Rejected {count} points from {dataset['label']} in box "
                f"({x0:.0f}, {y0:.0f})–({x1:.0f}, {y1:.0f}); Ctrl+Z to restore"
            )
        else:
            self._update_status(
                f"No points from {dataset['label']} were inside the selected box."
            )

    def _undo_point(self):
        if self.canvas.remove_last_point():
            self._refresh_table()
            if self.canvas.last_undo_kind == "box_rejection":
                self._update_status(
                    f"Restored {self.canvas.last_undo_count} box-rejected points"
                )
            else:
                self._update_status(self._tr("Last point removed"))

    def _clear_dataset(self):
        ds=self.canvas.datasets[self.canvas.active_ds]
        ds["pts_px"]=[]; ds["pts_real"]=[]
        self.canvas.clear_rejection_undo()
        self.canvas._redraw(); self._refresh_table(); self._update_status(f"Cleared: {ds['label']}")

    def _delete_dataset(self):
        idx=self.canvas.active_ds
        if len(self.canvas.datasets)<=1:
            QMessageBox.information(self,"Cannot Delete","At least one dataset must remain."); return
        ds=self.canvas.datasets[idx]
        if QMessageBox.question(self,"Delete Dataset",
            f"Delete <b>{ds['label']}</b> and all its points?",
            QMessageBox.Yes|QMessageBox.No,QMessageBox.No)!=QMessageBox.Yes: return
        self.canvas.datasets.pop(idx)
        self.canvas.clear_rejection_undo()
        for combo in (self.combo_ds,self.combo_ds_auto):
            combo.blockSignals(True); combo.removeItem(idx); combo.blockSignals(False)
        new_idx=max(0,idx-1); self.canvas.active_ds=new_idx
        for combo in (self.combo_ds,self.combo_ds_auto):
            combo.blockSignals(True); combo.setCurrentIndex(new_idx); combo.blockSignals(False)
        self.canvas._redraw(); self._refresh_table()

    def _on_data_click(self,x,y):
        self._refresh_table()
        ds=self.canvas.datasets[self.canvas.active_ds]
        self._update_status(f"DIGITIZE | {ds['label']} | pt {len(ds['pts_real'])}: ({x:.4g},{y:.4g})")

    def _add_dataset(self):
        n=len(self.canvas.datasets)+1; lbl=f"Dataset {n}"
        self.canvas.datasets.append({"label":lbl,"marker":"s","pts_px":[],"pts_real":[]})
        self.combo_ds.addItem(lbl); self.combo_ds_auto.addItem(lbl)
        self.combo_ds.setCurrentIndex(n-1); self.combo_ds_auto.setCurrentIndex(n-1)

    def _ds_changed(self,idx):
        if idx < 0 or idx >= len(self.canvas.datasets):
            return
        self.canvas.active_ds=idx
        self.canvas.clear_rejection_undo()
        if self.sender() is self.combo_ds:
            self.combo_ds_auto.blockSignals(True); self.combo_ds_auto.setCurrentIndex(idx); self.combo_ds_auto.blockSignals(False)
        else:
            self.combo_ds.blockSignals(True); self.combo_ds.setCurrentIndex(idx); self.combo_ds.blockSignals(False)

    # ── Color eyedrop ────────────────────────────────────────────────────────

    def _pick_color_dialog(self):
        r,g,b=self.color_swatch.get_rgb()
        col=QColorDialog.getColor(QColor(r,g,b),self,"Pick Curve Color")
        if col.isValid(): self.color_swatch.set_color((col.red(),col.green(),col.blue()))

    def _start_eyedrop(self):
        if self.canvas.image is None: QMessageBox.warning(self,"No Image","Load an image first."); return
        self._eyedrop_active=True; self.canvas.mode="none"
        self._update_status("EYEDROP — click a pixel on the curve")

    def _eyedrop_click(self,event):
        if not self._eyedrop_active: return
        if event.inaxes!=self.canvas.ax or self.canvas.image is None: return
        px=max(0,min(self.canvas.image.shape[1]-1,int(event.xdata)))
        py=max(0,min(self.canvas.image.shape[0]-1,int(event.ydata)))
        rgb=tuple(int(c) for c in self.canvas.image[py,px,:3])
        self.color_swatch.set_color(rgb); self._eyedrop_active=False
        self._update_status(f"Color sampled: RGB{rgb}")

    # ── Auto extraction ──────────────────────────────────────────────────────

    def _run_autopilot(self):
        if not self._ok():
            return
        self._eyedrop_active = False
        self.canvas.mode = "none"
        dialog = CurvePickerDialog(self)
        try:
            if dialog.exec() == QDialog.Accepted:
                try:
                    self._import_autopilot_curves(dialog.result_data, dialog.selected_indices)
                except (ValueError, OverflowError) as error:
                    QMessageBox.warning(self, self._tr("Could not add curves"), str(error))
        finally:
            dialog.deleteLater()

    def _import_autopilot_curves(self, result, indices):
        if not indices or not self._ok():
            return
        # Prepare every conversion before mutating datasets or extraction settings.
        prepared = []
        preset = extraction_preset(result["preset"])
        labels = {dataset["label"] for dataset in self.canvas.datasets}
        for index in indices:
            candidate = result["candidates"][index]
            points = subsample(candidate["points"], preset["max_points"])
            real = [list(self.canvas._px2data(*point)) for point in points]
            if not np.all(np.isfinite(real)):
                raise ValueError("Calibration produced non-finite curve coordinates.")
            label = self._tr("Curve {number}").format(number=index + 1) + f" ({self._tr(candidate['color'])})"
            unique, suffix = label, 2
            while unique in labels:
                unique = f"{label} {suffix}"
                suffix += 1
            labels.add(unique)
            prepared.append({"label": unique, "marker": "o", "pts_px": [list(point) for point in points], "pts_real": real})
        self.combo_extraction_preset.setCurrentIndex(self.combo_extraction_preset.findData(result["preset"]))
        self._apply_extraction_preset(announce=False)
        self.slider_dark.setValue(result["threshold"])
        self.chk_dark_auto_threshold.setChecked(False)
        self.chk_dark_grid.setChecked(result["suppress_grid"])
        parameters = result.get("parameters", {})
        pick = (result["candidates"][indices[0]].get("pick", parameters.get("pick", "topmost"))
                if len(indices) == 1 else parameters.get("pick", "topmost"))
        self.combo_dark_pick.setCurrentIndex({"topmost": 0, "bottommost": 1, "center": 2}[pick])
        self.spin_step.setValue(parameters.get("col_step", 1))
        strength = parameters.get("grid_strength", "strong" if result["suppress_grid"] else "standard")
        self.combo_dark_grid_strength.setCurrentIndex(self.combo_dark_grid_strength.findData(strength))
        self.chk_dark_yband.setChecked(False)
        if result["inferred_roi"]:
            self.canvas.roi = result["roi"]
            self._on_roi_updated(*self.canvas.roi)
        for dataset in prepared:
            self.canvas.datasets.append(dataset)
            self.combo_ds.addItem(dataset["label"])
            self.combo_ds_auto.addItem(dataset["label"])
        self.combo_ds.setCurrentIndex(len(self.canvas.datasets) - 1)
        self.canvas.clear_rejection_undo()
        self.canvas._redraw()
        self._refresh_table()
        self._update_status(self._tr("Autopilot added {count} curves as new datasets · Preset: {preset}").format(count=len(prepared), preset=self._tr(preset["label"])))

    def _run_auto(self,pts_px,label):
        pts_px=subsample(pts_px,self.spin_maxpts.value())
        added=self.canvas.bulk_add_pixels(pts_px)
        self.progress.setValue(100); QApplication.processEvents()
        self.progress.setVisible(False); self._refresh_table()
        self._update_status(f"{label}: {len(pts_px)} pts → {added} converted")
        if added==0: QMessageBox.information(self,"No Points Found","Adjust parameters or check calibration.")

    def _run_color(self):
        if not self._ok(): return
        rgb=self.color_swatch.get_rgb(); tol=self.slider_tol.value()
        step=self.spin_col_step_color.value(); smooth=self.spin_smooth.value(); mina=self.spin_min_area.value()
        idx=self.combo_color_pick.currentIndex()
        pick=["topmost","bottommost","center"][idx]
        roi=self.canvas.roi
        conn=self.chk_color_conn.isChecked()
        yb=None
        if self.chk_color_yband.isChecked():
            ym0=self.spin_color_ymin.value(); ym1=self.spin_color_ymax.value()
            if roi is not None:
                ym0=max(0,ym0-int(roi[1])); ym1=max(0,ym1-int(roi[1]))
            yb=(ym0,ym1)
        self.progress.setVisible(True); self.progress.setValue(20); QApplication.processEvents()
        pts=extract_by_color(self.canvas.image,rgb,tol,col_step=step,
                              smooth_window=smooth,min_area=mina,roi=roi,pick=pick,
                              connectivity=conn,
                              min_area_frac=self.spin_color_maf.value(),
                              min_width_frac=self.spin_color_mwf.value(),
                              max_aspect=self.spin_color_asp.value(),
                              top_n=self.spin_color_topn.value(),
                              y_band=yb)
        self.progress.setValue(80); QApplication.processEvents()
        self._run_auto(pts,f"Color extract (pick={pick}, conn={conn})")

    def _run_dark(self):
        if not self._ok(): return
        idx=self.combo_dark_pick.currentIndex()
        pick=["topmost","bottommost","center"][idx]
        roi=self.canvas.roi
        conn=self.chk_dark_conn.isChecked()
        # Build y_band in ROI-crop-relative coords if enabled
        yb=None
        if self.chk_dark_yband.isChecked():
            ym0=self.spin_dark_ymin.value(); ym1=self.spin_dark_ymax.value()
            if roi is not None:
                ym0=max(0,ym0-int(roi[1])); ym1=max(0,ym1-int(roi[1]))
            yb=(ym0,ym1)
        self.progress.setVisible(True); self.progress.setValue(20); QApplication.processEvents()
        arguments = dict(
            col_step=self.spin_step.value(), roi=roi,
            pick=pick, min_run=self.spin_run.value(),
            connectivity=conn,
            min_area_frac=self.spin_dark_maf.value(),
            min_width_frac=self.spin_dark_mwf.value(),
            max_aspect=self.spin_dark_asp.value(),
            top_n=self.spin_dark_topn.value(),
            y_band=yb,
            suppress_grid=self.chk_dark_grid.isChecked(),
            grid_strength=self.combo_dark_grid_strength.currentData(),
        )
        if self.chk_dark_auto_threshold.isChecked():
            pts, selected_threshold = extract_dark_curve_auto(
                self.canvas.image, **arguments
            )
            self.slider_dark.setValue(selected_threshold)
        else:
            selected_threshold = self.slider_dark.value()
            pts=extract_dark_curve(
                self.canvas.image, threshold=selected_threshold, **arguments
            )
        self.progress.setValue(80); QApplication.processEvents()
        self._run_auto(
            pts,
            f"Dark curve (threshold={selected_threshold}, pick={pick}, conn={conn}, "
            f"grid={self.combo_dark_grid_strength.currentData()})",
        )

    def _run_blob(self):
        if not self._ok(): return
        roi=self.canvas.roi
        self.progress.setVisible(True); self.progress.setValue(20); QApplication.processEvents()
        pts=extract_blobs(self.canvas.image,float(self.spin_bmin.value()),
                           float(self.spin_bmax.value()),self.spin_circ.value(),roi=roi)
        self.progress.setValue(80); QApplication.processEvents()
        self._run_auto(pts,"Blob detect")

    # ── Export ───────────────────────────────────────────────────────────────

    def _collect_all(self):
        return [{"dataset":ds["label"],"x":x,"y":y}
                for ds in self.canvas.datasets for x,y in ds["pts_real"]]

    def _export_csv(self):
        rows=self._collect_all()
        if not rows: QMessageBox.information(self,"No data","No points to export."); return
        path,_=QFileDialog.getSaveFileName(self,"Save CSV","digitized_data.csv","CSV (*.csv)")
        if path:
            with open(path,"w",newline="") as f:
                w=csv.DictWriter(f,fieldnames=["dataset","x","y"]); w.writeheader(); w.writerows(rows)
            QMessageBox.information(self,"Exported",f"Saved {len(rows)} points to:\n{path}")

    def _export_json(self):
        rows=self._collect_all()
        if not rows: QMessageBox.information(self,"No data","No points to export."); return
        path,_=QFileDialog.getSaveFileName(self,"Save JSON","digitized_data.json","JSON (*.json)")
        if path:
            grouped={}
            for r in rows: grouped.setdefault(r["dataset"],[]).append({"x":r["x"],"y":r["y"]})
            with open(path,"w") as f: json.dump(grouped,f,indent=2)
            QMessageBox.information(self,"Exported",f"Saved to:\n{path}")

    def _auto_baseline(self):
        """Set Y-offset spinbox to 5th-percentile of active dataset intensities."""
        ds = self.canvas.datasets[self.canvas.active_ds]
        if not ds["pts_real"]:
            QMessageBox.information(self, "No data", "Digitize some points first."); return
        ys = np.array([p[1] for p in ds["pts_real"]])
        baseline = float(np.percentile(ys, 5))
        self.spin_yoffset.setValue(round(baseline, 2))
        self._update_status(
            f"Auto baseline: {baseline:.2f}  (5th percentile of {len(ys)} points)")

    def set_reference_context(self, projects, initial_project=""):
        """Refresh projects when reopened from the CrystalTrace catalog."""
        current = self.reference_project_combo.currentText()
        values = sorted({
            str(project).strip() for project in projects if str(project).strip()
        })
        self.reference_project_combo.clear()
        self.reference_project_combo.addItems(values)
        preferred = str(initial_project).strip()
        if preferred in values:
            self.reference_project_combo.setCurrentText(preferred)
        elif current in values:
            self.reference_project_combo.setCurrentText(current)
        self.add_reference_button.setEnabled(
            self._add_reference_callback is not None and bool(values))

    def _active_reference_points(self):
        ds = self.canvas.datasets[self.canvas.active_ds]
        return prepare_reference_points(
            ds["pts_real"],
            baseline=self.spin_yoffset.value(),
            scale=self.spin_yscale.value(),
        )

    def _add_active_dataset_as_reference(self):
        if self._add_reference_callback is None:
            QMessageBox.information(
                self, "Reference database", "Open Plot Digitizer from CrystalTrace."
            )
            return
        project = self.reference_project_combo.currentText().strip()
        phase = self.reference_phase_edit.text().strip()
        if not project or not phase:
            QMessageBox.information(
                self, "Reference assignment", "Select a project and enter a phase name."
            )
            return
        points = self._active_reference_points()
        if len(points) < 2:
            QMessageBox.information(
                self,
                "No digitized data",
                "Extract or manually digitize at least two points first.",
            )
            return
        dataset = self.canvas.datasets[self.canvas.active_ds]
        image_source = self._image_path or "unsaved_patent_image"
        source_id = (
            f"{image_source}#digitized={dataset['label']};"
            "representation=full-digitized-curve"
        )
        try:
            self._add_reference_callback(
                project,
                phase,
                source_id,
                [point[0] for point in points],
                [point[1] for point in points],
            )
        except Exception as error:
            QMessageBox.critical(self, "Reference import failed", str(error))
            return
        self._update_status(
            f"Reference added: {project} / {phase} ({len(points)} points)"
        )
        QMessageBox.information(
            self,
            "Reference added",
            f"Added {dataset['label']} to {project} as {phase}.\n"
            f"{len(points)} digitized points were stored in the database.",
        )

    def _export_xrdml(self):
        """Export the active dataset as a HighScore-compatible XRDML file."""
        ds = self.canvas.datasets[self.canvas.active_ds]
        if not ds["pts_real"]:
            QMessageBox.information(self, "No data",
                "No digitized points in the active dataset.\n"
                "Run an extraction or manually digitize some points first.")
            return
        pts = sorted(ds["pts_real"], key=lambda p: p[0])
        t_arr = np.array([p[0] for p in pts])
        y_arr = np.array([p[1] for p in pts])

        path, _ = QFileDialog.getSaveFileName(
            self, "Save XRDML", ds["label"].replace(" ", "_") + ".xrdml",
            "XRDML files (*.xrdml);;All files (*)")
        if not path:
            return
        try:
            export_xrdml(
                path, t_arr, y_arr,
                dataset_label=ds["label"],
                wavelength_ka1=self.spin_ka1.value(),
                wavelength_ka2=self.spin_ka2.value(),
                generator_voltage=self.spin_kv.value(),
                generator_current=self.spin_ma.value(),
                counting_time=self.spin_ct.value(),
                y_scale=self.spin_yscale.value(),
                y_baseline=self.spin_yoffset.value(),
                comment=f"Digitized from image by PlotDigitizer — dataset: {ds['label']}")
            n = len(pts)
            self._update_status(f"XRDML exported: {n} pts → {path}")
            QMessageBox.information(self, "Exported",
                f"Saved {n} data points to:\n{path}\n\n"
                "Open in HighScore / HighScore Plus via File > Open.")
        except Exception as e:
            QMessageBox.critical(self, "Export failed", str(e))

    def _about(self):
        QMessageBox.about(self,"PlotDigitizer",
            "<b>PlotDigitizer</b> — Separated Axis Calibration + ROI<br><br>"
            "<b>Key fixes:</b><br>"
            "• Topmost pixel pick → no downward bias from averaging<br>"
            "• ROI box → no stray axis/border/label points<br>"
            "• X and Y calibrated independently → correct even for stacked panels<br>"
            "• Normal, Grid, and Waterfall extraction presets<br>"
            "• Autopilot with visual curve selection and color filtering<br>"
            "• Drag-box rejection with grouped Ctrl+Z restoration<br><br>"
            "<b>Recommended workflow for black-on-white PXRD:</b><br>"
            "1. Open an image; inspect the automatic ROI or draw your own box<br>"
            "2. Calibrate X (click tick marks) and Y (click tick marks)<br>"
            "3. Click Find all curves, then choose; add the desired trace<br>"
            "4. Inspect the points and reject false points by box<br><br>"
            "Ctrl+Z = undo the last point or restore the last rejected group")


def main(standalone=True):
    # Let Qt apply the platform's scale factor once.  The old code also
    # multiplied the application font and panel geometry by logical DPI,
    # which made the standalone window too large on lower-resolution screens.
    try:
        QApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )
    except AttributeError:
        pass
    app = QApplication(sys.argv)
    app.setApplicationName("PlotDigitizer")
    app.setStyle("Fusion")
    configure_application_appearance(app)
    win = MainWindow(standalone=standalone)
    win.show()
    sys.exit(app.exec())

if __name__=="__main__":
    main()
