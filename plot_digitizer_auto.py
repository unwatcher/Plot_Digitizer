#!/usr/bin/env python3
"""
PlotDigitizer GUI — PyQt5 + Matplotlib + OpenCV
pip install pyqt5 matplotlib numpy opencv-python

Calibration: X and Y axes calibrated SEPARATELY via independent linear fits.
Extraction: ROI crop prevents picking up axes/labels/borders.
Dark-curve mode: connectivity filter removes label blobs; topmost pixel of curve component.
Color mode:      connectivity filter + topmost pixel; both avoid label/text contamination.
"""

import sys, csv, json, math
import numpy as np
import cv2

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QLineEdit, QTableWidget,
    QTableWidgetItem, QGroupBox, QSplitter, QComboBox, QStatusBar,
    QMessageBox, QAction, QActionGroup, QHeaderView, QAbstractItemView, QSizePolicy,
    QSlider, QSpinBox, QDoubleSpinBox, QTabWidget, QScrollArea,
    QColorDialog, QFrame, QProgressBar, QCheckBox,
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.image as mpimg
import matplotlib.patches as mpatches


# ══════════════════════════════════════════════════════════════════════════════
#  Extraction algorithms
# ══════════════════════════════════════════════════════════════════════════════

def _build_color_mask(img_rgb, target_rgb, tolerance):
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    t = np.uint8([[[target_rgb[2], target_rgb[1], target_rgb[0]]]])
    th = cv2.cvtColor(t, cv2.COLOR_BGR2HSV)[0][0]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = int(th[0]), int(th[1]), int(th[2])
    lo = np.array([max(0,   h-tolerance), max(0,  s-60), max(0,  v-60)])
    hi = np.array([min(180, h+tolerance), min(255,s+60), min(255,v+60)])
    mask = cv2.inRange(hsv, lo, hi)
    if h < tolerance:
        lo2=np.array([180-(tolerance-h), max(0,s-60), max(0,v-60)])
        hi2=np.array([180, min(255,s+60), min(255,v+60)])
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo2, hi2))
    if h > 180-tolerance:
        lo2=np.array([0, max(0,s-60), max(0,v-60)])
        hi2=np.array([tolerance, min(255,s+60), min(255,v+60)])
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo2, hi2))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
    return mask


def _keep_curve_components(mask, min_area_frac=0.005, max_aspect=8.0,
                            min_width_frac=0.05, top_n=5):
    """
    Filter a binary mask to retain only components that look like a curve,
    discarding text labels, tick marks, and small noise.

    Strategy:
      1. Find all connected components.
      2. Score each by area and horizontal span (width).
      3. Keep the top_n largest components that also pass:
           - area >= min_area_frac * total_mask_area
           - horizontal width >= min_width_frac * image_width   (rejects narrow letters)
           - bounding-box aspect ratio (w/h) >= 1/max_aspect    (rejects tall thin text strokes)
      4. Merge them back into one mask.

    This robustly separates the continuous curve (large, wide) from peak labels
    (many small, narrow, tall isolated blobs).
    """
    h_img, w_img = mask.shape
    total_area = max(1, np.count_nonzero(mask))
    n_lab, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

    candidates = []
    for i in range(1, n_lab):
        area   = int(stats[i, cv2.CC_STAT_AREA])
        bw     = int(stats[i, cv2.CC_STAT_WIDTH])
        bh     = int(stats[i, cv2.CC_STAT_HEIGHT])
        if area < max(5, min_area_frac * total_area): continue
        if bw   < min_width_frac * w_img:             continue
        aspect = bw / max(1, bh)
        if aspect < 1.0 / max_aspect:                 continue   # too tall/thin → text
        candidates.append((area, i))

    candidates.sort(reverse=True)
    keep_ids = {idx for _, idx in candidates[:top_n]}

    clean = np.zeros_like(mask)
    for idx in keep_ids:
        clean[labels == idx] = 255
    return clean


def _mask_to_curve_pts(mask, col_step, pick, smooth_window, ox, oy,
                       y_band=None):
    """
    Convert a filtered binary mask to (pixel_x, pixel_y) points.

    pick options:
      'topmost'    - topmost (min row) dark pixel → upper curve in a stack
      'bottommost' - bottommost (max row) dark pixel → lower curve in a stack
      'center'     - weighted centroid

    y_band: (y_min_px, y_max_px) in CROP coordinates.
      Only pixels within this row range are considered.
      Use this to isolate a specific curve in a waterfall plot without
      needing a separate ROI box — set the band to the vertical extent
      of the target curve's baseline-to-top range.
    """
    img_h, img_w = mask.shape
    hw = max(0, smooth_window // 2)
    pts = []
    for x in range(0, img_w, col_step):
        x0b = max(0, x-hw); x1b = min(img_w, x+hw+1)
        band = mask[:, x0b:x1b]
        row_counts = band.sum(axis=1)
        # Apply y_band restriction BEFORE picking
        if y_band is not None:
            ylo = max(0, int(y_band[0])); yhi = min(img_h, int(y_band[1]))
            outside = np.ones(img_h, dtype=bool)
            outside[ylo:yhi] = False
            row_counts[outside] = 0
        nonzero = np.where(row_counts > 0)[0]
        if nonzero.size == 0: continue
        if pick == "topmost":
            cy = float(nonzero[0])
        elif pick == "bottommost":
            cy = float(nonzero[-1])
        else:
            cy = float(np.average(nonzero, weights=row_counts[nonzero]))
        pts.append((float(x + ox), cy + oy))
    return pts


def extract_by_color(img_rgb, target_rgb, tolerance=25, col_step=1,
                     smooth_window=3, min_area=10, roi=None,
                     pick="topmost", connectivity=True,
                     min_area_frac=0.005, min_width_frac=0.05,
                     max_aspect=8.0, top_n=5, y_band=None):
    """
    Extract curve points by color matching.
    connectivity=True  → filter out text/label blobs before column scan.
    y_band: (y_min, y_max) in crop-relative pixel rows — restricts which
            rows are scanned; isolates a specific curve in a waterfall plot.
    """
    if roi is not None:
        x0,y0,x1,y1 = [int(round(v)) for v in roi]
        x0=max(0,x0); y0=max(0,y0)
        x1=min(img_rgb.shape[1],x1); y1=min(img_rgb.shape[0],y1)
        crop = img_rgb[y0:y1, x0:x1]; ox,oy = x0, y0
    else:
        crop = img_rgb; ox,oy = 0,0

    mask = _build_color_mask(crop, target_rgb, tolerance)

    if min_area > 1:
        n_lab, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        clean = np.zeros_like(mask)
        for i in range(1, n_lab):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                clean[labels == i] = 255
        mask = clean

    if connectivity:
        mask = _keep_curve_components(mask, min_area_frac=min_area_frac,
                                       min_width_frac=min_width_frac,
                                       max_aspect=max_aspect, top_n=top_n)

    return _mask_to_curve_pts(mask, col_step, pick, smooth_window, ox, oy,
                              y_band=y_band)


def _dark_mask(gray, threshold):
    """Return binary mask of pixels darker than threshold."""
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    return mask


def extract_dark_curve(img_rgb, threshold=128, col_step=1, roi=None,
                       pick="topmost", min_run=2,
                       connectivity=True,
                       min_area_frac=0.003, min_width_frac=0.05,
                       max_aspect=8.0, top_n=5, y_band=None):
    """
    Extract a dark curve from a light background.

    connectivity=True  → uses connected-component filtering to remove
                         peak labels, tick marks, and isolated text blobs
                         BEFORE the column scan, so only the continuous
                         curve contributes data points.
    pick='topmost'     → topmost surviving pixel per column (curve top edge).
    pick='center'      → vertical centroid per column (thicker strokes).
    min_run            → min consecutive dark pixels needed (noise filter,
                         only active when connectivity=False).
    """
    if roi is not None:
        x0,y0,x1,y1 = [int(round(v)) for v in roi]
        x0=max(0,x0); y0=max(0,y0)
        x1=min(img_rgb.shape[1],x1); y1=min(img_rgb.shape[0],y1)
        crop = img_rgb[y0:y1, x0:x1]; ox,oy = x0, y0
    else:
        crop = img_rgb; ox,oy = 0,0

    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)

    if connectivity:
        # Build full dark mask, filter by connectivity, then column-scan
        mask = _dark_mask(gray, threshold)
        # Dilate slightly to reconnect thin curve strokes broken at anti-aliasing
        k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        mask = cv2.dilate(mask, k3, iterations=1)
        mask = _keep_curve_components(mask, min_area_frac=min_area_frac,
                                       min_width_frac=min_width_frac,
                                       max_aspect=max_aspect, top_n=top_n)
        # Erode back to undo dilation so we don't shift peak tops downward
        mask = cv2.erode(mask, k3, iterations=1)
        return _mask_to_curve_pts(mask, col_step, pick, smooth_window=1, ox=ox, oy=oy,
                              y_band=y_band)
    else:
        # Legacy column-by-column scan with min_run filter
        h, w = gray.shape
        pts = []
        for x in range(0, w, col_step):
            col  = gray[:, x]
            dark = np.where(col < threshold)[0]
            if dark.size < min_run: continue
            if pick == "topmost":
                found = None; run = 0
                for r in dark:
                    if run == 0 or r == dark[run-1]+1:
                        run += 1
                        if run >= min_run: found = dark[run-min_run]; break
                    else: run = 1
                if found is None: found = dark[0]
                pts.append((float(x+ox), float(found)+oy))
            else:
                pts.append((float(x+ox), float(np.mean(dark))+oy))
        return pts


def extract_blobs(img_rgb, min_area=15.0, max_area=3000.0, min_circ=0.4, roi=None):
    if roi is not None:
        x0,y0,x1,y1 = [int(round(v)) for v in roi]
        x0=max(0,x0); y0=max(0,y0)
        x1=min(img_rgb.shape[1],x1); y1=min(img_rgb.shape[0],y1)
        img_rgb = img_rgb[y0:y1, x0:x1]; ox,oy = x0,y0
    else:
        ox,oy = 0,0
    gray = cv2.cvtColor(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    p = cv2.SimpleBlobDetector_Params()
    p.filterByArea=True; p.minArea=min_area; p.maxArea=max_area
    p.filterByCircularity=True; p.minCircularity=min_circ
    p.filterByConvexity=False; p.filterByInertia=False
    det = cv2.SimpleBlobDetector_create(p)
    kps = det.detect(binary)
    return sorted([(kp.pt[0]+ox, kp.pt[1]+oy) for kp in kps], key=lambda p: p[0])


def subsample(pts, n):
    if len(pts) <= n: return pts
    idx = np.round(np.linspace(0, len(pts)-1, n)).astype(int)
    return [pts[i] for i in idx]

def export_xrdml(path, two_theta_arr, intensity_arr, dataset_label="Digitized",
                 wavelength_ka1=1.5405980, wavelength_ka2=1.5444260,
                 wavelength_kb=1.3922500,
                 generator_voltage=45.0, generator_current=40.0,
                 counting_time=1.0, y_scale=1.0,
                 comment="Digitized by PlotDigitizer"):
    """Write HighScore-compatible XRDML v2.2 matching real Empyrean output.
    counting_time: HighScore divides raw counts by this to display cps.
      Set to 1.0 if Y axis is already in cps (default).
      Set to actual dwell time T if Y axis is in raw counts.
    y_scale: multiplied into all intensities before writing.
      Use 1/inflation_factor to correct any observed scaling error.
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
    if np.min(y) < 0:
        y = y - np.min(y)
    y = y * float(y_scale)
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

    # ── PIXcel scanning normalisation correction ──────────────────────────
    # HighScore always applies the area-detector scanning formula regardless
    # of declared detector type:
    #   I_display = raw_counts * active_window_deg / (countingTime * step_deg)
    # To recover the true intensity, pre-divide the counts by the same factor
    # so it cancels when HighScore multiplies it back:
    #   counts_written = real_counts * step_deg / active_window_deg
    #   I_display = counts_written * active_window_deg / (countingTime * step_deg)
    #             = real_counts / countingTime  (correct for countingTime=1)
    # PIXcel3D: 255 channels x 0.055 mm pitch / 240 mm radius = 3.348 deg window
    PIXCEL_ACTIVE_DEG = 255 * 0.055 / 240.0 * (180.0 / math.pi)
    step_deg    = (end2t - start2t) / max(n - 1, 1)
    ct_val      = float(counting_time)
    norm_factor = PIXCEL_ACTIVE_DEG / (ct_val * step_deg) if step_deg > 0 else 1.0
    y_corrected = np.maximum(0, np.round(y_int.astype(float) / norm_factor).astype(int))

    el(dp, "commonCountingTime", "%.4f" % ct_val, unit="seconds")
    el(dp, "counts", " ".join(str(v) for v in y_corrected), unit="counts")

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
    calibration_click = pyqtSignal(float, float)
    datapoint_click   = pyqtSignal(float, float)
    roi_updated       = pyqtSignal(float, float, float, float)   # x0,y0,x1,y1
    DS_COLORS = ["#00d4ff","#ff9800","#4caf50","#e91e63",
                 "#9c27b0","#ffeb3b","#ff5722","#00bcd4"]

    def __init__(self, parent=None):
        self.fig = Figure(facecolor="#1e1e1e")
        self.ax  = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
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

        self.datasets = [{"label":"Dataset 1","marker":"o","pts_px":[],"pts_real":[]}]
        self.active_ds = 0
        self.mpl_connect("button_press_event",  self._on_press)
        self.mpl_connect("button_release_event",self._on_release)
        self._style()

    def _style(self):
        self.ax.set_facecolor("#111111")
        self.ax.tick_params(colors="#888888")
        for sp in self.ax.spines.values(): sp.set_color("#444444")
        self.fig.tight_layout(pad=0.3)

    def load_image(self, path):
        raw = mpimg.imread(path)
        if raw.dtype != np.uint8:
            raw = (np.clip(raw,0,1)*255).astype(np.uint8)
        if raw.ndim==2: raw=np.stack([raw]*3,axis=-1)
        elif raw.shape[2]==4: raw=raw[:,:,:3]
        self.image = raw
        self.roi = None; self._roi_click1 = None
        self.x_calib_pts=[]; self.x_calib_vals=[]
        self.y_calib_pts=[]; self.y_calib_vals=[]
        self._x_calib_ready=False; self._y_calib_ready=False
        for ds in self.datasets: ds["pts_px"]=[]; ds["pts_real"]=[]
        self._redraw()

    def fit_x_calib(self, pts_px, vals_real):
        tx = lambda v: math.log10(v) if self.scale_type=="loglog" else v
        pxs = np.array([p[0] for p in pts_px])
        rvs = np.array([tx(v) for v in vals_real])
        A = np.column_stack([pxs, np.ones_like(pxs)])
        res,_,_,_ = np.linalg.lstsq(A, rvs, rcond=None)
        self._calib_ax, self._calib_bx = res[0], res[1]
        self._x_calib_ready = True

    def fit_y_calib(self, pts_px, vals_real):
        ty = lambda v: math.log10(v) if self.scale_type in ("semilogy","loglog") else v
        pys = np.array([p[1] for p in pts_px])
        rvs = np.array([ty(v) for v in vals_real])
        A = np.column_stack([pys, np.ones_like(pys)])
        res,_,_,_ = np.linalg.lstsq(A, rvs, rcond=None)
        self._calib_ay, self._calib_by = res[0], res[1]
        self._y_calib_ready = True

    def _px2data(self, px, py):
        if not (self._x_calib_ready and self._y_calib_ready): return None,None
        xr = self._calib_ax*px + self._calib_bx
        yr = self._calib_ay*py + self._calib_by
        x = 10**xr if self.scale_type=="loglog" else xr
        y = 10**yr if self.scale_type in ("semilogy","loglog") else yr
        return round(x,6), round(y,6)

    def remove_last_point(self):
        ds = self.datasets[self.active_ds]
        if ds["pts_px"]:
            ds["pts_px"].pop(); ds["pts_real"].pop(); self._redraw(); return True
        return False

    def bulk_add_pixels(self, pts_px):
        ds = self.datasets[self.active_ds]; added=0
        for px,py in pts_px:
            x,y = self._px2data(px,py)
            if x is not None:
                ds["pts_px"].append([px,py]); ds["pts_real"].append([x,y]); added+=1
        if added: self._redraw()
        return added

    def _on_press(self, event):
        if event.inaxes != self.ax or self.image is None or event.button != 1: return
        px,py = event.xdata, event.ydata
        if self.mode == "roi":
            self._roi_click1 = (px,py)
        elif self.mode == "calibrate_x" and len(self.x_calib_pts) < self._x_target:
            self.x_calib_pts.append([px,py]); self.calibration_click.emit(px,py); self._redraw()
        elif self.mode == "calibrate_y" and len(self.y_calib_pts) < self._y_target:
            self.y_calib_pts.append([px,py]); self.calibration_click.emit(px,py); self._redraw()
        elif self.mode == "digitize":
            x,y = self._px2data(px,py)
            if x is None: return
            ds = self.datasets[self.active_ds]
            ds["pts_px"].append([px,py]); ds["pts_real"].append([x,y])
            self.datapoint_click.emit(x,y); self._redraw()

    def _on_release(self, event):
        if self.mode != "roi" or self._roi_click1 is None: return
        if event.inaxes != self.ax or self.image is None: return
        x0,y0 = self._roi_click1
        x1,y1 = event.xdata, event.ydata
        self.roi = (min(x0,x1), min(y0,y1), max(x0,x1), max(y0,y1))
        self._roi_click1 = None
        self.mode = "none"
        self.roi_updated.emit(*self.roi)
        self._redraw()

    def _redraw(self):
        self.ax.cla(); self._style()
        if self.image is None: self.draw(); return
        h,w = self.image.shape[:2]
        self.ax.imshow(self.image, extent=[0,w,h,0], aspect="auto", interpolation="bilinear")
        self.ax.set_xlim(0,w); self.ax.set_ylim(h,0)
        # ROI rectangle
        if self.roi is not None:
            rx0,ry0,rx1,ry1 = self.roi
            rect = mpatches.Rectangle((rx0,ry0), rx1-rx0, ry1-ry0,
                linewidth=1.5, edgecolor="#00ff88", facecolor="none",
                linestyle="--", zorder=8)
            self.ax.add_patch(rect)
            self.ax.annotate("ROI",xy=(rx0,ry0),xytext=(4,4),
                             textcoords="offset points",color="#00ff88",fontsize=8,zorder=9)
        # X calib points — orange ▼
        for i,pt in enumerate(self.x_calib_pts):
            self.ax.plot(pt[0],pt[1],"v",color="#ffaa00",markersize=11,
                         markeredgecolor="white",markeredgewidth=0.8,zorder=10)
            self.ax.annotate(f"X{i+1}",xy=(pt[0],pt[1]),xytext=(5,4),
                             textcoords="offset points",color="#ffaa00",fontsize=8,fontweight="bold",zorder=11)
        # Y calib points — cyan ▶
        for i,pt in enumerate(self.y_calib_pts):
            self.ax.plot(pt[0],pt[1],">",color="#00d4ff",markersize=11,
                         markeredgecolor="white",markeredgewidth=0.8,zorder=10)
            self.ax.annotate(f"Y{i+1}",xy=(pt[0],pt[1]),xytext=(5,4),
                             textcoords="offset points",color="#00d4ff",fontsize=8,fontweight="bold",zorder=11)
        # Dataset points
        for idx,ds in enumerate(self.datasets):
            if not ds["pts_px"]: continue
            xpx=[p[0] for p in ds["pts_px"]]; ypx=[p[1] for p in ds["pts_px"]]
            col=self.DS_COLORS[idx%len(self.DS_COLORS)]
            self.ax.scatter(xpx,ypx,color=col,marker=ds["marker"],
                            s=20,edgecolors="white",linewidths=0.3,zorder=9,label=ds["label"])
        if any(ds["pts_px"] for ds in self.datasets):
            self.ax.legend(loc="lower right",facecolor="#1e1e1e",
                           edgecolor="#555",labelcolor="white",fontsize=8)
        # Y=0 guide line — verify calibration zero is correctly placed
        if self._y_calib_ready and self.image is not None:
            if abs(self._calib_ay) > 1e-12:
                py_zero = -self._calib_by / self._calib_ay
                img_h = self.image.shape[0]
                if 0 <= py_zero <= img_h:
                    self.ax.axhline(py_zero, color="#00ff88", linewidth=1.0,
                                    linestyle="--", alpha=0.75, zorder=8)
                    self.ax.text(4, py_zero - 4, "Y=0 (calib)",
                                color="#00ff88", fontsize=8, zorder=9)
        self.fig.tight_layout(pad=0.3); self.draw()


# ══════════════════════════════════════════════════════════════════════════════
#  Color swatch
# ══════════════════════════════════════════════════════════════════════════════

class ColorSwatch(QFrame):
    clicked = pyqtSignal()
    def __init__(self, rgb=(220,50,50)):
        super().__init__(); self.setFixedSize(28,28); self.setFrameShape(QFrame.Box); self.set_color(rgb)
    def set_color(self,rgb): self._rgb=rgb; self.setStyleSheet(f"background:rgb{rgb};border:2px solid #555;border-radius:4px;")
    def get_rgb(self): return self._rgb
    def mousePressEvent(self,e): self.clicked.emit()


# ══════════════════════════════════════════════════════════════════════════════
#  Stylesheet
# ══════════════════════════════════════════════════════════════════════════════

DARK_QSS = """
    QMainWindow,QWidget{background:#1a1a2e;color:#e0e0e0}
    QGroupBox{border:1px solid #333a5c;border-radius:6px;margin-top:10px;
              padding-top:12px;font-weight:600;color:#8899cc}
    QGroupBox::title{subcontrol-origin:margin;left:10px;top:-1px}
    QPushButton{background:#252545;border:1px solid #3a3a6a;border-radius:5px;
                padding:6px 12px;color:#c8d8ff;font-size:9pt}
    QPushButton:hover{background:#303060;border-color:#5566aa}
    QPushButton:pressed{background:#1a1a40}
    QPushButton#primary{background:#01696f;border-color:#4f98a3;color:white;font-weight:600}
    QPushButton#primary:hover{background:#0c4e54}
    QPushButton#xbtn{background:#3a2800;border-color:#ffaa00;color:#ffcc66;font-weight:600}
    QPushButton#xbtn:hover{background:#4a3800}
    QPushButton#ybtn{background:#003a3a;border-color:#00d4ff;color:#66eeff;font-weight:600}
    QPushButton#ybtn:hover{background:#004a4a}
    QPushButton#danger{background:#6b1a1a;border-color:#aa4444;color:#ffaaaa}
    QPushButton#danger:hover{background:#8b2222}
    QPushButton#extract{background:#1a3a1a;border-color:#44aa44;color:#aaffaa;font-weight:600}
    QPushButton#extract:hover{background:#1f4f1f}
    QPushButton#roi{background:#1a2e1a;border-color:#00ff88;color:#88ffcc;font-weight:600}
    QPushButton#roi:hover{background:#1f3f1f}
    QLineEdit,QComboBox,QSpinBox,QDoubleSpinBox{
        background:#111122;border:1px solid #333a5c;border-radius:4px;padding:4px 6px;color:#e0e0e0}
    QLineEdit:focus,QComboBox:focus{border-color:#4f98a3}
    QTableWidget{background:#111122;gridline-color:#222244;border:1px solid #333a5c}
    QTableWidget::item{padding:3px;color:#d0d8ff}
    QTableWidget::item:selected{background:#1e3a60}
    QHeaderView::section{background:#1a1a3a;color:#8899cc;border:none;
                          padding:4px 6px;font-weight:600;font-size:8pt}
    QLabel{color:#c8d8ff}
    QStatusBar{background:#111122;color:#7788aa;font-size:8pt}
    QSplitter::handle{background:#222244}
    QMenuBar{background:#111122;color:#c8d8ff}
    QMenuBar::item:selected{background:#1e3a60}
    QMenu{background:#1a1a2e;color:#c8d8ff;border:1px solid #333a5c}
    QMenu::item:selected{background:#1e3a60}
    QComboBox QAbstractItemView{background:#111122;selection-background-color:#1e3a60}
    QTabWidget::pane{border:1px solid #333a5c;background:#1a1a2e}
    QTabBar::tab{background:#111122;color:#8899cc;padding:6px 12px;border:1px solid #333a5c}
    QTabBar::tab:selected{background:#1a1a2e;color:#c8d8ff;border-bottom:2px solid #4f98a3}
    QSlider::groove:horizontal{height:4px;background:#333a5c;border-radius:2px}
    QSlider::handle:horizontal{width:14px;height:14px;margin:-5px 0;background:#4f98a3;border-radius:7px}
    QSlider::sub-page:horizontal{background:#4f98a3;border-radius:2px}
    QProgressBar{background:#111122;border:1px solid #333a5c;border-radius:4px;text-align:center;color:#c8d8ff}
    QProgressBar::chunk{background:#01696f;border-radius:3px}
    QScrollArea{border:none}
    QCheckBox{color:#c8d8ff}
    QCheckBox::indicator{width:14px;height:14px;border:1px solid #333a5c;border-radius:3px;background:#111122}
    QCheckBox::indicator:checked{background:#4f98a3;border-color:#4f98a3}
"""


# ══════════════════════════════════════════════════════════════════════════════
#  Main Window
# ══════════════════════════════════════════════════════════════════════════════

ZH_LABELS = {
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
    "✕  Clear ROI": "✕  清除 ROI",
    "🖱  Click X Ticks on Image": "🖱  点击图像上的 X 轴刻度",
    "🖱  Click Y Ticks on Image": "🖱  点击图像上的 Y 轴刻度",
    "↺  Reset Calibration": "↺  重置校准",
    "＋  New Dataset": "＋  新建数据集",
    "✖  Delete Dataset": "✖  删除数据集",
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
    "⚙  XRDML options (wavelength, source)": "⚙  XRDML 选项 (波长、光源)",
}

ZH_LABELS.update({
    "PlotDigitizer — Separated Axis Calibration + ROI": "PlotDigitizer — 独立坐标轴校准 + ROI",
    "Image": "图像",
    "Axis Scale": "坐标轴比例",
    "Linear": "线性",
    "Semi-log Y": "半对数 Y",
    "Log-Log": "双对数",
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


class MainWindow(QMainWindow):
    def __init__(self, dpi_scale=1.0):
        super().__init__()
        self._dpi_scale = dpi_scale
        self.language = "zh"
        self._translatable_actions = []
        self._translatable_controls = []
        self._translatable_combos = []
        self._translatable_tabs = []
        self._translatable_table_headers = []
        self._translatable_tooltips = []
        self.setWindowTitle("PlotDigitizer — Separated Axis Calibration + ROI")
        min_w = int(1300 * self._dpi_scale); min_w = max(1100, min(min_w, 2400))
        min_h = int(800  * self._dpi_scale); min_h = max(700,  min(min_h, 1600))
        self.setMinimumSize(min_w, min_h)
        self._eyedrop_active = False
        self._build_ui(); self._build_menu()
        self._register_translatable_controls()
        self._apply_language()
        self.setStyleSheet(DARK_QSS)
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
                           ("Clear Active Dataset",None,self._clear_dataset),
                           ("Reset Calibration",None,self._reset_calibration),
                           ("Clear ROI",None,self._clear_roi)]:
            a=QAction(lbl,self)
            a.setProperty("i18n_key", lbl)
            if sc: a.setShortcut(sc)
            a.triggered.connect(fn); em.addAction(a)
            self._translatable_actions.append(a)
        self.menu_language=lm=mb.addMenu("&Language")
        self.language_group=QActionGroup(self); self.language_group.setExclusive(True)
        self.action_lang_en=QAction("English",self); self.action_lang_en.setCheckable(True)
        self.action_lang_zh=QAction("Chinese",self); self.action_lang_zh.setCheckable(True)
        self.action_lang_en.setChecked(self.language == "en")
        self.action_lang_zh.setChecked(self.language == "zh")
        for action, lang, key in [(self.action_lang_en,"en","English"),
                                  (self.action_lang_zh,"zh","Chinese")]:
            action.setProperty("i18n_key", key)
            action.triggered.connect(lambda checked, code=lang: checked and self._set_language(code))
            self.language_group.addAction(action); lm.addAction(action)
            self._translatable_actions.append(action)
        self.menu_help=hm=mb.addMenu("&Help")
        a=QAction("About",self); a.setProperty("i18n_key","About")
        a.triggered.connect(self._about); hm.addAction(a)
        self._translatable_actions.append(a)

    def _tr(self, text):
        return ZH_LABELS.get(text, text) if self.language == "zh" else text

    def _set_language(self, language):
        self.language = language
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
        if hasattr(self, "menu_file"):
            self.menu_file.setTitle(self._tr("&File"))
            self.menu_edit.setTitle(self._tr("&Edit"))
            self.menu_language.setTitle(self._tr("&Language"))
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

    def _sync_dynamic_labels(self):
        if not hasattr(self, "canvas"):
            return
        if self.canvas.roi is None:
            self.lbl_roi.setText(self._tr("No ROI — full image used"))
        else:
            x0,y0,x1,y1 = self.canvas.roi
            self.lbl_roi.setText(f"ROI: ({x0:.0f},{y0:.0f}) → ({x1:.0f},{y1:.0f})")
        self._update_calib_status()

    def _build_ui(self):
        central=QWidget(); self.setCentralWidget(central)
        root=QHBoxLayout(central); root.setContentsMargins(0,0,0,0); root.setSpacing(0)
        splitter=QSplitter(Qt.Horizontal); root.addWidget(splitter)

        scroll=QScrollArea(); scroll.setWidgetResizable(True)
        panel_w = int(420 * self._dpi_scale)
        panel_w = max(340, min(panel_w, 640))
        scroll.setMinimumWidth(panel_w)
        scroll.setMaximumWidth(panel_w + 60)
        left=QWidget(); scroll.setWidget(left)
        lv=QVBoxLayout(left); lv.setContentsMargins(8,8,8,8); lv.setSpacing(8)

        # Open image
        g=QGroupBox("Image"); gv=QVBoxLayout(g)
        b=QPushButton("📂  Open Image…"); b.setObjectName("primary")
        b.clicked.connect(self._open_image); gv.addWidget(b); lv.addWidget(g)

        # Scale
        g=QGroupBox("Axis Scale"); gv=QVBoxLayout(g)
        self.combo_scale=QComboBox(); self.combo_scale.addItems(["Linear","Semi-log Y","Log-Log"])
        self.combo_scale.currentIndexChanged.connect(self._scale_changed)
        gv.addWidget(self.combo_scale); lv.addWidget(g)

        # ── ROI ───────────────────────────────────────────────────────────────
        g=QGroupBox("Step 0 — Plot Area (ROI)  [recommended]"); gv=QVBoxLayout(g)
        gv.addWidget(QLabel(
            "Restrict extraction to the plot area only.\n"
            "Draw a box just inside the axis lines to\n"
            "exclude tick labels, borders, and margins."))
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

        # Dark curve mode — prominent for black-on-white plots
        dg=QGroupBox("Mode 1 — Dark Curve  ★ (black on white)"); dgv=QVBoxLayout(dg)
        dgv.addWidget(QLabel("Best for black/dark lines on light background.\n(e.g. patent figures, scanned PXRD)"))
        dgv.addWidget(QLabel("Darkness threshold (0=black, 255=white):"))
        self.slider_dark=QSlider(Qt.Horizontal); self.slider_dark.setRange(10,240); self.slider_dark.setValue(128)
        self.lbl_dark=QLabel("128"); self.slider_dark.valueChanged.connect(lambda v: self.lbl_dark.setText(str(v)))
        dr=QHBoxLayout(); dr.addWidget(self.slider_dark); dr.addWidget(self.lbl_dark); dgv.addLayout(dr)
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
        lv.addWidget(g); lv.addStretch(); splitter.addWidget(scroll)

        # Right panel
        right=QWidget(); rv=QVBoxLayout(right); rv.setContentsMargins(6,6,6,6); rv.setSpacing(6)
        self.canvas=PlotCanvas(self)
        self.canvas.calibration_click.connect(self._on_calib_click)
        self.canvas.datapoint_click.connect(self._on_data_click)
        self.canvas.roi_updated.connect(self._on_roi_updated)
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
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(0, lambda: splitter.setSizes(
            [int(420 * self._dpi_scale),
             splitter.width() - int(420 * self._dpi_scale)]))
        self.status=QStatusBar(); self.setStatusBar(self.status)

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
                self.canvas.fit_x_calib(self.canvas.x_calib_pts,self.canvas.x_calib_vals)
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
                self.canvas.fit_y_calib(self.canvas.y_calib_pts,self.canvas.y_calib_vals)
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
            self.canvas.load_image(path)
            self.lbl_calib.setText(self._tr("⬤ X: not set     ⬤ Y: not set"))
            self.lbl_x_calib.setText(self._tr("○ ○ ○  awaiting X points"))
            self.lbl_y_calib.setText(self._tr("○ ○ ○  awaiting Y points"))
            self.lbl_roi.setText(self._tr("No ROI — full image used"))
            self.table.setRowCount(0)
            self._update_status(f"Loaded: {path}")

    def _scale_changed(self,idx):
        self.canvas.scale_type=("linear","semilogy","loglog")[idx]
        self._update_status(self._tr("Scale: ")+self.combo_scale.currentText())

    # ── Manual digitize ──────────────────────────────────────────────────────

    def _start_digitize(self):
        if not self._ok(): return
        self.canvas.mode="digitize"
        self._update_status(f"DIGITIZE | {self.canvas.datasets[self.canvas.active_ds]['label']}")

    def _undo_point(self):
        if self.canvas.remove_last_point():
            self._refresh_table(); self._update_status(self._tr("Last point removed"))

    def _clear_dataset(self):
        ds=self.canvas.datasets[self.canvas.active_ds]
        ds["pts_px"]=[]; ds["pts_real"]=[]
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
        self.canvas.active_ds=idx
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
        pts=extract_dark_curve(self.canvas.image,self.slider_dark.value(),
                                self.spin_step.value(),roi=roi,
                                pick=pick,min_run=self.spin_run.value(),
                                connectivity=conn,
                                min_area_frac=self.spin_dark_maf.value(),
                                min_width_frac=self.spin_dark_mwf.value(),
                                max_aspect=self.spin_dark_asp.value(),
                                top_n=self.spin_dark_topn.value(),
                                y_band=yb)
        self.progress.setValue(80); QApplication.processEvents()
        self._run_auto(pts,f"Dark curve (pick={pick}, conn={conn})")

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
            "• X and Y calibrated independently → correct even for stacked panels<br><br>"
            "<b>Recommended workflow for black-on-white PXRD:</b><br>"
            "1. Draw ROI box just inside the axis lines<br>"
            "2. Calibrate X (click tick marks) and Y (click tick marks)<br>"
            "3. Auto Extract → Mode 1 Dark Curve → topmost → Extract<br><br>"
            "Ctrl+Z = undo last point")


def main():
    # High-DPI: must be set before QApplication is created
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps,    True)
    app = QApplication(sys.argv)
    app.setApplicationName("PlotDigitizer")
    app.setStyle("Fusion")

    # Scale base font to screen DPI so widgets resize proportionally.
    # 96 DPI is the reference; on a 192-DPI (2×) screen the font grows
    # naturally and all em-based spacing scales with it.
    screen = app.primaryScreen()
    dpi    = screen.logicalDotsPerInch()
    scale  = max(1.0, dpi / 96.0)
    base_pt = int(round(9 * scale))   # 9 pt @ 96 DPI; ~13 pt @ 144 DPI
    base_pt = max(9, min(base_pt, 14)) # clamp: never smaller than 9 pt or larger than 14 pt
    font = app.font()
    font.setPointSize(base_pt)
    app.setFont(font)

    win = MainWindow(dpi_scale=scale)
    win.show()
    sys.exit(app.exec_())

if __name__=="__main__":
    main()
