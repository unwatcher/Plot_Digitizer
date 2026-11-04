"""Scientific-plot extraction, presets and ROI detection without GUI dependencies.

Adapted from Plot_Digitizer by unwatcher (BSD 3-Clause License).
See THIRD_PARTY_NOTICES.md for the retained license and attribution.
Inputs and returned samples use RGB uint8 images and source-image pixel coordinates.
The extraction heuristics do not provide calibrated accuracy probabilities.
"""

import math

import cv2
import numpy as np


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


def _true_ranges(flags):
    """Return inclusive index ranges for contiguous true values."""
    ranges = []
    start = None
    for index, enabled in enumerate((*np.asarray(flags, dtype=bool), False)):
        if enabled and start is None:
            start = index
        elif not enabled and start is not None:
            ranges.append((start, index - 1))
            start = None
    return ranges


def _standard_long_plot_line_mask(mask, horizontal_fraction=0.85,
                                  vertical_fraction=0.90):
    """Detect clean, nearly straight grid and axis lines."""
    height, width = mask.shape
    horizontal_seed = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1)),
    )
    horizontal_seed = cv2.dilate(
        horizontal_seed,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)),
    )
    vertical_seed = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7)),
    )
    vertical_seed = cv2.dilate(
        vertical_seed,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1)),
    )

    horizontal_rows = np.count_nonzero(horizontal_seed, axis=1) >= (
        width * horizontal_fraction
    )
    vertical_columns = np.count_nonzero(vertical_seed, axis=0) >= (
        height * vertical_fraction
    )
    long_lines = np.zeros_like(mask)
    long_lines[horizontal_rows, :] = 255
    long_lines[:, vertical_columns] = 255

    hough_lines = cv2.HoughLinesP(
        mask,
        1,
        np.pi / 720,
        threshold=max(20, width // 30),
        minLineLength=max(20, round(width * 0.22)),
        maxLineGap=max(8, round(width * 0.04)),
    )
    if hough_lines is not None:
        thickness = max(3, round(min(height, width) / 250))
        for x1, y1, x2, y2 in hough_lines[:, 0]:
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            dx, dy = abs(x2 - x1), abs(y2 - y1)
            if dx >= width * 0.30 and dy <= max(4, dx * 0.035):
                cv2.line(long_lines, (x1, y1), (x2, y2), 255, thickness)
            elif dy >= height * 0.88 and dx <= max(4, dy * 0.025):
                cv2.line(long_lines, (x1, y1), (x2, y2), 255, thickness)
    return long_lines


def _credible_projection_groups(groups, coverage, radius, axis_length):
    """Reject broad dark regions and isolated peak-like projection bands."""
    max_span = max(5, radius * 6)
    candidates = [
        group for group in groups if group[1] - group[0] + 1 <= max_span
    ]
    if len(candidates) >= 3:
        centers = np.asarray(
            [(start + end) / 2.0 for start, end in candidates], dtype=float
        )
        gaps = np.diff(centers)
        best_matches = np.asarray([], dtype=int)
        for gap in gaps:
            tolerance = max(3.0, gap * 0.15)
            matches = np.flatnonzero(np.abs(gaps - gap) <= tolerance)
            if matches.size > best_matches.size:
                best_matches = matches
        if best_matches.size >= 2:
            selected = set(best_matches.tolist())
            selected.update((best_matches + 1).tolist())
            for index, (start, end) in enumerate(candidates):
                center = centers[index]
                peak_coverage = float(np.max(coverage[start:end + 1]))
                near_edge = (
                    center <= radius * 3
                    or center >= axis_length - radius * 3
                )
                if near_edge and peak_coverage >= 0.80:
                    selected.add(index)
            return [candidates[index] for index in sorted(selected)]
        return candidates
    if len(candidates) == 2:
        return candidates
    credible = []
    for start, end in candidates:
        center = (start + end) / 2.0
        peak_coverage = float(np.max(coverage[start:end + 1]))
        near_edge = center <= radius * 3 or center >= axis_length - radius * 3
        if near_edge or peak_coverage >= 0.96:
            credible.append((start, end))
    return credible


def _strong_grid_line_mask(mask):
    """Detect long grid bands in noisy scans, including wavy or broken lines."""
    height, width = mask.shape
    row_radius = max(2, min(14, round(height * 0.005)))
    column_radius = max(2, min(14, round(width * 0.003)))

    row_expanded = cv2.dilate(
        mask,
        cv2.getStructuringElement(
            cv2.MORPH_RECT, (1, row_radius * 2 + 1)
        ),
    )
    column_expanded = cv2.dilate(
        mask,
        cv2.getStructuringElement(
            cv2.MORPH_RECT, (column_radius * 2 + 1, 1)
        ),
    )
    row_coverage = np.count_nonzero(row_expanded, axis=1) / max(1, width)
    column_coverage = np.count_nonzero(column_expanded, axis=0) / max(1, height)

    horizontal_groups = _credible_projection_groups(
        _true_ranges(row_coverage >= 0.70),
        row_coverage,
        row_radius,
        height,
    )
    vertical_groups = _credible_projection_groups(
        _true_ranges(column_coverage >= 0.88),
        column_coverage,
        column_radius,
        width,
    )

    padding = max(2, min(8, round(min(height, width) * 0.003)))
    long_lines = np.zeros_like(mask)
    for start, end in horizontal_groups:
        y0 = max(0, start - row_radius - padding)
        y1 = min(height, end + row_radius + padding + 1)
        long_lines[y0:y1, :] = 255
    for start, end in vertical_groups:
        x0 = max(0, start - column_radius - padding)
        x1 = min(width, end + column_radius + padding + 1)
        long_lines[:, x0:x1] = 255
    return long_lines


def _long_plot_line_mask(mask, strength="standard",
                         horizontal_fraction=0.85,
                         vertical_fraction=0.90):
    strength = str(strength).strip().lower()
    if strength not in {"standard", "strong"}:
        raise ValueError("Grid suppression strength must be 'standard' or 'strong'.")
    if strength == "strong":
        return _strong_grid_line_mask(mask)
    return _standard_long_plot_line_mask(
        mask,
        horizontal_fraction=horizontal_fraction,
        vertical_fraction=vertical_fraction,
    )


def suppress_grid_line_signals(mask, strength="standard"):
    """Remove long grid/axis signals from a binary dark-pixel mask.

    ``strong`` uses band coverage rather than requiring perfectly straight
    rows or columns, which makes it suitable for scanned patent figures.
    """
    if mask.ndim != 2:
        raise ValueError("Grid suppression expects a two-dimensional mask.")
    binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    long_lines = _long_plot_line_mask(binary, strength=strength)
    return cv2.bitwise_and(binary, cv2.bitwise_not(long_lines))


def _suppress_long_plot_lines(mask, horizontal_fraction=0.85,
                              vertical_fraction=0.90,
                              aggressive_horizontal=False):
    """Backward-compatible wrapper for grid/axis signal suppression."""
    strength = "strong" if aggressive_horizontal else "standard"
    long_lines = _long_plot_line_mask(
        mask,
        strength=strength,
        horizontal_fraction=horizontal_fraction,
        vertical_fraction=vertical_fraction,
    )
    return cv2.bitwise_and(mask, cv2.bitwise_not(long_lines))


AUTO_DARK_THRESHOLDS = (128, 160, 190, 220, 235)


EXTRACTION_PRESETS = {
    "normal": {
        "label": "Normal scan",
        "summary": "Clean single trace without a long grid.",
        "auto_threshold": True,
        "threshold": 160,
        "col_step": 1,
        "min_run": 2,
        "pick": "topmost",
        "y_band": False,
        "connectivity": True,
        "min_area_frac": 0.003,
        "min_width_frac": 0.05,
        "max_aspect": 8.0,
        "top_n": 5,
        "suppress_grid": False,
        "grid_strength": "standard",
        "max_points": 3000,
    },
    "grid": {
        "label": "Grid / scanned patent",
        "summary": "Single trace with long straight, broken, or wavy grid lines.",
        "auto_threshold": True,
        "threshold": 128,
        "col_step": 1,
        "min_run": 2,
        "pick": "topmost",
        "y_band": False,
        "connectivity": True,
        "min_area_frac": 0.003,
        "min_width_frac": 0.05,
        "max_aspect": 8.0,
        "top_n": 5,
        "suppress_grid": True,
        "grid_strength": "strong",
        "max_points": 5000,
    },
    "waterfall": {
        "label": "Waterfall / stacked traces",
        "summary": "One trace at a time using a manually adjusted pixel Y-band.",
        "auto_threshold": True,
        "threshold": 160,
        "col_step": 1,
        "min_run": 2,
        "pick": "topmost",
        "y_band": True,
        "connectivity": True,
        "min_area_frac": 0.001,
        "min_width_frac": 0.015,
        "max_aspect": 12.0,
        "top_n": 12,
        "suppress_grid": False,
        "grid_strength": "standard",
        "max_points": 5000,
    },
}

EXTRACTION_PRESETS["transmittance"] = {
    **EXTRACTION_PRESETS["normal"],
    "label": "IR / downward dips",
    "summary": "Single transmittance trace with downward absorption dips.",
    "pick": "bottommost",
}


def extraction_preset(name):
    """Return a copy of a named dark-curve extraction preset."""
    key = str(name).strip().lower()
    if key not in EXTRACTION_PRESETS:
        raise ValueError(f"Unknown extraction preset: {name}")
    return dict(EXTRACTION_PRESETS[key])


def _plot_line_segments(mask):
    """Find long near-horizontal/vertical segments, allowing scan skew."""
    height, width = mask.shape
    original_height, original_width = height, width
    if max(height, width) > 1000:
        scale = 1000 / max(height, width)
        resized = cv2.resize(mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        mask = np.uint8(resized >= 64) * 255
        height, width = mask.shape
    sx, sy = original_width / width, original_height / height
    lines = cv2.HoughLinesP(
        mask, 1, np.pi / 900,
        threshold=max(30, round(min(height, width) * 0.10)),
        minLineLength=max(12, round(min(height, width) * 0.35)),
        maxLineGap=max(3, round(max(height, width) * 0.015)),
    )
    horizontal, vertical = [], []
    radius = max(2, round(min(height, width) * 0.003))
    horizontal_support = cv2.dilate(mask, np.ones((radius * 2 + 1, 1), np.uint8))
    vertical_support = cv2.dilate(mask, np.ones((1, radius * 2 + 1), np.uint8)).T

    def extend(line, support):
        a0, b0, a1, b1 = line
        slope = (b1 - b0) / (a1 - a0)
        intercept = b0 - slope * a0
        columns = np.arange(support.shape[1])
        rows = np.rint(slope * columns + intercept).astype(int)
        valid = (rows >= 0) & (rows < support.shape[0])
        observed = np.zeros(support.shape[1], np.uint8)
        observed[valid] = support[rows[valid], columns[valid]]
        joined = cv2.morphologyEx(observed[None, :], cv2.MORPH_CLOSE,
                                  np.ones((1, max(3, round(support.shape[1] * 0.02)) | 1), np.uint8))[0]
        middle = (a0 + a1) / 2
        for start, end in _true_ranges(joined > 0):
            if start <= middle <= end:
                return start, slope * start + intercept, end, slope * end + intercept
        return line

    for line in ([] if lines is None else lines[:, 0]):
        x0, y0, x1, y1 = (float(value) for value in line)
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        if dx >= width * 0.40 and dy <= max(2, dx * 0.035):
            x0, y0, x1, y1 = extend((x0, y0, x1, y1) if x0 < x1 else (x1, y1, x0, y0), horizontal_support)
            line = (x0 * sx, y0 * sy, x1 * sx, y1 * sy)
            horizontal.append(line)
        if dy >= height * 0.40 and dx <= max(2, dy * 0.035):
            y0, x0, y1, x1 = extend((y0, x0, y1, x1) if y0 < y1 else (y1, x1, y0, x0), vertical_support)
            line = (x0 * sx, y0 * sy, x1 * sx, y1 * sy)
            vertical.append(line)
    return horizontal, vertical


def _slanted_plot_roi(gray):
    """Locate an axis corner by line intersections when a scan is tilted."""
    height, width = gray.shape
    padding = max(2, round(min(height, width) * 0.003))
    frames = []
    for threshold in (150, 205, 235):
        horizontal, vertical = _plot_line_segments(_dark_mask(gray, threshold))
        for hx0, hy0, hx1, hy1 in horizontal:
            slope_h = (hy1 - hy0) / (hx1 - hx0)
            intercept_h = hy0 - slope_h * hx0
            for vx0, vy0, vx1, vy1 in vertical:
                slope_v = (vx1 - vx0) / (vy1 - vy0)
                intercept_v = vx0 - slope_v * vy0
                corner_x = (intercept_v + slope_v * intercept_h) / (1 - slope_v * slope_h)
                corner_y = slope_h * corner_x + intercept_h
                if (corner_x >= width * 0.35 or corner_y < height * 0.55
                        or not hx0 - padding <= corner_x <= hx0 + (hx1 - hx0) * 0.15
                        or not vy1 - height * 0.10 <= corner_y <= vy1 + height * 0.05):
                    continue
                left = min(vx0, slope_v * corner_y + intercept_v) + padding
                top, right = vy0 + padding, hx1 - padding
                bottom = max(slope_h * left + intercept_h, hy1) - padding
                has_top = False
                for tx0, ty0, tx1, ty1 in horizontal:
                    if (abs((ty0 + ty1) / 2 - vy0) < height * 0.035
                            and tx0 <= corner_x + padding and tx1 >= hx1 - width * 0.04):
                        top = min(ty0, ty1) + padding
                        has_top = True
                closed_right = False
                for rx0, ry0, rx1, ry1 in vertical:
                    if (has_top and abs((rx0 + rx1) / 2 - hx1) < width * 0.035
                            and abs(ry0 - vy0) < height * 0.035
                            and abs(ry1 - corner_y) < max(6, height * 0.02)):
                        closed_right = True
                    if (abs((rx0 + rx1) / 2 - hx1) < width * 0.035
                            and ry0 <= vy0 + height * 0.035 and ry1 >= corner_y - height * 0.05):
                        right = min(right, max(rx0, rx1) - padding)
                if right - left > width * 0.35 and bottom - top > height * 0.35:
                    box = tuple(int(value) for value in (math.ceil(left), math.ceil(top), math.floor(right), math.floor(bottom)))
                    # A connected instrument table below the plot can look
                    # like a larger L-shaped frame. Prefer a closed plot box.
                    frames.append((closed_right, (right - left) * (bottom - top), -threshold, box, has_top))
        if frames:
            break
    return (max(frames)[3], max(frames)[4]) if frames else None


def _include_peaks_above_axis(gray, box):
    """An open Y-axis can stop below a tall peak; include its connected trace."""
    left, top, right, bottom = box
    mask = _dark_mask(gray[:bottom, left:right], 205)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    for _, y, width, height, _ in stats[1:]:
        if width >= mask.shape[1] * 0.25 and y + height > top + 8 and y < top:
            top = max(0, int(y) - 3)
    return left, top, right, bottom


def _bottom_axis_roi(gray):
    """Infer bounds from a ticked X-axis when the drawing omits the Y-axis."""
    height, width = gray.shape
    mask = _dark_mask(gray, 235)
    horizontal, _ = _plot_line_segments(mask)
    boxes = []
    for x0, y0, x1, y1 in horizontal:
        if min(y0, y1) < height * 0.65 or x1 - x0 < width * 0.60:
            continue
        left, right = max(0, round(x0)), min(width, round(x1) + 1)
        axis_y = round((y0 + y1) / 2)
        # Ticks must extend upward, above the axis stroke itself. This avoids
        # interpreting a flat curve or an underline beneath text as an axis.
        tick_height = max(5, round(height * 0.012))
        band = mask[max(0, axis_y - tick_height):max(0, axis_y - 2), left:right]
        ticks = _true_ranges(np.mean(band > 0, axis=0) >= 0.65) if band.size else []
        ticks = [(lo, hi) for lo, hi in ticks if hi - lo < max(6, width * 0.008)]
        if len(ticks) < 5:
            continue
        body = mask[:max(0, axis_y - tick_height - 2), left:right]
        joined = cv2.morphologyEx(body, cv2.MORPH_CLOSE, np.ones((3, 5), np.uint8))
        _, _, stats, _ = cv2.connectedComponentsWithStats(joined)
        traces = [row for row in stats[1:] if row[2] >= (right - left) * 0.40
                  and row[3] >= height * 0.10]
        if not traces:
            continue
        top = max(0, min(int(row[1]) for row in traces) - 3)
        boxes.append(((right - left) * (axis_y - top), (left, top, right, axis_y - 2)))
    return max(boxes)[1] if boxes else None


def detect_plot_roi(img_rgb, cancelled=None):
    """Return a rectangular plot area, or None when no axis frame is found.

    Detects L-shaped and boxed frames on light backgrounds. Short gaps are
    closed before finding long perpendicular axes; line intersections handle
    tilted scans. Connected peaks can extend above an open Y-axis. Work is bounded;
    the returned half-open box always uses original image pixel coordinates.
    """
    image = np.asarray(img_rgb)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("ROI detection expects an RGB image with 8-bit channels.")
    cancelled = cancelled or (lambda: False)
    if cancelled():
        raise InterruptedError("Curve detection cancelled")
    height, width = image.shape[:2]
    if min(height, width) < 16:
        return None
    scale = min(1.0, 1800 / max(height, width))
    if min(round(width * scale), round(height * scale)) < 8:
        return None
    small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else image
    sh, sw = small.shape[:2]
    if min(sh, sw) < 8:
        return None
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    tolerance = max(5, round(min(sh, sw) * 0.015))
    frames = []
    for threshold in (150, 205, 235):
        if cancelled():
            raise InterruptedError("Curve detection cancelled")
        mask = _dark_mask(gray, threshold)
        # Odd kernel lengths avoid shifting the inferred bounds by a pixel.
        horizontal = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((1, max(3, sw // 100) | 1), np.uint8))
        horizontal = cv2.morphologyEx(horizontal, cv2.MORPH_OPEN, np.ones((1, max(8, round(sw * 0.40)) | 1), np.uint8))
        vertical = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((max(3, sh // 100) | 1, 1), np.uint8))
        vertical = cv2.morphologyEx(vertical, cv2.MORPH_OPEN, np.ones((max(8, round(sh * 0.40)) | 1, 1), np.uint8))
        _, _, hs, _ = cv2.connectedComponentsWithStats(horizontal)
        _, _, vs, _ = cv2.connectedComponentsWithStats(vertical)
        hs = [row for row in hs[1:] if row[3] <= max(6, sh * 0.02)]
        vs = [row for row in vs[1:] if row[2] <= max(6, sw * 0.02)]
        for hx, hy, hw, hh, _ in hs:
            if hy < sh * 0.55:
                continue
            for vx, vy, vw, vh, _ in vs:
                closed_right = any(abs(rx - (hx + hw)) <= tolerance and abs(ry - vy) <= tolerance
                                   and abs(ry + rh - (hy + hh / 2)) <= tolerance
                                   for rx, ry, rw, rh, _ in vs)
                if (vx >= sw * 0.35 or abs(hx - (vx + vw / 2)) > tolerance
                        or (abs(vy + vh - (hy + hh / 2)) > tolerance and not closed_right)):
                    continue
                left, top, right, bottom = vx + vw + 1, vy + 2, hx + hw - 2, hy - 1
                has_top = False
                # Trim the full thickness of the opposite borders on boxed plots.
                for tx, ty, tw, th, _ in hs:
                    if (abs(ty - vy) <= tolerance and abs(tx - hx) <= tolerance
                            and abs(tx + tw - hx - hw) <= tolerance):
                        top = max(top, ty + th + 1)
                        has_top = True
                for rx, ry, rw, rh, _ in vs:
                    if (abs(rx - (hx + hw)) <= tolerance and abs(ry - vy) <= tolerance
                            and abs(ry + rh - (hy + hh / 2)) <= tolerance):
                        right = min(right, rx - 1)
                if right - left > sw * 0.35 and bottom - top > sh * 0.35:
                    box = (max(0, math.ceil(left * width / sw)), max(0, math.ceil(top * height / sh)),
                           min(width, math.floor(right * width / sw)), min(height, math.floor(bottom * height / sh)))
                    if min(box[2] - box[0], box[3] - box[1]) >= 8:
                        frames.append((has_top and closed_right, (right - left) * (bottom - top), -threshold, box, has_top))
    box = max(frames)[3] if frames else None
    if box is not None:
        has_top = max(frames)[4]
        small_box = (math.ceil(box[0] * sw / width), math.ceil(box[1] * sh / height),
                     math.floor(box[2] * sw / width), math.floor(box[3] * sh / height))
    else:
        slanted = _slanted_plot_roi(gray)
        if slanted is None:
            small_box = _bottom_axis_roi(gray)
            if small_box is None:
                return None
            has_top = True
        else:
            small_box, has_top = slanted
    left, top, right, bottom = small_box if has_top else _include_peaks_above_axis(gray, small_box)
    return (max(0, math.ceil(left * width / sw)), max(0, math.ceil(top * height / sh)),
            min(width, math.floor(right * width / sw)), min(height, math.floor(bottom * height / sh)))


def _autopilot_roi(image, cancelled=None):
    """Use the same frame detector as the visible Auto-detect ROI action."""
    height, width = image.shape[:2]
    return detect_plot_roi(image, cancelled=cancelled) or (0, 0, width, height)


def _autopilot_tracks(mask, cancelled, pick="center", removed_lines=None, prefer_bottom=False,
                      support_peaks=True):
    """Find complete supported paths through column runs, then separate them.

    A global path survives forks, sharp peaks, and grid gaps that can fragment a
    greedy column tracker. Output contains observed runs only, without filling
    missing columns or assigning a shared crossing to multiple curves.
    """
    height, width = mask.shape
    if np.count_nonzero(mask) > mask.size * 0.30:
        return []
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    keep = stats[:, cv2.CC_STAT_AREA] >= 6
    keep[0] = False
    mask = np.uint8(keep[labels]) * 255
    max_gap = max(5, round(width * 0.025))
    runs, columns = [], []
    for x in range(width):
        if x % 32 == 0 and cancelled():
            raise InterruptedError("Curve detection cancelled")
        rows = np.flatnonzero(mask[:, x])
        cuts = []
        for cut in np.flatnonzero(np.diff(rows) > 1):
            start, end = rows[cut] + 1, rows[cut + 1]
            grid_gap = (removed_lines is not None and end - start <= max(12, height * 0.02)
                        and np.mean(removed_lines[start:end, x] > 0) >= 0.60)
            if not grid_gap:
                cuts.append(cut + 1)
        splits = np.split(rows, cuts) if rows.size else []
        columns.append(list(range(len(runs), len(runs) + len(splits))))
        runs.extend((float(x), float(run[0]), float(run[-1])) for run in splits)
    if not runs:
        return []
    runs = np.asarray(runs)
    available = np.ones(len(runs), dtype=bool)
    candidates = []
    while np.count_nonzero(available) >= max(12, width * 0.28):
        scores = np.zeros(len(runs))
        previous = np.full(len(runs), -1, dtype=int)
        active = []
        for x, indices in enumerate(columns):
            if x % 32 == 0 and cancelled():
                raise InterruptedError("Curve detection cancelled")
            active = [index for index in active if x - runs[index, 0] <= max_gap]
            parents = np.asarray(active, dtype=int)
            for index in indices:
                if not available[index]:
                    continue
                _, lo, hi = runs[index]
                # Extract upper traces first. At touching black strokes this
                # keeps a lower trace from borrowing the upper trace's peaks.
                position = (lo + hi) / (2 * height)
                # A steep observed stroke carries peak information. Without
                # this support reward, a smooth shortcut can skip a narrow
                # peak/dip even though it has excellent X coverage.
                peak_reward = min(0.60, (hi - lo) / max(20, height * 0.50)) if support_peaks else 0
                reward = 1 + 0.15 * (position if prefer_bottom else 1 - position) + peak_reward
                scores[index] = reward
                if parents.size:
                    gaps = x - runs[parents, 0]
                    distances = np.maximum(0, np.maximum(lo - runs[parents, 2], runs[parents, 1] - hi))
                    centers = np.abs((lo + hi - runs[parents, 1] - runs[parents, 2]) / 2)
                    values = (scores[parents] + reward - 0.12 * (gaps - 1)
                              - distances * 0.06 - np.minimum(0.25, centers * 0.001))
                    values[distances > max(8, height * (0.12 if support_peaks else 0.06)) + gaps] = -np.inf
                    best = int(np.argmax(values))
                    if values[best] > reward:
                        scores[index] = values[best]
                        previous[index] = parents[best]
            active.extend(index for index in indices if available[index])
        end = int(np.argmax(scores))
        path = []
        while end >= 0:
            path.append(end)
            end = previous[end]
        path.reverse()
        points = runs[path]
        if len(path) < max(12, width * 0.28) or np.ptp(points[:, 0]) < width * 0.45:
            break
        if pick == "bounds":
            track = [tuple(map(float, run)) for run in points]
        else:
            track = [(float(x), float(lo if pick == "topmost" else hi if pick == "bottommost"
                                      else (lo + hi) / 2)) for x, lo, hi in points]
        candidates.append(track)
        # Clear the whole selected stroke, including its near-vertical outline
        # in adjacent columns, so it cannot become a ghost peak in curve two.
        selected = np.zeros_like(mask)
        for index in path:
            x, lo, hi = runs[index]
            selected[int(lo):int(hi) + 1, int(x)] = 255
        selected = cv2.dilate(selected, np.ones((5, 5), np.uint8))
        covered = np.vstack((np.zeros((1, width), np.uint16),
                             np.cumsum(selected > 0, axis=0, dtype=np.uint16)))
        ix, lo, hi = runs.astype(int).T
        available[(covered[hi + 1, ix] - covered[lo, ix]) > 0] = False
    return candidates


def _autopilot_color_name(rgb):
    hue, saturation, _ = cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2HSV)[0, 0]
    if saturation < 50:
        return "Black / gray"
    if hue < 10 or hue >= 170:
        return "Red"
    if hue < 24:
        return "Orange"
    if hue < 38:
        return "Yellow"
    if hue < 85:
        return "Green"
    if hue < 103:
        return "Cyan"
    if hue < 135:
        return "Blue"
    if hue < 158:
        return "Purple"
    return "Pink"


def _slanted_grid_mask(mask):
    """Suppress repeated long grid lines at their actual scanned angle."""
    height, width = mask.shape
    all_horizontal, all_vertical = _plot_line_segments(mask)
    horizontal = [line for line in all_horizontal if line[2] - line[0] >= width * 0.70]
    vertical = [line for line in all_vertical if line[3] - line[1] >= height * 0.88]

    def distinct_count(values, tolerance):
        centers = []
        for value in sorted(values):
            if not centers or value - centers[-1] > tolerance:
                centers.append(value)
        return len(centers)

    rows = distinct_count([(line[1] + line[3]) / 2 for line in horizontal], max(5, height * 0.02))
    columns = distinct_count([(line[0] + line[2]) / 2 for line in vertical], max(5, width * 0.02))
    lines = np.zeros_like(mask)
    # Stacked spectra also have many nearly horizontal baselines. Perpendicular
    # interior lines distinguish a grid from the sides of the plot frame.
    interior_vertical = [line for line in vertical
                         if width * 0.025 < (line[0] + line[2]) / 2 < width * 0.975]
    has_grid = rows >= 2 and bool(interior_vertical)
    selected = horizontal + vertical if has_grid else (
        [line for line in all_horizontal if (line[1] + line[3]) / 2 < height * 0.025
         or (line[1] + line[3]) / 2 > height * 0.975]
        + [line for line in all_vertical if (line[0] + line[2]) / 2 < width * 0.015
           or (line[0] + line[2]) / 2 > width * 0.985])
    thickness = max(3, min(9, round(min(height, width) * 0.006)))
    for line in selected:
        x0, y0, x1, y1 = (round(value) for value in line)
        cv2.line(lines, (x0, y0), (x1, y1), 255, thickness)
    return lines, has_grid


def _autopilot_valid_tracks(tracks, height):
    """Discard thin, nearly flat frame remnants at the crop boundaries."""
    valid = []
    margin = max(6, height * 0.03)
    for track in tracks:
        runs = np.asarray(track)
        centers = (runs[:, 1] + runs[:, 2]) / 2
        border_fraction = max(np.mean(centers < margin), np.mean(centers > height - margin))
        if (border_fraction >= 0.90
                and np.percentile(centers, 95) - np.percentile(centers, 5) <= max(10, height * 0.04)):
            continue
        valid.append(track)
    return valid


def _autopilot_polarity(tracks):
    """Locate the modal baseline, using tail asymmetry for weak baselines.

    The first supported path is used before tracing in the inferred direction;
    lower paths can already have switched identity at touching strokes.
    """
    if not tracks:
        return "up"
    runs = np.asarray(tracks[0])
    center = (runs[:, 1] + runs[:, 2]) / 2
    middle = np.median(center)
    dynamic = float(np.ptp(center))
    histogram, edges = np.histogram(center, bins=max(8, min(60, round(dynamic / 4))))
    index = int(np.argmax(histogram))
    baseline = (edges[index] + edges[index + 1]) / 2
    if abs(baseline - middle) > max(3, dynamic * 0.08):
        return "down" if baseline < middle else "up"
    up = max(0, middle - np.percentile(runs[:, 1], 5))
    down = max(0, np.percentile(runs[:, 2], 95) - middle)
    return "down" if down > up * 1.20 else "up"


def _autopilot_overlap_regions(tracks, mask):
    """Locate close same-color paths and substantial strokes left unresolved.

    The warnings do not assert a curve count or reconstruct hidden crossings.
    Output ranges are diagnostic only; no interpolated points are exported.
    """
    height, width = mask.shape
    regions = []
    tolerance = max(8, height * 0.012)
    for first, left in enumerate(tracks):
        a = np.asarray(left)
        for second in range(first + 1, len(tracks)):
            b = np.asarray(tracks[second])
            xs = np.arange(max(a[0, 0], b[0, 0]), min(a[-1, 0], b[-1, 0]) + 1)
            if not len(xs):
                continue
            alo, ahi = (np.interp(xs, a[:, 0], a[:, side]) for side in (1, 2))
            blo, bhi = (np.interp(xs, b[:, 0], b[:, side]) for side in (1, 2))
            supported = np.ones(len(xs), dtype=bool)
            for path in (a, b):
                indices = np.clip(np.searchsorted(path[:, 0], xs), 1, len(path) - 1)
                supported &= path[indices, 0] - path[indices - 1, 0] <= max(12, width * 0.06)
            distance = np.maximum(alo - bhi, blo - ahi)
            close = xs[(distance <= tolerance) & supported]
            if len(close):
                for group in np.split(close, np.flatnonzero(np.diff(close) > max(8, width * 0.015)) + 1):
                    regions.append({"x0": float(group[0]), "x1": float(group[-1] + 1),
                                    "kind": "touching", "curves": [first, second]})
    residual = mask.copy()
    selected = np.zeros_like(mask)
    for track in tracks:
        for x, lo, hi in track:
            selected[int(lo):int(hi) + 1, int(x)] = 255
    selected = cv2.dilate(selected, np.ones((5, 5), np.uint8))
    residual[selected > 0] = 0
    residual = cv2.morphologyEx(residual, cv2.MORPH_CLOSE, np.ones((3, 5), np.uint8))
    _, _, stats, _ = cv2.connectedComponentsWithStats(residual)
    for x, y, w, h, area in stats[1:]:
        if w >= width * 0.15 and h >= max(10, height * 0.05) and area >= width * 0.15:
            regions.append({"x0": float(x), "x1": float(x + w),
                            "kind": "unresolved_strokes", "curves": []})
    return regions


def _autopilot_grid_evidence(mask):
    height, width = mask.shape
    rows = _true_ranges(np.count_nonzero(cv2.dilate(mask, np.ones((5, 1), np.uint8)), axis=1) >= width * 0.75)
    columns = _true_ranges(np.count_nonzero(cv2.dilate(mask, np.ones((1, 5), np.uint8)), axis=0) >= height * 0.88)
    interior = [group for group in columns if width * 0.025 < sum(group) / 2 < width * 0.975]
    lines, slanted_grid = _slanted_grid_mask(mask)
    projection_grid = len(rows) >= 2 and bool(interior)
    return lines, slanted_grid or projection_grid, projection_grid


def extract_plot_curves(img_rgb, roi=None, cancelled=None):
    """Detect separate colored/neutral traces and recommend an extraction preset.

    Results contain observed pixel samples only. This is a heuristic for line
    plots on light backgrounds; touching traces of one color can be ambiguous.
    No datasets, calibration, or caller-owned image data are changed.
    """
    image = np.asarray(img_rgb)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("Autopilot expects an RGB image with 8-bit channels.")
    cancelled = cancelled or (lambda: False)
    if cancelled():
        raise InterruptedError("Curve detection cancelled")
    height, width = image.shape[:2]
    if min(height, width) < 8:
        raise ValueError("The plot area must be at least 8 pixels wide and high.")
    if roi is None:
        box = _autopilot_roi(image, cancelled=cancelled)
    else:
        try:
            roi = np.asarray(roi, dtype=float)
        except (TypeError, ValueError):
            raise ValueError("Choose a valid plot area before running Autopilot.") from None
        if roi.shape != (4,) or not np.all(np.isfinite(roi)):
            raise ValueError("Choose a valid plot area before running Autopilot.")
        x0, y0, x1, y1 = (round(float(value)) for value in roi)
        box = (max(0, min(x0, x1)), max(0, min(y0, y1)),
               min(width, max(x0, x1)), min(height, max(y0, y1)))
    x0, y0, x1, y1 = box
    if min(x1 - x0, y1 - y0) < 8:
        raise ValueError("The plot area must be at least 8 pixels wide and high.")
    crop = image[y0:y1, x0:x1]
    # Bound tracing cost on large scans; coordinates are mapped back to the source.
    scale = min(1.0, 1800 / max(crop.shape[:2]))
    size = (round(crop.shape[1] * scale), round(crop.shape[0] * scale))
    if min(size) < 8:
        raise ValueError("The plot area is too narrow after resizing. Choose a tighter plot crop.")
    small = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else crop
    sh, sw = small.shape[:2]
    sx, sy = (x1 - x0) / sw, (y1 - y0) / sh
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    colored = hsv[:, :, 1] >= 50
    candidates = []
    overlap_regions = []

    def add_regions(regions, color, offset):
        for region in regions:
            overlap_regions.append({"x0": x0 + region["x0"] * sx,
                                    "x1": x0 + region["x1"] * sx,
                                    "kind": region["kind"], "color": color,
                                    "curves": [index + offset for index in region["curves"]]})

    def add_tracks(mask, rgb):
        tracks = _autopilot_valid_tracks(_autopilot_tracks(mask, cancelled, pick="bounds"), sh)
        color = _autopilot_color_name(rgb)
        add_regions(_autopilot_overlap_regions(tracks, mask), color, len(candidates))
        for track in tracks:
            points = [(x0 + (x + 0.5) * sx - 0.5, y0 + ((lo + hi) / 2 + 0.5) * sy - 0.5) for x, lo, hi in track]
            candidates.append({"points": points, "rgb": tuple(int(v) for v in rgb),
                               "color": color, "coverage": len(track) / sw, "pick": "center"})

    # Circular hue peaks separate colors without asking the user to tune tolerance.
    histogram = np.bincount(hsv[:, :, 0][colored], minlength=180).astype(float)
    smooth = sum(np.roll(histogram, shift) * weight for shift, weight in ((-2, 1), (-1, 2), (0, 3), (1, 2), (2, 1)))
    while smooth.max() >= max(12, sw * 0.20):
        if cancelled():
            raise InterruptedError("Curve detection cancelled")
        hue = int(np.argmax(smooth))
        distance = np.abs(hsv[:, :, 0].astype(int) - hue)
        distance = np.minimum(distance, 180 - distance)
        select = colored & (distance <= 7)
        if np.any(select):
            rgb = np.median(small[select], axis=0).astype(np.uint8)
            add_tracks(np.uint8(select) * 255, rgb)
        suppress = np.minimum(abs(np.arange(180) - hue), 180 - abs(np.arange(180) - hue)) <= 14
        smooth[suppress] = 0

    neutral = np.uint8(~colored) * 255
    _, grid_context, _ = _autopilot_grid_evidence(cv2.bitwise_and(_dark_mask(gray, max(AUTO_DARK_THRESHOLDS)), neutral))
    best = None
    trials = []
    for threshold in AUTO_DARK_THRESHOLDS:
        if cancelled():
            raise InterruptedError("Curve detection cancelled")
        mask = cv2.bitwise_and(_dark_mask(gray, threshold), neutral)
        removed_lines, has_grid, projection_grid = _autopilot_grid_evidence(mask)
        # A light gray grid may disappear at the winning dark threshold, but
        # remains part of the image context and the recommended preset.
        detected_grid = has_grid or grid_context
        if projection_grid:
            removed_lines = cv2.bitwise_or(removed_lines, _long_plot_line_mask(mask, strength="strong"))
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(removed_lines))
        baseline_tracks = _autopilot_valid_tracks(_autopilot_tracks(
            mask, cancelled, pick="bounds", removed_lines=removed_lines, support_peaks=False), sh)
        polarity = _autopilot_polarity(baseline_tracks)
        tracks = _autopilot_valid_tracks(_autopilot_tracks(
            mask, cancelled, pick="bounds", removed_lines=removed_lines,
            prefer_bottom=polarity == "down"), sh)
        # Favor long supported traces and avoid threshold-induced fragmentation.
        score = sum(len(track) / sw for track in tracks) - 0.08 * len(tracks)
        trials.append({"threshold": threshold, "curves": len(tracks), "support_score": round(score, 4),
                       "polarity": polarity, "suppress_grid": detected_grid})
        choice = (score, -abs(threshold - 160))
        if best is None or choice > best[0]:
            best = (choice, tracks, threshold, detected_grid, polarity, mask)
    _, tracks, threshold, detected_grid, polarity, best_mask = best
    pick = "bottommost" if polarity == "down" else "topmost"
    add_regions(_autopilot_overlap_regions(tracks, best_mask), "Black / gray", len(candidates))
    for track in tracks:
        observed = [(x, hi if polarity == "down" else lo) for x, lo, hi in track]
        values = [small[round(y), round(x)] for x, y in observed]
        rgb = np.median(values, axis=0).astype(np.uint8)
        points = [(x0 + (x + 0.5) * sx - 0.5, y0 + (y + 0.5) * sy - 0.5) for x, y in observed]
        candidates.append({"points": points, "rgb": tuple(int(v) for v in rgb),
                           "color": "Black / gray", "coverage": len(track) / sw, "pick": pick})
    for index, candidate in enumerate(candidates):
        candidate["_track_id"] = index
    candidates.sort(key=lambda candidate: float(np.median(np.asarray(candidate["points"])[:, 1])))
    remap = {candidate.pop("_track_id"): index for index, candidate in enumerate(candidates)}
    for region in overlap_regions:
        region["curves"] = sorted(remap[index] for index in region["curves"])
    credible_trials = [trial for trial in trials if trial["support_score"] >= best[0][0] * 0.85 and trial["curves"]]
    threshold_sensitive = len({trial["curves"] for trial in credible_trials}) > 1
    for index, candidate in enumerate(candidates, 1):
        candidate["label"] = f"Curve {index} ({candidate['color']})"
        candidate["review_regions"] = [dict(region) for region in overlap_regions
                                        if index - 1 in region["curves"]
                                        or (not region["curves"] and region["color"] == candidate["color"])]
        sensitive = threshold_sensitive and candidate["color"] == "Black / gray"
        candidate["separation"] = ("review" if candidate["review_regions"] or sensitive else
                                   "limited" if candidate["coverage"] < 0.80 else "supported")
    preset = ("grid" if detected_grid else "waterfall" if len(candidates) > 1
              else "transmittance" if tracks and polarity == "down" else "normal")
    return {"candidates": candidates, "roi": box, "inferred_roi": roi is None and box != (0, 0, width, height),
            "preset": preset, "threshold": threshold, "suppress_grid": detected_grid,
            "polarity": polarity, "overlap_regions": overlap_regions,
            "review_required": bool(overlap_regions) or threshold_sensitive,
            "confidence_calibrated": False, "threshold_sensitive": threshold_sensitive,
            "threshold_trials": trials,
            "parameters": {"threshold": threshold, "pick": pick, "col_step": 1,
                           "suppress_grid": detected_grid,
                           "grid_strength": "strong" if detected_grid else "standard",
                           "min_component_area": 6, "max_gap": max(5, round(sw * 0.025)),
                           "max_jump": max(8, sh * 0.12),
                           "hue_tolerance": 7, "working_scale": [sx, sy]}}


def reject_points_in_box(points_px, points_real, box):
    """Partition paired points, rejecting those inside an inclusive pixel box."""
    if len(points_px) != len(points_real):
        raise ValueError("Pixel and calibrated point lists must have equal length.")
    x0, y0, x1, y1 = [float(value) for value in box]
    left, right = sorted((x0, x1))
    top, bottom = sorted((y0, y1))
    kept_px, kept_real, removed = [], [], []
    for index, (pixel, real) in enumerate(zip(points_px, points_real)):
        px, py = pixel
        pixel_copy = list(pixel)
        real_copy = list(real)
        if left <= px <= right and top <= py <= bottom:
            removed.append((index, pixel_copy, real_copy))
        else:
            kept_px.append(pixel_copy)
            kept_real.append(real_copy)
    return kept_px, kept_real, removed


def extract_dark_curve(img_rgb, threshold=128, col_step=1, roi=None,
                       pick="topmost", min_run=2,
                       connectivity=True,
                       min_area_frac=0.003, min_width_frac=0.05,
                       max_aspect=8.0, top_n=5, y_band=None,
                       suppress_grid=True, aggressive_grid=False,
                       grid_strength=None):
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

    raw_mask = _dark_mask(gray, threshold)
    if grid_strength is None:
        grid_strength = "strong" if aggressive_grid else "standard"
    grid_line_mask = None
    if suppress_grid:
        grid_line_mask = _long_plot_line_mask(
            raw_mask,
            strength=grid_strength,
        )

    if connectivity:
        # Build full dark mask, filter by connectivity, then column-scan
        mask = raw_mask
        # Dilate slightly to reconnect thin curve strokes broken at anti-aliasing
        k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        mask = cv2.dilate(mask, k3, iterations=1)
        mask = _keep_curve_components(mask, min_area_frac=min_area_frac,
                                       min_width_frac=min_width_frac,
                                       max_aspect=max_aspect, top_n=top_n)
        # Erode back to undo dilation so we don't shift peak tops downward
        mask = cv2.erode(mask, k3, iterations=1)
        if grid_line_mask is not None:
            mask = cv2.bitwise_and(mask, cv2.bitwise_not(grid_line_mask))
        return _mask_to_curve_pts(mask, col_step, pick, smooth_window=1, ox=ox, oy=oy,
                              y_band=y_band)
    else:
        if grid_line_mask is not None:
            mask = cv2.bitwise_and(raw_mask, cv2.bitwise_not(grid_line_mask))
            return _mask_to_curve_pts(
                mask, col_step, pick, smooth_window=1, ox=ox, oy=oy,
                y_band=y_band,
            )
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
            elif pick == "bottommost":
                pts.append((float(x+ox), float(dark[-1])+oy))
            else:
                pts.append((float(x+ox), float(np.mean(dark))+oy))
        return pts


def extract_dark_curve_auto(img_rgb, thresholds=AUTO_DARK_THRESHOLDS, **kwargs):
    """Select a dark threshold using coverage and trace-shape diagnostics."""
    roi = kwargs.get("roi")
    if roi is None:
        width = img_rgb.shape[1]
        height = img_rgb.shape[0]
    else:
        x0, y0, x1, y1 = roi
        width = max(1, int(round(x1 - x0)))
        height = max(1, int(round(y1 - y0)))
    best = None
    for threshold in thresholds:
        points = extract_dark_curve(img_rgb, threshold=threshold, **kwargs)
        if len(points) < 2:
            score = -math.inf
        else:
            x_values = np.asarray([point[0] for point in points], dtype=float)
            y_values = np.asarray([point[1] for point in points], dtype=float)
            coverage = float(np.ptp(x_values)) / width
            density = len(points) / width
            dynamic = float(np.percentile(y_values, 95) - np.percentile(y_values, 5))
            dynamic_score = min(1.0, dynamic / max(5.0, height * 0.12))
            y_range = max(1.0, float(np.ptp(y_values)))
            jumps = float(np.mean(np.abs(np.diff(y_values)) > y_range * 0.20))
            longest_run = 1
            current_run = 1
            for previous, current in zip(y_values, y_values[1:]):
                if previous == current:
                    current_run += 1
                    longest_run = max(longest_run, current_run)
                else:
                    current_run = 1
            plateau = longest_run / len(y_values)
            score = (
                2.0 * min(1.0, coverage)
                + min(1.0, density)
                + dynamic_score
                - 4.0 * max(0.0, plateau - 0.04)
                - 2.0 * max(0.0, jumps - 0.05)
            )
        candidate = (score, -abs(threshold - 190), threshold, points)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    return best[3], best[2]


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


def prepare_reference_points(points, baseline=0.0, scale=1.0):
    """Sort, deduplicate, and baseline-correct digitized reference points."""
    grouped = {}
    for x_value, y_value in points:
        if not (math.isfinite(x_value) and math.isfinite(y_value)):
            continue
        grouped.setdefault(float(x_value), []).append(
            max(0.0, (float(y_value) - float(baseline)) * float(scale))
        )
    return [
        (x_value, sum(values) / len(values))
        for x_value, values in sorted(grouped.items())
    ]

