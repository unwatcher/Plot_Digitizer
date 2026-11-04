"""Local OCR and tick matching for reviewable plot-axis calibration proposals.

The local OCR adapter uses Windows OCR or an installed Tesseract executable.
Images never leave this computer. OCR output is
untrusted data: only finite numeric words near observed axis ticks are used.
"""

from __future__ import annotations

import itertools
import math
import re

import cv2
import numpy as np

try:
    from .axis_ocr import read_local_ocr
except ImportError:  # standalone archive
    from axis_ocr import read_local_ocr


def parse_axis_number(text):
    value = str(text).strip().replace("−", "-").replace("–", "-").replace("，", ",")
    value = value.strip("()[]")
    # Accept grouped thousands, decimal comma and scientific notation; avoid
    # silently repairing letters such as O/0 or guessing missing decimal dots.
    if re.fullmatch(r"[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?", value):
        value = value.replace(",", "")
    elif re.fullmatch(r"[+-]?\d+,\d{1,2}", value):
        value = value.replace(",", ".")
    if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", value):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _ranges(enabled):
    indices = np.flatnonzero(enabled)
    return np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1) if indices.size else []


def _axis_geometry(gray, roi, axis):
    """Find an axis line close to the ROI edge, followed by perpendicular ticks."""
    left, top, right, bottom = roi
    height, width = gray.shape
    mask = np.uint8(gray < 205) * 255
    span = right - left if axis == "x" else bottom - top
    radius = max(12, round(min(right - left, bottom - top) * 0.045))
    if axis == "x":
        ox, oy = max(0, left - 5), max(0, bottom - radius)
        area = mask[oy:min(height, bottom + radius), ox:min(width, right + 5)]
    else:
        ox, oy = max(0, left - radius), max(0, top - 5)
        area = mask[oy:min(height, bottom + 5), ox:min(width, left + radius)]
    lines = cv2.HoughLinesP(area, 1, np.pi / 720, threshold=max(20, round(span * .15)),
                            minLineLength=max(20, round(span * .60)), maxLineGap=max(3, round(span * .02)))
    choices = []
    for line in ([] if lines is None else lines[:, 0]):
        x0, y0, x1, y1 = line.astype(float)
        x0 += ox; x1 += ox; y0 += oy; y1 += oy
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        if axis == "x" and dx >= span * .60 and dy <= max(3, dx * .04):
            slope = (y1 - y0) / (x1 - x0)
            intercept = y0 - slope * x0
            distance = abs(slope * (left + right) / 2 + intercept - bottom)
            choices.append((distance, -dx, slope, intercept))
        if axis == "y" and dy >= span * .60 and dx <= max(3, dy * .04):
            slope = (x1 - x0) / (y1 - y0)
            intercept = x0 - slope * y0
            distance = abs(slope * (top + bottom) / 2 + intercept - left)
            choices.append((distance, -dy, slope, intercept))
    if not choices:
        return {"found": False, "ticks": [], "line": None}
    _, _, slope, intercept = min(choices)
    start, end = (left, right) if axis == "x" else (top, bottom)
    start, end = max(0, start - radius), min(width if axis == "x" else height, end + radius)
    strength = []
    tick_radius = max(7, round(min(right - left, bottom - top) * .025))
    kernel = max(15, round(span * .035)) | 1
    support = cv2.blur(mask.astype(np.float32), (kernel, 1) if axis == "x" else (1, kernel))
    for coordinate in range(start, end):
        position = round(slope * coordinate + intercept)
        extent = height if axis == "x" else width
        lower, upper = max(0, position - 7), min(extent, position + 8)
        values = support[lower:upper, coordinate] if axis == "x" else support[coordinate, lower:upper]
        exclude_left, exclude_right = position - 2, position + 2
        if len(values):
            nearby = np.arange(lower, upper)
            index = int(np.argmax(values - .1 * np.abs(nearby - position)))
            position = int(nearby[index])
            lo = hi = index
            cutoff = values[index] * .65
            while lo > 0 and values[lo - 1] >= cutoff:
                lo -= 1
            while hi + 1 < len(values) and values[hi + 1] >= cutoff:
                hi += 1
            exclude_left, exclude_right = int(nearby[lo]) - 1, int(nearby[hi]) + 1
        if axis == "x":
            sample = mask[max(0, position - tick_radius):min(height, position + tick_radius + 1), coordinate]
        else:
            sample = mask[coordinate, max(0, position - tick_radius):min(width, position + tick_radius + 1)]
        # Only count pixels away from the line's own thickness.
        sample_start = max(0, position - tick_radius)
        before = sample[:max(0, exclude_left - sample_start)]
        after = sample[max(0, exclude_right - sample_start + 1):]
        strength.append(max(int(np.count_nonzero(before)), int(np.count_nonzero(after))))
    strength = np.asarray(strength)
    ticks = []
    for group in _ranges(strength >= 2):
        if len(group) > max(9, span * .012):
            continue
        position = float(start + np.average(group, weights=strength[group]))
        point = [position, slope * position + intercept] if axis == "x" else [slope * position + intercept, position]
        ticks.append({"point": point, "strength": int(strength[group].max())})
    return {"found": True, "ticks": ticks, "line": [float(slope), float(intercept)]}


def _consistent_axis_pairs(pairs, axis):
    """Reject OCR outliers using tick positions and linear/log consistency."""
    coordinate = 0 if axis == "x" else 1
    pairs = sorted(pairs, key=lambda pair: pair["pixel"][coordinate])
    # One OCR word may have competed for a tick. Prefer the closest label.
    unique = {}
    for pair in pairs:
        key = round(pair["pixel"][coordinate], 1)
        if key not in unique or pair["label_distance"] < unique[key]["label_distance"]:
            unique[key] = pair
    by_value = {}
    for pair in unique.values():
        value = pair["value"]
        if value not in by_value or pair["label_distance"] < by_value[value]["label_distance"]:
            by_value[value] = pair
    pairs = sorted(by_value.values(), key=lambda pair: pair["pixel"][coordinate])
    if len(pairs) < 2:
        return {"points": pairs, "scale": "linear", "fit_error_px": None, "rejected": []}
    pixels = np.array([pair["pixel"][coordinate] for pair in pairs])
    values = np.array([pair["value"] for pair in pairs])
    tolerance = max(3, float(np.ptp(pixels)) * .012)
    best = None
    for scale in ("linear", "log"):
        if scale == "log" and (len(pairs) < 3 or np.any(values <= 0)):
            continue
        transformed = np.log10(values) if scale == "log" else values
        for i, j in itertools.combinations(range(len(pairs)), 2):
            if transformed[j] == transformed[i]:
                continue
            slope = (pixels[j] - pixels[i]) / (transformed[j] - transformed[i])
            predicted = pixels[i] + (transformed - transformed[i]) * slope
            inliers = np.abs(predicted - pixels) <= tolerance
            error = float(np.mean(np.abs(predicted[inliers] - pixels[inliers])))
            score = (int(inliers.sum()), -error, scale == "linear")
            if best is None or score > best[0]:
                best = (score, inliers, scale)
    if best is None:
        return {"points": [], "scale": "linear", "fit_error_px": None, "rejected": pairs}
    _, inliers, scale = best
    selected = [pair for pair, keep in zip(pairs, inliers) if keep]
    if len(pairs) >= 3 and len(selected) < 3:
        return {"points": [], "scale": scale, "fit_error_px": None, "rejected": pairs}
    vals = np.array([pair["value"] for pair in selected])
    coords = np.array([pair["pixel"][coordinate] for pair in selected])
    transformed = np.log10(vals) if scale == "log" else vals
    coefficients = np.polyfit(transformed, coords, 1)
    error = float(np.sqrt(np.mean((np.polyval(coefficients, transformed) - coords) ** 2)))
    rejected = [pair for pair, keep in zip(pairs, inliers) if not keep]
    heights = [pair["label_box"][3] for pair in selected if "label_box" in pair]
    typical_height = float(np.median(heights)) if heights else None
    distinct_rejected = [pair for pair in rejected
                         if (typical_height is None or "label_box" not in pair
                             or typical_height * .55 <= pair["label_box"][3] <= typical_height * 1.50)
                         and all(abs(pair["pixel"][coordinate] - accepted["pixel"][coordinate]) > 3
                                 for accepted in selected)]
    return {"points": selected, "scale": scale, "fit_error_px": error,
            "scale_conflict": len(distinct_rejected) >= 2 and len(distinct_rejected) >= len(pairs) * .20,
            "rejected": rejected}


def _tick_near_label(gray, geometry, axis, center, label_size):
    """Find a local tick peak above the axis stroke's background thickness.

    A faint or wavy axis can join otherwise separate ticks in a global binary
    projection. Label-guided local prominence still requires observed pixels.
    """
    if geometry["line"] is None:
        return None
    height, width = gray.shape
    slope, intercept = geometry["line"]
    coordinate = 0 if axis == "x" else 1
    radius = max(8, round(label_size * .8))
    extent = width if axis == "x" else height
    start, end = max(0, round(center[coordinate]) - radius), min(extent, round(center[coordinate]) + radius + 1)
    ray = max(9, min(26, round(min(height, width) * .04)))
    strengths = []
    for position in range(start, end):
        at = round(slope * position + intercept)
        if axis == "x":
            sample = gray[max(0, at + 3):min(height, at + ray), position]
        else:
            sample = gray[position, max(0, at - ray):max(0, at - 2)]
        strengths.append(int(np.count_nonzero(sample < 215)))
    signal = np.asarray(strengths, dtype=float)
    if not signal.size:
        return None
    background = np.percentile(signal, 30)
    if signal.max() < background + 2:
        return None
    choices = []
    for group in _ranges(signal >= max(background + 1.5, signal.max() * .65)):
        if len(group) > max(9, label_size * .65):
            continue
        position = float(start + np.average(group, weights=signal[group] - background))
        strength = float(signal[group].max() - background)
        score = strength * 1.8 - abs(position - center[coordinate])
        point = [position, slope * position + intercept] if axis == "x" else [slope * position + intercept, position]
        choices.append((score, point, strength))
    if not choices:
        return None
    _, point, strength = max(choices)
    return {"point": point, "strength": strength}


def suggest_axis_calibration(image, roi, cancelled=None, ocr_reader=None):
    """Return editable suggestions; never mutate image, ROI or calibration.

    A proposal always requires review. Numeric labels must match observed ticks
    and a consistent axis model. Missing axes are returned as partial results.
    """
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("Axis detection expects an RGB image with 8-bit channels.")
    height, width = image.shape[:2]
    try:
        roi = np.asarray(roi, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("Choose a valid plot area before detecting axis labels.") from None
    if roi.shape != (4,) or not np.all(np.isfinite(roi)):
        raise ValueError("Choose a valid plot area before detecting axis labels.")
    left, top, right, bottom = map(round, roi)
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise ValueError("The plot area must lie inside the image.")
    cancelled = cancelled or (lambda: False)
    if cancelled():
        raise InterruptedError("Axis detection cancelled")
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    geometry = {axis: _axis_geometry(gray, (left, top, right, bottom), axis) for axis in ("x", "y")}
    strips, mapping, label_bands = {}, {}, {}
    for axis in ("x", "y"):
        line = geometry[axis]["line"]
        if line is None:
            continue
        slope, intercept = line
        if axis == "x":
            position = slope * (left + right) / 2 + intercept
            box = (max(0, left - 35), max(0, round(position) + 4), min(width, right + 35),
                   min(height, round(position) + max(55, round(height * .13))))
        else:
            position = slope * (top + bottom) / 2 + intercept
            box = (max(0, round(position) - max(85, round(width * .17))), max(0, top - 22),
                   max(0, round(position) - 4), min(height, bottom + 22))
        x0, y0, x1, y1 = box
        if x1 <= x0 or y1 <= y0:
            continue
        crop = image[y0:y1, x0:x1]
        scale = min(3.5, 2450 / max(crop.shape[:2]))
        resized = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        border = 24
        strips[axis] = cv2.copyMakeBorder(resized, border, border, border, border, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        mapping[axis] = (x0, y0, scale, border)
        label_bands[axis] = box
    if not strips:
        return {"axes": {axis: {"points": [], "scale": "linear", "rejected": [], "fit_error_px": None} for axis in ("x", "y")},
                "roi": tuple(roi), "engine": None, "review_required": True, "ready": False}
    # Full-plot context can recognize labels missed in isolated strips. Match
    # its words only inside the same axis-label bands, excluding plot content.
    scale = min(2.0, 2450 / max(height, width))
    resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    strips["full"] = cv2.copyMakeBorder(resized, 24, 24, 24, 24, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    mapping["full"] = (0, 0, scale, 24)
    recognized = (ocr_reader or read_local_ocr)(strips, cancelled)
    axes = {}
    for axis in ("x", "y"):
        pairs = []
        for view in (axis, "full"):
            if view not in recognized or axis not in label_bands:
                continue
            x0, y0, scale, border = mapping[view]
            for word in recognized[view]["words"]:
                value = parse_axis_number(word["text"])
                if value is None:
                    continue
                center = [x0 + (float(word["x"]) + float(word["width"]) / 2 - border) / scale,
                          y0 + (float(word["y"]) + float(word["height"]) / 2 - border) / scale]
                bx0, by0, bx1, by1 = label_bands[axis]
                if not (bx0 <= center[0] <= bx1 and by0 <= center[1] <= by1):
                    continue
                coordinate = 0 if axis == "x" else 1
                label_size = float(word["width"] if axis == "x" else word["height"]) / scale
                tick = _tick_near_label(gray, geometry[axis], axis, center, label_size)
                if tick is None:
                    ticks = geometry[axis]["ticks"]
                    if not ticks:
                        continue
                    tick = min(ticks, key=lambda candidate: abs(candidate["point"][coordinate] - center[coordinate]))
                distance = abs(tick["point"][coordinate] - center[coordinate])
                if distance > max(7, label_size * .65):
                    continue
                pairs.append({"pixel": list(map(float, tick["point"])), "value": value,
                              "text": str(word["text"]),
                              "label_box": [x0 + (float(word["x"]) - border) / scale,
                                            y0 + (float(word["y"]) - border) / scale,
                                            float(word["width"]) / scale, float(word["height"]) / scale],
                              "label_distance": float(distance), "tick_observed": True})
        axes[axis] = _consistent_axis_pairs(pairs, axis)
        axes[axis]["ticks_found"] = max(len(geometry[axis]["ticks"]), len(axes[axis]["points"]))
    return {"axes": axes, "roi": tuple(roi), "engine": next(iter(recognized.values()))["engine"] if recognized else None,
            "review_required": True,
            "ready": all(len(axes[axis]["points"]) >= 2 and not axes[axis].get("scale_conflict") for axis in ("x", "y"))}


def calibration_coefficients(points, axis, scale="linear"):
    """Validate reviewed rows and prepare a nondegenerate axis fit atomically."""
    if axis not in ("x", "y") or scale not in ("linear", "log"):
        raise ValueError("Choose a linear or logarithmic X/Y axis.")
    if len(points) < 2:
        raise ValueError("Each axis needs at least two calibration points.")
    coordinate = 0 if axis == "x" else 1
    pixels = np.asarray([point["pixel"][coordinate] for point in points], dtype=float)
    values = np.asarray([point["value"] for point in points], dtype=float)
    if not np.all(np.isfinite(pixels)) or not np.all(np.isfinite(values)):
        raise ValueError("Calibration coordinates and values must be finite.")
    if np.ptp(pixels) < 2 or np.ptp(values) == 0:
        raise ValueError("Use distinct tick positions and numeric values for each axis.")
    if np.any(np.diff(np.sort(pixels)) <= 1e-6):
        raise ValueError("Each calibration label must use a distinct tick position.")
    ordered = values[np.argsort(pixels)]
    if not (np.all(np.diff(ordered) > 0) or np.all(np.diff(ordered) < 0)):
        raise ValueError("Axis values must increase or decrease consistently along the ticks.")
    if scale == "log":
        if np.any(values <= 0):
            raise ValueError("Logarithmic axis labels must be positive.")
        values = np.log10(values)
    coefficients = np.polyfit(pixels, values, 1)
    if not np.all(np.isfinite(coefficients)) or coefficients[0] == 0:
        raise ValueError("The selected ticks do not define a valid calibration.")
    return tuple(map(float, coefficients))
