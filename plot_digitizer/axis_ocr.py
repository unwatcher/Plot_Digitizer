"""Local OCR adapters for scientific-axis labels, independent of Qt and fitting.

Uses Windows OCR when available, or an installed Tesseract executable.
Image files are temporary and are never sent to a remote service.
"""

import csv
import io
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import cv2
import numpy as np


def _run_ocr_process(command, cancelled, timeout=35):
    if cancelled():
        raise InterruptedError("Axis detection cancelled")
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               creationflags=flags)
    deadline = time.monotonic() + timeout
    try:
        while True:
            if cancelled():
                raise InterruptedError("Axis detection cancelled")
            if time.monotonic() >= deadline:
                raise RuntimeError("Local OCR timed out. Try a tighter plot crop.")
            try:
                output, error = process.communicate(timeout=0.1)
                break
            except subprocess.TimeoutExpired:
                continue
        if process.returncode:
            raise RuntimeError("Local OCR failed. Check that a Windows OCR language or Tesseract is installed.")
        return output.decode("utf-8-sig", errors="replace")
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()


def _read_windows_ocr(jobs, root, cancelled):
    script = Path(__file__).with_name("windows_axis_ocr.ps1")
    if not script.is_file():
        raise RuntimeError("The Windows OCR helper is missing from this installation.")
    input_path = root / "jobs.json"
    input_path.write_text(json.dumps(jobs), encoding="utf8")
    powershell = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32/WindowsPowerShell/v1.0/powershell.exe"
    output = _run_ocr_process([str(powershell), "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                               "-File", str(script), "-InputJson", str(input_path)], cancelled)
    result = json.loads(output)
    if isinstance(result, dict):
        result = [result]
    known_ids = {job["id"] for job in jobs}
    if not isinstance(result, list) or any(not isinstance(entry, dict) or entry.get("id") not in known_ids
                                          or not isinstance(entry.get("words"), list) for entry in result):
        raise RuntimeError("The local OCR engine returned an invalid response.")
    return result


def _normalize_word(word):
    """Normalize OCR geometry and discard malformed or non-finite boxes."""
    try:
        result = {key: float(word[key]) for key in ("x", "y", "width", "height")}
        result["text"] = str(word["text"]).strip()
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    if (not result["text"] or not all(math.isfinite(result[key]) for key in ("x", "y", "width", "height"))
            or min(result["x"], result["y"]) < 0 or min(result["width"], result["height"]) <= 0):
        return None
    return result


def read_local_ocr(images, cancelled=None):
    """Read word boxes locally, falling back to installed Tesseract on failure."""
    cancelled = cancelled or (lambda: False)
    if cancelled():
        raise InterruptedError("Axis detection cancelled")
    images = {name: np.asarray(image) for name, image in images.items()}
    for image in images.values():
        if (image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8
                or min(image.shape[:2]) < 1):
            raise ValueError("Local OCR expects non-empty RGB images with 8-bit channels.")
    if not images:
        return {}
    with tempfile.TemporaryDirectory(prefix="crystaltrace-axis-ocr-") as directory:
        root = Path(directory)
        jobs = []
        tiles = {}
        prepared = dict(images)
        contexts = {}
        if sys.platform == "win32":
            # Windows OCR can ignore widely spaced standalone numbers. Give
            # each observed word image a text-line context, retaining a map to
            # its original pixels. The prefix contains no numeric labels.
            for name, image in images.items():
                if cancelled():
                    raise InterruptedError("Axis detection cancelled")
                mask = np.uint8(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) < 215) * 255
                joined = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 9), np.uint8))
                _, _, stats, _ = cv2.connectedComponentsWithStats(joined)
                boxes = sorted([(int(x), int(y), int(w), int(h)) for x, y, w, h, area in stats[1:]
                                if 8 <= h <= 90 and max(4, h * .35) <= w <= 360 and area >= 12], key=lambda b: (b[1], b[0]))[:30]
                if not boxes:
                    continue
                sheet = np.full((len(boxes) * 115, 570, 3), 255, np.uint8)
                positions = []
                for i, (x, y, w, h) in enumerate(boxes):
                    zoom = min(3.0, 40 / h, 360 / w)
                    patch = image[y:y + h, x:x + w]
                    patch = cv2.resize(patch, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_CUBIC)
                    patch_gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
                    _, binary = cv2.threshold(patch_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    patch = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
                    ph, pw = patch.shape[:2]
                    py = i * 115 + (115 - ph) // 2
                    cv2.putText(sheet, "Tick value", (10, i * 115 + 67), cv2.FONT_HERSHEY_SIMPLEX,
                                .75, (0, 0, 0), 2, cv2.LINE_AA)
                    sheet[py:py + ph, 180:180 + pw] = patch
                    positions.append((180, py, x, y, pw, ph, zoom))
                context_id = ("context", name)
                prepared[context_id] = sheet
                contexts[context_id] = (name, positions)
        for index, (name, image) in enumerate(prepared.items()):
            horizontal = image.shape[1] >= image.shape[0]
            length = image.shape[1] if horizontal else image.shape[0]
            starts = list(range(0, max(1, length - 160), 740)) if sys.platform == "win32" and length > 950 else [0]
            for tile_index, start in enumerate(starts):
                if cancelled():
                    raise InterruptedError("Axis detection cancelled")
                x, y = (start, 0) if horizontal else (0, start)
                tile = image[y:y + (900 if not horizontal and len(starts) > 1 else image.shape[0]),
                             x:x + (900 if horizontal and len(starts) > 1 else image.shape[1])]
                tile_id = f"{index}-{tile_index}"
                path = root / f"axis-{tile_id}.png"
                if not cv2.imwrite(str(path), cv2.cvtColor(tile, cv2.COLOR_RGB2BGR)):
                    raise RuntimeError("Could not prepare an image for local OCR.")
                jobs.append({"id": tile_id, "path": str(path)})
                tiles[tile_id] = (name, x, y)
        if sys.platform == "win32":
            try:
                result = _read_windows_ocr(jobs, root, cancelled)
            except InterruptedError:
                raise
            except (OSError, RuntimeError, ValueError) as error:
                if not shutil.which("tesseract"):
                    raise RuntimeError("Windows OCR is unavailable or failed. Enable an OCR language or install Tesseract on PATH.") from error
                result = None
        else:
            result = None
        if result is not None:
            combined = {name: {"words": [], "engine": "Windows OCR"} for name in images}
            for entry in result:
                name, x, y = tiles[entry["id"]]
                for raw_word in entry["words"]:
                    word = _normalize_word(raw_word)
                    if word is None:
                        continue
                    word["x"] += x; word["y"] += y
                    if name in contexts:
                        source, positions = contexts[name]
                        cx, cy = word["x"] + word["width"] / 2, word["y"] + word["height"] / 2
                        for px, py, sx, sy, w, h, zoom in positions:
                            if px <= cx <= px + w and py <= cy <= py + h:
                                word["x"] = sx + (word["x"] - px) / zoom
                                word["y"] = sy + (word["y"] - py) / zoom
                                word["width"] /= zoom; word["height"] /= zoom
                                combined[source]["words"].append(word)
                                break
                    else:
                        combined[name]["words"].append(word)
            return combined
        executable = shutil.which("tesseract")
        if not executable:
            raise RuntimeError("Automatic calibration needs local OCR: Windows OCR, or Tesseract installed on PATH.")
        results = {name: {"words": [], "engine": "Tesseract"} for name in images}
        for job in jobs:
            name, x, y = tiles[job["id"]]
            if name in contexts:
                continue
            output = _run_ocr_process([executable, job["path"], "stdout", "--psm", "11", "tsv"], cancelled)
            for row in csv.DictReader(io.StringIO(output), delimiter="\t"):
                if not row.get("text", "").strip():
                    continue
                word = _normalize_word({"text": row["text"], "x": row.get("left"), "y": row.get("top"),
                                      "width": row.get("width"), "height": row.get("height")})
                if word is not None:
                    word["x"] += x; word["y"] += y
                    results[name]["words"].append(word)
        return results


