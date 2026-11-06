"""Reproduce the README curve preview from pixels and reviewed printed ticks."""

import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from plot_digitizer.digitizer_core import extract_plot_curves
from plot_digitizer.axis_calibration import calibration_coefficients


def render():
    source = ROOT / "examples/patent_plots/us8198452b2-fig1.png"
    image = cv2.cvtColor(cv2.imread(str(source)), cv2.COLOR_BGR2RGB)
    result = extract_plot_curves(image)
    if len(result["candidates"]) != 1:
        raise RuntimeError("Expected one PXRD curve for this example.")
    # Reviewed against the printed 3/60 degree and 0/20000 count axis ticks.
    reviewed = {
        "x": [{"pixel": [76, 433], "value": 3}, {"pixel": [600, 433], "value": 60}],
        "y": [{"pixel": [77, 433], "value": 0}, {"pixel": [77, 33], "value": 20000}],
    }
    pixels = np.asarray(result["candidates"][0]["points"], dtype=float)
    x = np.polyval(calibration_coefficients(reviewed["x"], "x"), pixels[:, 0])
    y = np.polyval(calibration_coefficients(reviewed["y"], "y"), pixels[:, 1])
    target = ROOT / "docs/images/pxrd-extracted.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    figure = Figure(figsize=(6.7, 5.1), dpi=120, facecolor="white")
    FigureCanvasAgg(figure)
    axis = figure.add_subplot(111)
    # Break the preview line across missing observed columns.
    gap = np.flatnonzero(np.diff(pixels[:, 0]) > 1.6) + 1
    for index, segment in enumerate(np.split(np.arange(len(x)), gap)):
        axis.plot(x[segment], y[segment], color="#1565a8", linewidth=.9,
                  label="Extracted curve" if index == 0 else None)
    axis.set(xlabel=r"2$\theta$ (degrees)", ylabel="Intensity (counts)",
             xlim=(3, 65), ylim=(0, 20000), title="Autopilot extraction")
    axis.set_yticks([0, 5000, 10000, 15000, 20000])
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(target, dpi=120)
    with (ROOT / "examples/pxrd-extracted.csv").open("w", newline="", encoding="utf8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["two_theta_deg", "intensity_counts"])
        writer.writerows(zip(x, y))
    metadata = {"source": source.relative_to(ROOT).as_posix(),
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "calibration": reviewed, "calibration_review": "Printed tick coordinates manually reviewed; linear axes.",
                "roi": result["roi"], "preset": result["preset"], "parameters": result["parameters"],
                "observed_points": len(pixels), "smoothing": False, "gap_interpolation": False,
                "limitation": "One illustrative extraction; not an independent accuracy estimate."}
    (ROOT / "examples/pxrd-extraction.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf8")
    print(json.dumps({"image": str(target), "points": len(pixels), "preset": result["preset"]}))


if __name__ == "__main__":
    render()
