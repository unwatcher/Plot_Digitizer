"""Run the image-only patent benchmark and write an inspectable overlay gallery.

No annotations are passed to the extractor. Sparse landmarks and curve counts
are checks, not ground-truth curves or a claim of calibrated numeric accuracy.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import html
import itertools
import json
from pathlib import Path
import sys
import time

APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT))
if sys.platform == "win32" and (APP_ROOT / "runtime_deps/py314").is_dir():
    sys.path.insert(0, str(APP_ROOT / "runtime_deps/py314"))

import cv2
import numpy as np

from plot_digitizer.digitizer_core import extract_plot_curves
from plot_digitizer.axis_calibration import suggest_axis_calibration

DEFAULT_MANIFEST = APP_ROOT / "examples/patent_plots/manifest.json"


def image_fingerprint(image):
    """Compare decoded pixels and normalized ink, including resized copies."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    ink = 255 - gray
    ys, xs = np.where(ink > 40)
    if len(xs):
        ink = ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    thumb = cv2.resize(ink, (64, 64), interpolation=cv2.INTER_AREA).astype(float)
    dct = cv2.dct(thumb.astype(np.float32))[:8, :8].flatten()[1:]
    return {
        "pixel_sha256": hashlib.sha256(np.asarray(image.shape, dtype=np.int32).tobytes() + image.tobytes()).hexdigest(),
        "phash": dct > np.median(dct),
        "thumb": thumb - thumb.mean(),
    }


def audit_corpus(manifest, images):
    sources = {source["id"]: source for source in manifest["sources"]}
    families = {}
    for case in manifest["cases"]:
        source = sources[case["source_id"]]
        families.setdefault(source["family_group"], set()).add(source["split"])
    fingerprints = [image_fingerprint(image) for image in images]
    exact, near = [], []
    for (i, left), (j, right) in itertools.combinations(enumerate(fingerprints), 2):
        ids = [manifest["cases"][n]["id"] for n in (i, j)]
        if left["pixel_sha256"] == right["pixel_sha256"]:
            exact.append(ids)
            continue
        distance = int(np.count_nonzero(left["phash"] != right["phash"]))
        norm = np.linalg.norm(left["thumb"]) * np.linalg.norm(right["thumb"])
        similarity = float(np.sum(left["thumb"] * right["thumb"]) / norm) if norm else 0.0
        if distance <= 12 and similarity >= 0.94:
            near.append({"cases": ids, "phash_distance": distance, "ink_correlation": round(similarity, 4)})
    return {"exact_duplicate_pairs": exact, "near_duplicate_candidates": near,
            "family_split_leaks": sorted(family for family, splits in families.items() if len(splits) > 1),
            "family_count": len(families),
            "limitation": "Perceptual matches require visual review; distinct panels from one family are correlated, not independent samples."}


def measure_case(case, result):
    ax0, ay0, ax1, ay1 = result["roi"]
    bx0, by0, bx1, by1 = case["plot_bounds"]
    intersection = max(0, min(ax1, bx1) - max(ax0, bx0)) * max(0, min(ay1, by1) - max(ay0, by0))
    actual_area = (ax1 - ax0) * (ay1 - ay0)
    expected_area = (bx1 - bx0) * (by1 - by0)
    roi_iou = intersection / max(1, actual_area + expected_area - intersection)
    candidates = result["candidates"]
    # A landmark is assigned to one candidate; separate anchors must stay on it.
    landmark_checks = []
    for group in case.get("landmarks", []):
        distances = []
        for candidate in candidates:
            points = np.asarray(candidate["points"])
            # A steep one-pixel-wide stroke spans many rows between adjacent
            # samples. Measure that supported segment, but never bridge gaps.
            delta = np.diff(points, axis=0)
            step = result.get("parameters", {}).get("working_scale", [1, 1])[0]
            supported = (delta[:, 0] > 0) & (delta[:, 0] <= step * 1.6)
            starts, vectors = points[:-1][supported], delta[supported]
            errors = []
            for point in group["points"]:
                error = float(np.min(np.linalg.norm(points - point, axis=1)))
                if len(starts):
                    fraction = np.clip(np.sum((point - starts) * vectors, axis=1) / np.sum(vectors * vectors, axis=1), 0, 1)
                    error = min(error, float(np.min(np.linalg.norm(starts + fraction[:, None] * vectors - point, axis=1))))
                errors.append(error)
            distances.append(max(errors))
        error = min(distances, default=None)
        landmark_checks.append({"label": group["label"], "max_error_px": error,
                                "passed": error is not None and error <= group.get("tolerance_px", 8)})
    count_match = len(candidates) == case["expected_curves"] if case["count_evaluable"] else None
    count_ok = count_match is True
    landmarks_ok = all(item["passed"] for item in landmark_checks)
    polarity = result.get("polarity")
    preset_match = result["preset"] == case["expected_preset"]
    # These are deliberately separate: a review warning cannot make extraction
    # errors pass. Indeterminate source counts are not scored as correct either.
    return {"roi_iou": round(roi_iou, 4), "roi_match": roi_iou >= 0.85,
            "expected_curves": case["expected_curves"], "actual_curves": len(candidates),
            "count_match": count_match, "preset_match": preset_match,
            "polarity_match": polarity == case["expected_polarity"] if polarity else None,
            "landmark_checks": landmark_checks,
            "landmark_metric": "Distance to observed samples or segments between adjacent observed columns; missing columns are never bridged.",
            "overlap_review_expected": case["requires_overlap_review"],
            "overlap_review_detected": bool(result.get("overlap_regions")),
            "extraction_checks_passed": count_ok and roi_iou >= 0.85 and landmarks_ok,
            "coverage_is_accuracy": False}


def write_overlay(image, case, result, target):
    overlay = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    highlighted = overlay.copy()
    for region in result.get("overlap_regions", []):
        x0, x1 = max(0, round(region["x0"]) - 4), min(image.shape[1], round(region["x1"]) + 4)
        highlighted[:, x0:x1] = (30, 180, 240)
    overlay = cv2.addWeighted(overlay, .88, highlighted, .12, 0)
    colors = [(235, 100, 10), (45, 40, 235), (30, 170, 30), (190, 20, 160), (10, 170, 220), (200, 170, 20)]
    for index, candidate in enumerate(result["candidates"]):
        points = np.rint(candidate["points"]).astype(int)
        color = colors[index % len(colors)]
        for x, y in points:
            cv2.circle(overlay, (int(x), int(y)), 1, color, -1)
        cv2.putText(overlay, str(index + 1), tuple(points[len(points) // 6]), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 2)
    for bounds, color in ((case["plot_bounds"], (220, 0, 200)), (result["roi"], (20, 170, 20))):
        x0, y0, x1, y1 = map(round, bounds)
        cv2.rectangle(overlay, (x0, y0), (x1 - 1, y1 - 1), color, 1)
    for group in case.get("landmarks", []):
        for point in group["points"]:
            cv2.drawMarker(overlay, tuple(map(round, point)), (0, 0, 0), cv2.MARKER_CROSS, 12, 1)
    cv2.imwrite(str(target), overlay)


def write_calibration_overlay(image, result, target):
    overlay = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    for axis, color in (("x", (10, 125, 230)), ("y", (190, 155, 0))):
        for point in result["axes"][axis]["points"]:
            x, y = map(round, point["pixel"])
            cv2.drawMarker(overlay, (x, y), color, cv2.MARKER_CROSS, 15, 2)
            cv2.putText(overlay, f'{axis.upper()} {point["value"]:g}', (x + 3, max(14, y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1, cv2.LINE_AA)
    cv2.imwrite(str(target), overlay)


def write_gallery(report, target):
    escape = html.escape
    cards = []
    for record in report["results"]:
        case, result, metrics = record["case"], record["result"], record["metrics"]
        state = "PASS" if metrics["extraction_checks_passed"] else "REVIEW / FAIL"
        calibration = record.get("calibration")
        calibration_html = ""
        if calibration:
            nx, ny = (len(calibration["axes"][axis]["points"]) for axis in ("x", "y"))
            calibration_html = (f'<p>Local OCR suggestions: {nx} X / {ny} Y ticks. All calibration proposals require review.</p>'
                                f'<img loading="lazy" src="{record["calibration_overlay"]}" alt="Suggested calibration ticks">')
        cards.append(f'<article><h2>{escape(case["id"])} · {state}</h2>'
                     f'<p>{case["modality"]} · {case["topology"]} · {record["split"]} · '
                     f'{metrics["actual_curves"]}/{metrics["expected_curves"]} curves · ROI IoU {metrics["roi_iou"]:.3f} · '
                     f'preset {result["preset"]} · threshold {result["threshold"]} · {record["seconds"]:.2f}s</p>'
                     f'<a href="{record["source_url"]}">Patent source · Fig. {escape(case["figure"])}</a>'
                     f'<img loading="lazy" src="{record["overlay"]}" alt="Extraction overlay for {escape(case["id"])}">'
                     + calibration_html + f'<details><summary>Settings, annotations and checks</summary><pre>{escape(json.dumps({"annotations":case,"metrics":metrics,"settings":{k:v for k,v in result.items() if k!="candidates"},"calibration":calibration},indent=2))}</pre></details></article>')
    reviews = ""
    if (target.parent.parent / "native-review/review.json").is_file():
        reviews = ('<p>Native Qt review examples: <a href="../native-review/scanned_grid_pxrd-calibration-review.png">Original grid scan calibration</a> · '
                   '<a href="../native-review/stacked_black_pxrd-calibration-review.png">Original stacked scan calibration</a> · '
                   '<a href="../native-review/crossed-curves-review.png">Crossed-curve separation review</a>.</p>')
    target.write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Patent plot digitizer benchmark</title>'
                     '<style>body{font:15px system-ui;margin:24px auto;max-width:1200px;background:#f3f5f7;color:#172332}h1{font-size:25px}h2{font-size:18px}article{padding:18px;margin:18px 0;background:white;border:1px solid #ccc;border-radius:8px}img{display:block;max-width:100%;margin:12px auto}pre{white-space:pre-wrap;overflow-wrap:anywhere}a{color:#145ac4}</style>'
                     '<h1>Automatic mode: sourced patent figures</h1><p>Green = automatic ROI; magenta = annotated plot bounds. Dots = observed samples; black crosses = sparse reference landmarks. Missing columns are not interpolated.</p>'
                     '<p>Count, bounds and landmarks are limited checks, not whole-curve accuracy. Crossings may switch identity. Axis calibration, page rotation and panel selection remain outside this benchmark.</p>'
                     + reviews + '<pre>' + escape(json.dumps({"summary": report["summary"], "audit": report["audit"]}, indent=2)) + '</pre>'
                     + ''.join(cards) + '</html>', encoding="utf8")


def summarize(records):
    groups = [check for record in records for check in record["metrics"]["landmark_checks"]]
    return {"figures": len(records), "modalities": dict(Counter(record["case"]["modality"] for record in records)),
            "extraction_checks_passed": sum(record["metrics"]["extraction_checks_passed"] for record in records),
            "count_correct": sum(record["metrics"]["count_match"] is True for record in records),
            "count_evaluable": sum(record["metrics"]["count_match"] is not None for record in records),
            "roi_correct": sum(record["metrics"]["roi_match"] for record in records),
            "preset_correct": sum(record["metrics"]["preset_match"] for record in records),
            "landmark_groups_correct": sum(group["passed"] for group in groups), "landmark_groups": len(groups),
            "overlap_cases": sum(record["case"]["requires_overlap_review"] for record in records),
            "overlap_flagged": sum(record["metrics"]["overlap_review_detected"] for record in records if record["case"]["requires_overlap_review"]),
            "unexpected_overlap_flags": sum(record["metrics"]["overlap_review_detected"] for record in records if not record["case"]["requires_overlap_review"]),
            "calibration_proposals_ready": sum(record.get("calibration", {}).get("ready", False) for record in records),
            "local_ocr_seconds": round(sum(record.get("calibration_seconds", 0) for record in records), 2),
            "seconds": round(sum(record["seconds"] for record in records), 2)}


def run(manifest_path, output, split="all", calibration=False):
    manifest = json.loads(manifest_path.read_text(encoding="utf8"))
    sources = {source["id"]: source for source in manifest["sources"]}
    images = []
    for case in manifest["cases"]:
        path = manifest_path.parent / case["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != case["sha256"]:
            raise ValueError(f'Fixture checksum mismatch: {case["id"]}')
        images.append(cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB))
    audit = audit_corpus(manifest, images)
    if audit["exact_duplicate_pairs"] or audit["family_split_leaks"]:
        raise ValueError(f"Duplicate or family leakage: {audit}")
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for case, image in zip(manifest["cases"], images):
        source = sources[case["source_id"]]
        if split != "all" and source["split"] != split:
            continue
        started = time.perf_counter()
        result = extract_plot_curves(image)
        elapsed = time.perf_counter() - started
        metrics = measure_case(case, result)
        overlay = case["id"] + "-overlay.png"
        write_overlay(image, case, result, output / overlay)
        record = {"case": case, "split": source["split"], "source_url": source["page_url"],
                  "result": result, "metrics": metrics, "seconds": round(elapsed, 3), "overlay": overlay}
        records.append(record)
        if calibration:
            calibration_started = time.perf_counter()
            try:
                record["calibration"] = suggest_axis_calibration(image, result["roi"])
                record["calibration_overlay"] = case["id"] + "-calibration.png"
                write_calibration_overlay(image, record["calibration"], output / record["calibration_overlay"])
            except RuntimeError as error:
                record["calibration_error"] = str(error)
            record["calibration_seconds"] = round(time.perf_counter() - calibration_started, 3)
        print(f'{case["id"]}: {metrics["actual_curves"]}/{metrics["expected_curves"]} curves, IoU={metrics["roi_iou"]:.3f}, {result["preset"]}, t={result["threshold"]}, {elapsed:.2f}s', flush=True)
    summary = summarize(records)
    report = {"manifest": str(manifest_path), "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
              "extractor_sha256": hashlib.sha256((APP_ROOT / "plot_digitizer/digitizer_core.py").read_bytes()).hexdigest(),
              "calibration_module_sha256": hashlib.sha256((APP_ROOT / "plot_digitizer/axis_calibration.py").read_bytes()).hexdigest(),
              "ocr_adapter_sha256": hashlib.sha256((APP_ROOT / "plot_digitizer/axis_ocr.py").read_bytes()).hexdigest(),
              "ocr_helper_sha256": hashlib.sha256((APP_ROOT / "plot_digitizer/windows_axis_ocr.ps1").read_bytes()).hexdigest(),
              "versions": {"opencv": cv2.__version__, "numpy": np.__version__},
              "audit": audit, "summary": summary,
              "splits": {name: summarize([record for record in records if record["split"] == name])
                         for name in sorted({source["split"] for source in sources.values()})}, "results": records}
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
    write_gallery(report, output / "index.html")
    print(json.dumps(summary), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=APP_ROOT / "output/patent_digitizer_benchmark/current")
    parser.add_argument("--split", choices=("all", "examples"), default="all")
    parser.add_argument("--calibration", action="store_true", help="Also run installed local OCR and show editable axis-tick proposals.")
    args = parser.parse_args()
    run(args.manifest, args.output, args.split, args.calibration)
