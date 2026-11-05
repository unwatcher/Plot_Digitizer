import copy
import hashlib
import json
from pathlib import Path
import unittest

import cv2
import numpy as np

from plot_digitizer.digitizer_core import extract_plot_curves, extract_dark_curve
from scripts.benchmark_plot_digitizer import audit_corpus, measure_case

ROOT = Path(__file__).resolve().parents[1] / "examples/patent_plots"

class PatentExampleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf8"))
        cls.images = [cv2.cvtColor(cv2.imread(str(ROOT / case["file"])), cv2.COLOR_BGR2RGB)
                      for case in cls.manifest["cases"]]

    def test_only_four_reviewed_examples_with_source_and_rights_provenance_are_bundled(self):
        self.assertEqual(len(self.manifest["cases"]), 4)
        self.assertEqual({c["modality"] for c in self.manifest["cases"]}, {"IR", "Raman", "PXRD"})
        self.assertEqual({p.name for p in ROOT.glob("*.png")}, {c["file"] for c in self.manifest["cases"]})
        for source in self.manifest["sources"]:
            self.assertEqual(source["jurisdiction"], "US")
            review = source["rights_review"]
            self.assertEqual(review["classification"], "screened-us-utility-patent-drawing")
            self.assertEqual(review["notice_search_hits"], [])
            self.assertTrue(review["policy_url"].startswith("https://www.uspto.gov/"))
            self.assertFalse(review["software_license_applies_to_figure"])
        for case in self.manifest["cases"]:
            self.assertEqual(hashlib.sha256((ROOT / case["file"]).read_bytes()).hexdigest(), case["sha256"])
        audit = audit_corpus(self.manifest, self.images)
        self.assertEqual(audit["family_count"], 2)
        self.assertFalse(audit["exact_duplicate_pairs"])
        self.assertFalse(audit["near_duplicate_candidates"])
        self.assertFalse(audit["family_split_leaks"])

    def test_example_roi_count_preset_and_visible_landmarks_remain_supported(self):
        for case, image in zip(self.manifest["cases"], self.images):
            with self.subTest(case=case["id"]):
                result = extract_plot_curves(image)
                metrics = measure_case(case, result)
                self.assertTrue(metrics["extraction_checks_passed"], metrics)
                self.assertFalse(result["confidence_calibrated"])
                if case["requires_overlap_review"]:
                    self.assertTrue(result["review_required"])

    def test_duplicate_audit_catches_resaved_resized_and_cross_split_copies(self):
        manifest = copy.deepcopy(self.manifest)
        manifest["cases"] = [manifest["cases"][0], dict(manifest["cases"][0], id="copy"),
                              dict(manifest["cases"][0], id="resized-copy")]
        image = self.images[0]
        images = [image, image.copy(), cv2.resize(image, None, fx=1.5, fy=1.5)]
        audit = audit_corpus(manifest, images)
        self.assertEqual(len(audit["exact_duplicate_pairs"]), 1)
        self.assertTrue(audit["near_duplicate_candidates"])
        second_source = dict(manifest["sources"][0], id="related-publication", split="development")
        manifest["sources"].append(second_source)
        manifest["cases"][1]["source_id"] = "related-publication"
        self.assertTrue(audit_corpus(manifest, images)["family_split_leaks"])

    def test_identity_checks_reject_high_coverage_envelopes_at_a_crossing(self):
        xs = np.arange(101)
        a, b = 20 + .6 * xs, 80 - .6 * xs
        case = {"plot_bounds": [0, 0, 101, 101], "count_evaluable": True, "expected_curves": 2,
                "expected_preset": "waterfall", "expected_polarity": "up", "requires_overlap_review": True,
                "landmarks": [{"label": "A", "points": [[0, 20], [100, 80]], "tolerance_px": 3},
                              {"label": "B", "points": [[0, 80], [100, 20]], "tolerance_px": 3}]}
        result = {"roi": [0, 0, 101, 101], "preset": "waterfall", "polarity": "up", "overlap_regions": [{}],
                  "candidates": [{"points": list(zip(xs, ys)), "coverage": 1}
                                 for ys in (np.minimum(a, b), np.maximum(a, b))]}
        metrics = measure_case(case, result)
        self.assertTrue(metrics["count_match"])
        self.assertTrue(metrics["roi_match"])
        self.assertFalse(metrics["extraction_checks_passed"])
        self.assertFalse(any(check["passed"] for check in metrics["landmark_checks"]))

    def test_steep_stroke_check_uses_adjacent_samples_without_bridging_missing_columns(self):
        case = {"plot_bounds": [0, 0, 101, 101], "count_evaluable": True, "expected_curves": 1,
                "expected_preset": "normal", "expected_polarity": "up", "requires_overlap_review": False,
                "landmarks": [{"label": "Steep edge", "points": [[50.5, 50]], "tolerance_px": 2}]}
        result = {"roi": [0, 0, 101, 101], "preset": "normal", "polarity": "up",
                  "candidates": [{"points": [[50, 20], [51, 80]]}]}
        self.assertTrue(measure_case(case, result)["landmark_checks"][0]["passed"])
        result["candidates"][0]["points"] = [[40, 20], [60, 80]]
        self.assertFalse(measure_case(case, result)["landmark_checks"][0]["passed"])


if __name__ == "__main__":
    unittest.main()
