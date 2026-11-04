"""Reproduce curated patent plot inputs from their original public PDFs.

Downloads are optional: tests and the benchmark use committed PNG fixtures.
Rendering requires Poppler's pdftoppm; pass --pdftoppm if it is not on PATH.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request

APP_ROOT = Path(__file__).resolve().parents[1]
runtime = APP_ROOT / "runtime_deps/py314"
if sys.platform == "win32" and runtime.is_dir():
    sys.path.insert(0, str(runtime))
from PIL import Image


def reproduce(manifest_path, cache, output, renderer):
    manifest = json.loads(manifest_path.read_text(encoding="utf8"))
    cache.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    sources = {source["id"]: source for source in manifest["sources"]}
    for source in sources.values():
        path = cache / (source["id"] + ".pdf")
        if not path.exists():
            request = urllib.request.Request(source["pdf_url"], headers={"User-Agent": "CrystalTrace-patent-plot-benchmark/1"})
            with urllib.request.urlopen(request, timeout=60) as response:
                data = response.read(40 * 1024 * 1024 + 1)
            if len(data) > 40 * 1024 * 1024 or not data.startswith(b"%PDF-"):
                raise ValueError(f'Unexpected PDF response: {source["id"]}')
            if hashlib.sha256(data).hexdigest() != source["pdf_sha256"]:
                raise ValueError(f'PDF checksum changed: {source["id"]}')
            path.write_bytes(data)
        if hashlib.sha256(path.read_bytes()).hexdigest() != source["pdf_sha256"]:
            raise ValueError(f'Cached PDF checksum mismatch: {source["id"]}')
    rendered = {}
    records = []
    for case in manifest["cases"]:
        key = (case["source_id"], case["pdf_page_1based"], case["dpi"], case["rotation_clockwise"])
        if key not in rendered:
            source, page, dpi, rotation = key
            prefix = cache / f"{source}-page{page}-{dpi}dpi"
            subprocess.run([renderer, "-f", str(page), "-l", str(page), "-singlefile", "-r", str(dpi),
                            "-png", str(cache / (source + ".pdf")), str(prefix)], check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            with Image.open(str(prefix) + ".png") as image:
                rendered[key] = image.convert("RGB").rotate(-rotation, expand=True)
        target = output / case["file"]
        rendered[key].crop(case["crop_ltrb"]).save(target, optimize=True)
        checksum = hashlib.sha256(target.read_bytes()).hexdigest()
        # Rasterizers can change anti-aliasing between versions. Keep the
        # committed fixture checksums authoritative for regression runs.
        records.append({"id": case["id"], "sha256": checksum, "matches_fixture": checksum == case["sha256"]})
        print(case["id"], "matches fixture" if records[-1]["matches_fixture"] else "rasterizer differs", flush=True)
    (output / "reproduction.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf8")
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=APP_ROOT / "examples/patent_plots/manifest.json")
    parser.add_argument("--cache", type=Path, default=APP_ROOT / "output/patent_digitizer_benchmark/sources")
    parser.add_argument("--output", type=Path, default=APP_ROOT / "output/patent_digitizer_benchmark/reproduced")
    parser.add_argument("--pdftoppm", default=shutil.which("pdftoppm"))
    args = parser.parse_args()
    if not args.pdftoppm:
        parser.error("Install Poppler or provide --pdftoppm with its executable path.")
    reproduce(args.manifest, args.cache, args.output, args.pdftoppm)
