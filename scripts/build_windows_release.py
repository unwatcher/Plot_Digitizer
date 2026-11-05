"""Build and verify portable Windows folders and hash-pinned offline wheels."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SUPPORT = ROOT / "release/windows"
CONFIG = json.loads((SUPPORT / "config.json").read_text(encoding="utf8"))
spec = importlib.util.spec_from_file_location("windows_offline_installer", SUPPORT / "install_offline_dependencies.py")
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


def sha256(path):
    return installer.sha256(path)


def source_files():
    """Explicit source roots exclude local outputs, wheels, caches and secrets."""
    paths = [ROOT / name for name in ("README.md", "LICENSE", "THIRD_PARTY_NOTICES.md",
                                      "plot_digitizer_auto.py", "requirements.txt", "requirements-core.txt")]
    for directory in ("plot_digitizer", "scripts", "tests", "examples", "docs", "release"):
        paths.extend(p for p in (ROOT / directory).rglob("*") if p.is_file()
                     and "__pycache__" not in p.parts and p.suffix.lower() in
                     {".py", ".ps1", ".bat", ".txt", ".json", ".md", ".png", ".csv"})
    return sorted(set(paths))


def ensure_python(cache):
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / Path(CONFIG["python"]["url"]).name
    if not path.exists():
        request = urllib.request.Request(CONFIG["python"]["url"], headers={"User-Agent": "PlotDigitizer-release-builder"})
        with urllib.request.urlopen(request, timeout=60) as response:
            chunks, size = [], 0
            while block := response.read(1024 * 1024):
                size += len(block)
                if size > 40 * 1024 * 1024:
                    raise ValueError("Portable Python download exceeded its size limit.")
                chunks.append(block)
            expected_length = response.headers.get("Content-Length")
            if expected_length and size != int(expected_length):
                raise ValueError("Portable Python download was incomplete.")
            data = b"".join(chunks)
        if len(data) > 40 * 1024 * 1024 or not data.startswith(b"PK"):
            raise ValueError("Unexpected portable Python response.")
        if hashlib.sha256(data).hexdigest() != CONFIG["python"]["sha256"]:
            raise ValueError("Portable Python download failed SHA-256 verification.")
        path.write_bytes(data)
    if sha256(path) != CONFIG["python"]["sha256"]:
        raise ValueError("Cached portable Python failed SHA-256 verification.")
    return path


def write_archive(folder, target):
    """Stable ordering/timestamps keep retries of one source revision identical."""
    folder = installer.filesystem_path(folder)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with zipfile.ZipFile(temporary, "w", allowZip64=True) as archive:
        for path in sorted(folder.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo", ".log"}:
                continue
            relative = (Path(folder.name) / path.relative_to(folder)).as_posix()
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_STORED if path.suffix == ".whl" else zipfile.ZIP_DEFLATED
            with installer.filesystem_path(path).open("rb") as source, archive.open(info, "w", force_zip64=True) as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
    temporary.replace(target)


def prepare_base(folder, python_zip, revision, variant):
    folder.mkdir(parents=True)
    files = source_files()
    for path in files:
        destination = folder / path.relative_to(ROOT)
        installer.filesystem_path(destination.parent).mkdir(parents=True, exist_ok=True)
        shutil.copy2(installer.filesystem_path(path), installer.filesystem_path(destination))
    for name in ("app_launcher.py", "install_offline_dependencies.py", "verify_offline_bundle.py",
                 "launch_plot_digitizer.bat", "install_dependencies.bat", "find_native_python314.ps1",
                 "wheel-manifest.json", "requirements-offline.txt"):
        shutil.copy2(SUPPORT / name, folder / name)
    with zipfile.ZipFile(python_zip) as archive:
        installer.safe_extract(archive, folder / "python314")
    metadata = {"version": CONFIG["version"], "variant": variant, "platform": CONFIG["platform"],
                "source_revision": revision, "python": CONFIG["python"],
                "wheels": installer.EXPECTED_WHEELS,
                "source_sha256": {path.relative_to(ROOT).as_posix(): sha256(path) for path in files}}
    (folder / "release-info.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf8")
    (folder / "START_HERE.txt").write_text(
        "PlotDigitizer " + CONFIG["version"] + " - Windows x64\n\n"
        "Extract the complete ZIP to a writable folder. Portable Python is included.\n"
        "Double-click launch_plot_digitizer.bat.\n"
        "For the offline-wheels edition, run install_dependencies.bat first (or launch to install automatically).\n"
        "Installation uses the verified local wheels only; no pip, internet or administrator access is needed.\n"
        "For curve selection and axis calibration, see README.md.\n"
        "Windows OCR needs an OCR language already installed in Windows; manual calibration remains available.\n",
        encoding="utf8")
    return metadata


def verify_folder(folder):
    command = [str(folder / "python314/python.exe"), "-I", "-B", str(folder / "verify_offline_bundle.py")]
    environment = dict(os.environ, QT_QPA_PLATFORM="offscreen", PYTHONPATH="__invalid_external_runtime__")
    # Font cache stays outside the folder that is about to be archived.
    environment["MPLCONFIGDIR"] = str(folder.parent / "matplotlib-cache")
    result = subprocess.run(command, cwd=folder.parent, env=environment,
                            capture_output=True, text=True, timeout=120)
    if result.returncode:
        raise RuntimeError("Packaged verification failed:\n" + result.stdout + result.stderr)
    print(result.stdout.strip(), flush=True)


def verify_archive(path, offline=False):
    # Verify actual ZIP contents after extraction, not just the staging folder.
    workspace = installer.filesystem_path(ROOT / ".tmp")
    workspace.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="package verify spaces-", dir=workspace) as temporary:
        root = Path(temporary).resolve()
        if not root.is_relative_to(workspace.resolve()):
            raise ValueError("Package verification directory escaped its workspace.")
        with zipfile.ZipFile(path) as archive:
            installer.safe_extract(archive, root)
        folders = [p for p in root.iterdir() if p.is_dir()]
        if len(folders) != 1:
            raise ValueError("Expected one application folder in the ZIP.")
        folder = folders[0]
        if offline:
            command = [str(folder / "python314/python.exe"), "-I", "-B", str(folder / "install_offline_dependencies.py")]
            result = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=240)
            if result.returncode:
                raise RuntimeError("Offline wheel installation failed:\n" + result.stdout + result.stderr)
        verify_folder(folder)


def release_notes(repository, revision):
    tag = "v" + CONFIG["version"]
    base = "https://github.com/" + repository + "/releases/download/" + tag
    point_count = json.loads((ROOT / "examples/pxrd-extraction.json").read_text(encoding="utf8"))["observed_points"]
    return ("# PlotDigitizer " + tag + " for Windows x64\n\n"
            "Two portable folders are provided; both include Python " + CONFIG["python"]["version"] + ".\n\n"
            "- **Ready-to-run:** extract the folder and double-click `launch_plot_digitizer.bat`.\n"
            "- **Offline wheels:** extract, run `install_dependencies.bat`, then launch. All 16 wheel hashes are checked; installation uses no pip or internet.\n\n"
            "Source revision: `" + revision + "`. Both ZIP contents are verified in fresh folders using only their bundled interpreter.\n\n"
            "## Original and extracted curve\n\n"
            "| Original PXRD figure | Automatic extraction |\n| --- | --- |\n"
            "| ![Original patent plot](" + base + "/example-pxrd-original.png) | ![Extracted curve](" + base + "/example-pxrd-extracted.png) |\n\n"
            "The preview contains " + str(point_count) + " observed points with reviewed printed-axis calibration. No smoothing or filling of missing columns is applied. The CSV and settings record are included.\n\n"
            "Four U.S. patent examples are bundled with source/reuse documentation. Same-color crossings, missed OCR labels and nonuniform axes can require manual correction; automatic diagnostics are not calibrated confidence probabilities. Windows OCR requires installed language support.\n\n"
            "See `SHA256SUMS.txt` and `release-manifest.json` for checksums and build provenance. Source uses the existing BSD 3-Clause license; Python/dependency licenses and the separate figure reuse note are included.\n")


def build(wheelhouse, cache, output, revision, repository="unwatcher/Plot_Digitizer"):
    installer.check_interpreter()
    installer.verify_wheelhouse(wheelhouse)
    python_zip = ensure_python(cache)
    output.mkdir(parents=True, exist_ok=True)
    workspace = installer.filesystem_path(ROOT / ".tmp")
    workspace.mkdir(exist_ok=True)
    artifacts = []
    with tempfile.TemporaryDirectory(prefix="windows-release-", dir=workspace) as temporary:
        staging = Path(temporary).resolve()
        if not staging.is_relative_to(workspace.resolve()):
            raise ValueError("Release staging directory escaped its workspace.")
        for variant in ("ready", "offline-wheels"):
            name = "PlotDigitizer-" + CONFIG["version"] + "-" + variant + "-win64"
            folder = staging / name
            metadata = prepare_base(folder, python_zip, revision, variant)
            if variant == "ready":
                installer.install(folder / "runtime_deps/py314", wheelhouse)
                verify_folder(folder)
            else:
                wheels = folder / "wheelhouse_py314"
                wheels.mkdir()
                for filename in installer.EXPECTED_WHEELS:
                    shutil.copy2(Path(wheelhouse) / filename, wheels / filename)
            target = output / (name + ".zip")
            write_archive(folder, target)
            verify_archive(target, offline=variant == "offline-wheels")
            artifacts.append({"file": target.name, "bytes": target.stat().st_size, "sha256": sha256(target),
                              "variant": variant, "verified_after_extraction": True})
            print("Verified " + target.name, flush=True)
    shutil.copy2(ROOT / "examples/patent_plots/us8198452b2-fig1.png", output / "example-pxrd-original.png")
    shutil.copy2(ROOT / "docs/images/pxrd-extracted.png", output / "example-pxrd-extracted.png")
    manifest = {"version": CONFIG["version"], "source_revision": revision, "platform": CONFIG["platform"],
                "python": CONFIG["python"], "wheels": installer.EXPECTED_WHEELS, "artifacts": artifacts,
                "source_sha256": metadata["source_sha256"], "verification": "Both actual ZIPs extracted and run with bundled Python; offline wheels installed without pip/network."}
    (output / "release-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf8")
    checksum_files = [output / a["file"] for a in artifacts] + [output / name for name in
                     ("example-pxrd-original.png", "example-pxrd-extracted.png", "release-manifest.json")]
    (output / "SHA256SUMS.txt").write_text("".join(sha256(p) + "  " + p.name + "\n" for p in checksum_files), encoding="ascii")
    (output / "RELEASE_NOTES.md").write_text(release_notes(repository, revision), encoding="utf8")
    print(json.dumps({"built": artifacts}), flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheelhouse", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=ROOT / ".cache/downloads")
    parser.add_argument("--output", type=Path, default=ROOT / "dist")
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--repository", default="unwatcher/Plot_Digitizer")
    args = parser.parse_args()
    build(args.wheelhouse.resolve(), args.cache.resolve(), args.output.resolve(), args.source_revision, args.repository)
