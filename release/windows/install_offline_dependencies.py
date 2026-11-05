"""Restore the private runtime from verified local wheels; no pip or network."""

from __future__ import annotations

import argparse
import hashlib
import json
import ntpath
from pathlib import Path, PurePosixPath
import shutil
import stat
import sys
import sysconfig
import uuid
import zipfile


EXPECTED_WHEELS = json.loads(Path(__file__).with_name("wheel-manifest.json").read_text(encoding="utf8"))


def filesystem_path(path):
    resolved = str(Path(path).resolve())
    if sys.platform == "win32" and not resolved.startswith("\\\\?\\"):
        return Path("\\\\?\\UNC\\" + resolved[2:] if resolved.startswith("\\\\") else "\\\\?\\" + resolved)
    return Path(resolved)


def check_interpreter():
    if (sys.implementation.name != "cpython" or sys.version_info[:2] != (3, 14)
            or sys.platform != "win32" or sysconfig.get_platform() != "win-amd64"
            or sysconfig.get_config_var("Py_GIL_DISABLED")):
        raise RuntimeError("Standard Windows x64 CPython 3.14 is required; use the bundled python314 runtime.")


def sha256(path):
    digest = hashlib.sha256()
    with filesystem_path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_wheelhouse(wheelhouse):
    wheelhouse = filesystem_path(wheelhouse)
    if not wheelhouse.is_dir():
        raise RuntimeError("Offline wheels are missing. Extract the offline-wheels download first.")
    missing = [name for name in EXPECTED_WHEELS if not (wheelhouse / name).is_file()]
    invalid = [name for name, expected in EXPECTED_WHEELS.items()
               if name not in missing and sha256(wheelhouse / name) != expected]
    if missing or invalid:
        raise RuntimeError("Offline wheel verification failed: missing=" + repr(missing) + "; invalid=" + repr(invalid))


def safe_extract(archive, destination):
    """Reject paths outside the destination and symbolic-link ZIP entries."""
    for entry in archive.infolist():
        name = entry.filename.replace("\\", "/")
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or ntpath.splitdrive(name)[0]:
            raise ValueError("Archive contains an unsafe path: " + entry.filename)
        if stat.S_ISLNK(entry.external_attr >> 16):
            raise ValueError("Archive symbolic links are not supported.")
    archive.extractall(filesystem_path(destination))


def _remove_owned(path, parent, prefix, ignore_errors=False):
    path, parent = Path(path).resolve(), Path(parent).resolve()
    if path.parent != parent or not path.name.startswith(prefix):
        raise ValueError("Refusing cleanup outside the owned runtime staging directory.")
    if filesystem_path(path).exists():
        shutil.rmtree(filesystem_path(path), ignore_errors=ignore_errors)


def install(target, wheelhouse):
    check_interpreter()
    target = Path(target).absolute()
    if target.name != "py314" or target.parent.name != "runtime_deps":
        raise ValueError("The installer only replaces runtime_deps/py314.")
    if any(path.is_symlink() or getattr(path, "is_junction", lambda: False)()
           for path in (target, target.parent)):
        raise ValueError("Runtime directories must not be symbolic links or junctions.")
    verify_wheelhouse(wheelhouse)
    parent = target.parent.resolve()
    target = parent / "py314"
    filesystem_path(parent).mkdir(parents=True, exist_ok=True)
    temporary = parent / (".plotdigitizer-stage-" + uuid.uuid4().hex)
    backup = parent / (".plotdigitizer-backup-" + uuid.uuid4().hex)
    filesystem_path(temporary).mkdir()
    target_moved = False
    try:
        for filename in EXPECTED_WHEELS:
            print("Installing " + filename, flush=True)
            with zipfile.ZipFile(filesystem_path(Path(wheelhouse) / filename)) as archive:
                safe_extract(archive, temporary)
        # All rename targets are direct children of the checked runtime parent.
        if filesystem_path(target).exists():
            filesystem_path(target).replace(filesystem_path(backup))
            target_moved = True
        filesystem_path(temporary).replace(filesystem_path(target))
    except Exception as error:
        if target_moved and not filesystem_path(target).exists() and filesystem_path(backup).exists():
            filesystem_path(backup).replace(filesystem_path(target))
        if isinstance(error, PermissionError):
            raise RuntimeError("Close PlotDigitizer and other processes using its runtime, then retry.") from error
        raise
    finally:
        _remove_owned(temporary, parent, ".plotdigitizer-stage-", ignore_errors=True)
    if target_moved:
        _remove_owned(backup, parent, ".plotdigitizer-backup-", ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check local wheel hashes without installing.")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    try:
        if args.check:
            verify_wheelhouse(root / "wheelhouse_py314")
        else:
            install(root / "runtime_deps/py314", root / "wheelhouse_py314")
    except Exception as error:
        print("ERROR: " + str(error), file=sys.stderr)
        return 1
    print("Offline runtime verification complete." if args.check else "Offline runtime installed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
