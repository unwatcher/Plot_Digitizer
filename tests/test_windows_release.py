"""Exercise offline installation, failed replacements and reproducible archives."""

import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


installer = load("release_installer_test", "release/windows/install_offline_dependencies.py")
builder = load("release_builder_test", "scripts/build_windows_release.py")
publisher = load("release_publisher_test", "scripts/publish_windows_release.py")


class WindowsReleaseTests(unittest.TestCase):
    def fake_wheel(self, root, contents=None):
        wheels = root / "wheels"
        wheels.mkdir()
        wheel = wheels / "sample-1.0-py3-none-any.whl"
        with zipfile.ZipFile(wheel, "w") as archive:
            for name, data in (contents or {"sample.py": "value=42\n"}).items():
                archive.writestr(name, data)
        return wheels, {wheel.name: hashlib.sha256(wheel.read_bytes()).hexdigest()}

    def test_install_restores_only_the_private_runtime_using_local_wheels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            wheels, expected = self.fake_wheel(root)
            target = root / "runtime_deps/py314"
            target.mkdir(parents=True)
            (target / "old.txt").write_text("old")
            with patch.object(installer, "EXPECTED_WHEELS", expected), patch.object(installer, "check_interpreter"):
                installer.install(target, wheels)
            self.assertEqual((target / "sample.py").read_text(), "value=42\n")
            self.assertFalse((target / "old.txt").exists())
            self.assertEqual([p.name for p in target.parent.iterdir()], ["py314"])

    def test_missing_or_corrupt_wheel_preserves_the_existing_runtime(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            wheels, expected = self.fake_wheel(root)
            target = root / "runtime_deps/py314"
            target.mkdir(parents=True)
            marker = target / "keep.txt"; marker.write_text("keep")
            wheel = next(wheels.iterdir()); wheel.write_bytes(b"corrupt")
            with patch.object(installer, "EXPECTED_WHEELS", expected), patch.object(installer, "check_interpreter"):
                with self.assertRaisesRegex(RuntimeError, "verification failed"):
                    installer.install(target, wheels)
            self.assertEqual(marker.read_text(), "keep")

    def test_failed_activation_rolls_back_before_reporting_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            wheels, expected = self.fake_wheel(root)
            target = root / "runtime_deps/py314"; target.mkdir(parents=True)
            (target / "keep.txt").write_text("keep")
            original = Path.replace
            def replace(source, destination):
                if source.name.startswith(".plotdigitizer-stage-") and Path(destination).name == "py314":
                    raise PermissionError("locked")
                return original(source, destination)
            with patch.object(installer, "EXPECTED_WHEELS", expected), patch.object(installer, "check_interpreter"), patch.object(Path, "replace", replace):
                with self.assertRaisesRegex(RuntimeError, "Close PlotDigitizer"):
                    installer.install(target, wheels)
            self.assertEqual((target / "keep.txt").read_text(), "keep")

    def test_archive_traversal_is_rejected_before_activation(self):
        for name in ("../outside.txt", "C:/outside.txt", "..\\outside.txt", "/outside.txt"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                wheels, expected = self.fake_wheel(root, {name: "bad"})
                with patch.object(installer, "EXPECTED_WHEELS", expected), patch.object(installer, "check_interpreter"):
                    with self.assertRaises(ValueError):
                        installer.install(root / "runtime_deps/py314", wheels)
                self.assertFalse((root / "outside.txt").exists())

    def test_cleanup_and_install_reject_unowned_targets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(installer, "check_interpreter"):
                with self.assertRaisesRegex(ValueError, "only replaces"):
                    installer.install(root, root)
            with patch.object(installer.shutil, "rmtree") as remove:
                with self.assertRaises(ValueError):
                    installer._remove_owned(root, root, ".plotdigitizer-stage-")
            remove.assert_not_called()

    def test_long_runtime_paths_work_without_global_python_changes(self):
        # TemporaryDirectory itself also needs the extended Windows prefix
        # when cleaning up the intentionally over-MAX_PATH test tree.
        with tempfile.TemporaryDirectory(dir=installer.filesystem_path(Path(tempfile.gettempdir()))) as temporary:
            root = Path(temporary)
            wheels, expected = self.fake_wheel(root)
            target = root.joinpath(*(["long-folder-name-with-spaces"] * 11), "runtime_deps", "py314")
            with patch.object(installer, "EXPECTED_WHEELS", expected), patch.object(installer, "check_interpreter"):
                installer.install(target, wheels)
            self.assertEqual((installer.filesystem_path(target) / "sample.py").read_text(), "value=42\n")

    def test_archive_is_stable_and_excludes_python_caches(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root / "app"; folder.mkdir()
            source = folder / "app.py"; source.write_text("print('ok')\n")
            (folder / "__pycache__").mkdir(); (folder / "__pycache__/junk.pyc").write_bytes(b"cache")
            first, second = root / "first.zip", root / "second.zip"
            builder.write_archive(folder, first)
            os.utime(source, (1, 1))
            builder.write_archive(folder, second)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            with zipfile.ZipFile(first) as archive:
                self.assertEqual(archive.namelist(), ["app/app.py"])

    def test_incomplete_download_is_not_cached(self):
        response = io.BytesIO(b"PK incomplete")
        response.headers = {"Content-Length": "9999"}
        with tempfile.TemporaryDirectory() as temporary, patch.object(builder.urllib.request, "urlopen", return_value=response):
            with self.assertRaisesRegex(ValueError, "incomplete"):
                builder.ensure_python(Path(temporary))
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_source_includes_the_preview_and_license_but_not_local_outputs(self):
        files = {p.relative_to(ROOT).as_posix() for p in builder.source_files()}
        self.assertIn("LICENSE", files)
        self.assertIn("docs/images/pxrd-extracted.png", files)
        self.assertIn("examples/pxrd-extracted.csv", files)
        self.assertFalse(any(p.startswith(("output/", "dist/", ".cache/")) for p in files))

    def test_publisher_rejects_mismatched_source_before_calling_github(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "release-manifest.json").write_text(json.dumps({"source_revision": "other"}))
            with patch.object(publisher, "gh") as github:
                with self.assertRaisesRegex(ValueError, "checked-out commit"):
                    publisher.publish("owner/repo", "expected", root)
            github.assert_not_called()

    def release_files(self, root):
        artifacts = []
        for variant in ("ready", "offline-wheels"):
            name = variant + ".zip"
            (root / name).write_bytes((variant + "-test-artifact").encode())
            artifacts.append({"file": name, "variant": variant, "verified_after_extraction": True,
                              "sha256": publisher.digest(root / name)})
        (root / "release-manifest.json").write_text(json.dumps({"source_revision": "expected", "version": "test", "artifacts": artifacts}))
        (root / "example-pxrd-original.png").write_bytes(b"original")
        (root / "example-pxrd-extracted.png").write_bytes(b"extracted")
        (root / "RELEASE_NOTES.md").write_text("Test release")
        names = [a["file"] for a in artifacts] + ["release-manifest.json", "example-pxrd-original.png", "example-pxrd-extracted.png"]
        (root / "SHA256SUMS.txt").write_text("".join(publisher.digest(root / name) + "  " + name + "\n" for name in names))
        return names + ["SHA256SUMS.txt"]

    def test_preexisting_mismatched_tag_without_a_release_makes_no_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.release_files(root)
            with patch.object(publisher, "gh", return_value=json.dumps({"object": {"type": "commit", "sha": "wrong"}})) as github:
                with self.assertRaisesRegex(ValueError, "tag points to another commit"):
                    publisher.publish("owner/repo", "expected", root)
            self.assertEqual(github.call_count, 1)
            self.assertEqual(github.call_args.args[0], "api")

    def test_annotated_tags_are_dereferenced_to_their_source_commit(self):
        responses = [json.dumps({"object": {"type": "tag", "sha": "annotation"}}),
                     json.dumps({"object": {"type": "commit", "sha": "expected"}})]
        with patch.object(publisher, "gh", side_effect=responses):
            self.assertEqual(publisher.resolve_tag_commit("owner/repo", "vtest"), "expected")

    def test_changed_preview_asset_cannot_be_published_with_stale_checksums(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.release_files(root)
            (root / "example-pxrd-extracted.png").write_bytes(b"changed")
            with patch.object(publisher, "gh") as github:
                with self.assertRaisesRegex(ValueError, "asset checksum mismatch"):
                    publisher.publish("owner/repo", "expected", root)
            github.assert_not_called()

    def test_published_assets_and_tag_are_verified_after_creation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            names = self.release_files(root)
            release = {"html_url": "https://github.com/owner/repo/releases/tag/vtest",
                       "assets": [{"name": name, "size": (root / name).stat().st_size,
                                   "digest": "sha256:" + publisher.digest(root / name)} for name in names]}
            responses = [None, None, "created", json.dumps(release),
                         json.dumps({"object": {"type": "commit", "sha": "expected"}})]
            with patch.object(publisher, "gh", side_effect=responses) as github:
                publisher.publish("owner/repo", "expected", root)
            self.assertEqual(github.call_args_list[2].args[:2], ("release", "create"))


if __name__ == "__main__":
    unittest.main()
