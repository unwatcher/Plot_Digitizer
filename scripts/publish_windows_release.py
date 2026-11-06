"""Upload verified release outputs using GitHub CLI authentication from Actions."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def gh(*arguments, allow_missing=False):
    result = subprocess.run(["gh", *arguments], capture_output=True, text=True)
    if result.returncode:
        if allow_missing and "404" in result.stderr:
            return None
        raise RuntimeError("GitHub release command failed: " + result.stderr.strip())
    return result.stdout


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def resolve_tag_commit(repository, tag):
    response = gh("api", "repos/" + repository + "/git/ref/tags/" + tag, allow_missing=True)
    if response is None:
        return None
    target = json.loads(response)["object"]
    visited = set()
    for _ in range(16):
        if target["type"] == "commit":
            return target["sha"]
        if target["type"] != "tag" or target["sha"] in visited:
            raise ValueError("Release tag does not resolve to a commit.")
        visited.add(target["sha"])
        target = json.loads(gh("api", "repos/" + repository + "/git/tags/" + target["sha"]))["object"]
    raise ValueError("Release tag nesting is too deep.")


def publish(repository, source_revision, directory):
    manifest = json.loads((directory / "release-manifest.json").read_text(encoding="utf8"))
    if manifest["source_revision"] != source_revision:
        raise ValueError("Release files do not belong to the checked-out commit.")
    artifacts = manifest["artifacts"]
    if len(artifacts) != 2 or {a["variant"] for a in artifacts} != {"ready", "offline-wheels"}:
        raise ValueError("Both release package variants are required.")
    for artifact in artifacts:
        name = artifact["file"]
        if Path(name).name != name or not artifact["verified_after_extraction"]:
            raise ValueError("Unverified release artifact.")
        if digest(directory / name) != artifact["sha256"]:
            raise ValueError("Release package checksum mismatch: " + name)
    names = [artifact["file"] for artifact in artifacts] + [
        "SHA256SUMS.txt", "release-manifest.json", "example-pxrd-original.png", "example-pxrd-extracted.png"]
    for name in names:
        if not (directory / name).is_file():
            raise ValueError("Release file is missing: " + name)
    checked_names = set()
    for line in (directory / "SHA256SUMS.txt").read_text(encoding="ascii").splitlines():
        expected, name = line.split("  ", 1)
        if name not in names or name == "SHA256SUMS.txt" or name in checked_names:
            raise ValueError("Unexpected or duplicate release checksum entry.")
        if digest(directory / name) != expected:
            raise ValueError("Release asset checksum mismatch: " + name)
        checked_names.add(name)
    if checked_names != set(names) - {"SHA256SUMS.txt"}:
        raise ValueError("Release checksum file is incomplete.")
    tag = "v" + manifest["version"]
    existing_commit = resolve_tag_commit(repository, tag)
    if existing_commit is not None and existing_commit != source_revision:
        raise ValueError("The existing release tag points to another commit. Bump the release version.")
    response = gh("api", "repos/" + repository + "/releases/tags/" + tag, allow_missing=True)
    if response is None:
        gh("release", "create", tag, "--repo", repository, "--target", source_revision,
           "--prerelease", "--title", "PlotDigitizer " + tag + " - Windows x64",
           "--notes-file", str(directory / "RELEASE_NOTES.md"), *[str(directory / name) for name in names])
    else:
        release = json.loads(response)
        if release["target_commitish"] != source_revision:
            raise ValueError("The existing release belongs to another commit. Bump the version before publishing.")
        existing = {asset["name"]: asset for asset in release["assets"]}
        for name in names:
            expected = "sha256:" + digest(directory / name)
            if name in existing:
                if existing[name].get("digest") != expected:
                    raise ValueError("An existing release asset differs; it will not be overwritten: " + name)
            else:
                gh("release", "upload", tag, str(directory / name), "--repo", repository)
    # Confirm server hashes and sizes, including after an interrupted-upload retry.
    release = json.loads(gh("api", "repos/" + repository + "/releases/tags/" + tag))
    if resolve_tag_commit(repository, tag) != source_revision:
        raise RuntimeError("The published tag does not match the verified source commit.")
    remote = {asset["name"]: asset for asset in release["assets"]}
    for name in names:
        path = directory / name
        if (name not in remote or remote[name]["size"] != path.stat().st_size
                or remote[name].get("digest") != "sha256:" + digest(path)):
            raise RuntimeError("The uploaded release asset failed server-side checksum verification: " + name)
    print(release["html_url"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--dist", type=Path, default=Path("dist"))
    args = parser.parse_args()
    publish(args.repository, args.source_revision, args.dist.resolve())
