"""Verify the sizes and SHA-256 checksums of the bundled processed datasets."""

import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LFS_HEADER = b"version https://git-lfs.github.com/spec/v1"


def main():
    manifest = json.loads((ROOT / "datasets" / "manifest.json").read_text(encoding="utf-8"))
    errors = []
    total = 0
    for entry in manifest["files"]:
        relative = Path(entry["path"])
        path = (ROOT / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to((ROOT / "datasets").resolve()):
            errors.append(f"Invalid manifest path: {entry['path']}")
            continue
        if not path.is_file():
            errors.append(f"Missing: {entry['path']}")
            continue
        with path.open("rb") as stream:
            if stream.read(len(LFS_HEADER)) == LFS_HEADER:
                errors.append(f"LFS pointer: {entry['path']}; run git lfs pull.")
                continue
            stream.seek(0)
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if path.stat().st_size != entry["bytes"]:
            errors.append(f"Size mismatch: {entry['path']}")
        if digest != entry["sha256"]:
            errors.append(f"Checksum mismatch: {entry['path']}")
        if path.name == "data.dat":
            description = json.loads(path.with_name("desc.json").read_text(encoding="utf-8"))
            expected = math.prod(description["shape"]) * 4
            if expected != path.stat().st_size:
                errors.append(f"Shape/float32 size mismatch: {entry['path']}")
        total += path.stat().st_size
    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        raise SystemExit(1)
    count = sum(entry["path"].endswith("/data.dat") for entry in manifest["files"])
    print(f"PASS: {len(manifest['files'])} files in {count} datasets; {total / 1024**3:.3f} GiB.")


if __name__ == "__main__":
    main()
