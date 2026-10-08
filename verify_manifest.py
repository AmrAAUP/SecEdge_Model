"""Verify this downloaded evidence tree without running any experiment."""
import hashlib
import json
from pathlib import Path
import sys

root = Path(__file__).resolve().parent
manifest = json.loads((root / "MANIFEST_SHA256.json").read_text(encoding="utf-8"))
errors = []
for record in manifest["files"]:
    path = root / record["path"]
    resolved = path.resolve()
    if not resolved.is_relative_to(root) or not path.is_file():
        errors.append(record["path"] + ": missing or invalid path")
        continue
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != record["sha256"] or path.stat().st_size != record["bytes"]:
        errors.append(record["path"] + ": byte count or SHA-256 mismatch")
if errors:
    print("FAIL\n" + "\n".join(errors))
    sys.exit(1)
print("PASS: " + str(len(manifest["files"])) + " published files match their recorded sizes and SHA-256 hashes.")
print("The manifest excludes itself. Hashes verify identity, not historical timing or scientific validity.")
