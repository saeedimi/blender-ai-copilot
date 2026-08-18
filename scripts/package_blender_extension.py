#!/usr/bin/env python3
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "blender_extension"
DIST = ROOT / "dist"
OUTPUT = DIST / "blender_ai_copilot_extension.zip"

required = [SOURCE / "blender_manifest.toml", SOURCE / "__init__.py"]
missing = [str(p) for p in required if not p.exists()]
if missing:
    raise SystemExit("Missing required extension files: " + ", ".join(missing))

DIST.mkdir(exist_ok=True)

with zipfile.ZipFile(OUTPUT, "w", zipfile.ZIP_DEFLATED) as zf:
    for path in sorted(SOURCE.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            zf.write(path, arcname=str(path.relative_to(SOURCE)))

print(f"Created: {OUTPUT}")
