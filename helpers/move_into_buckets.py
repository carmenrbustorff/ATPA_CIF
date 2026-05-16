from pathlib import Path
import re

ROOT = Path("experiments")
BUCKET = 50

for p in sorted(ROOT.iterdir()):
    if not p.is_dir(): 
        continue
    m = re.match(r"iter_(\d+)_", p.name)
    if not m:
        continue
    num = int(m.group(1))
    bstart = (num // BUCKET) * BUCKET + 1
    bend = (num // BUCKET + 1) * BUCKET
    bucket = ROOT / f"iterations_{bstart:04d}-{bend:04d}"
    bucket.mkdir(parents=True, exist_ok=True)
    dest = bucket / p.name
    if dest.exists():
        print("Exists, skipping:", dest)
        continue
    p.rename(dest)
    print("Moved", p, "->", dest)