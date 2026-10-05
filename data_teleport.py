"""
Data teleport - one-time helper: copies the Warwick Capacity_Check csv files
from the downloaded nested folders into one flat folder data/WARWICK_RAW.
Needed before warwick_explorer.py and warwick_extractor.py (section 3.1.1).

If two files have the same name, the folder names are added in front
(or _2, _3 ... at the end), so nothing gets overwritten.

SOURCE_DIR is a path on my laptop - change it on another computer.
Run once: python data_teleport.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

SOURCE_DIR = Path(r"PATH/TO/WARWICK_RAW/DIB_Data/.csvfiles/Capacity_Check")  # set before running


def _flatten_unique_name(src: Path, source_root: Path, dest_dir: Path, used: set[str]) -> Path:
    # where the downloaded Warwick files are (path on my machine)
    rel = src.relative_to(source_root)
    base = rel.name
    # 1) plain file name if it is free
    candidate = dest_dir / base
    if base not in used and not candidate.exists():
        return candidate
    # 2) put the folder names in front of the file name
    stem = rel.stem
    suffix = rel.suffix
    parts = rel.parent.parts
    if parts:
        prefixed = "_".join(parts) + "_" + base
        candidate = dest_dir / prefixed
        if prefixed not in used and not candidate.exists():
            return candidate
    # 3) add _2, _3 ... at the end until the name is free
    n = 2
    while True:
        alt = f"{stem}_{n}{suffix}"
        candidate = dest_dir / alt
        if alt not in used and not candidate.exists():
            return candidate
        n += 1


def main() -> None:
    root = Path(__file__).resolve().parent
    dest_dir = root / "data" / "WARWICK_RAW"

    if not SOURCE_DIR.is_dir():
        print(f"Source directory not found: {SOURCE_DIR}", file=sys.stderr)
        sys.exit(1)

    dest_dir.mkdir(parents=True, exist_ok=True)
    #Rglob also looks inside all subfolders
    csv_files = sorted(SOURCE_DIR.rglob("*.csv"))
    used_names: set[str] = set()
    copied: list[Path] = []

    for src in csv_files:
        dst = _flatten_unique_name(src, SOURCE_DIR, dest_dir, used_names)
        shutil.copy2(src, dst)
        used_names.add(dst.name)
        copied.append(dst)

    print(f"Destination: {dest_dir.resolve()}")
    print(f"Total CSV files copied: {len(copied)}")
    print("Files:")
    for p in copied:
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
