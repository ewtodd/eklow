#!/usr/bin/env python3
"""Makes the --image-rows-dir of eklow-server from llama.cpp's own image rows: the test source of rows until the server's encoder
is used. Takes a run of eklow-oracle --mmproj (DIR/tokens.json lists its images, DIR/image_K.f32 holds their rows) and the
pictures that run was given, in order, and files each one's rows under the FNV-1a hash of the picture's bytes, as the server
looks them up.

  make_rows_dir.py ROWS_DIR ORACLE_DIR PICTURE [PICTURE ...]
"""
import json
import shutil
import sys
from pathlib import Path


def fnv1a(data: bytes) -> int:
    h = 0xCBF29CE484222325
    for b in data:
        h = ((h ^ b) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def main():
    if len(sys.argv) < 4:
        raise SystemExit(__doc__)
    out, d = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    images = json.load(open(d / "tokens.json"))["images"]
    if len(images) != len(sys.argv) - 3:
        raise SystemExit(f"{d} has {len(images)} images, {len(sys.argv) - 3} pictures given")
    for k, (pic, im) in enumerate(zip(sys.argv[3:], images)):
        name = f"{fnv1a(Path(pic).read_bytes()):016x}"
        shutil.copy(d / f"image_{k}.f32", out / f"{name}.f32")
        (out / f"{name}.grid").write_text(f"{im['nx']} {im['ny']}\n")
        print(f"{pic}: {name} {im['nx']}x{im['ny']} = {im['n']} rows")


main()
