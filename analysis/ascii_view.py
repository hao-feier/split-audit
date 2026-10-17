"""ASCII luminance preview of a PNG (for text-only QA of figures).

Usage:  python3 analysis/ascii_view.py <png> [width]
Prints a coarse brightness map so the figure can be sanity-checked
layout (panels present, bars/curves/labels positioned, nothing blank).
"""
import sys
from pathlib import Path
from PIL import Image

RAMP = " .:-=+*#%@"   # light -> dark


def view(path, width=72):
    im = Image.open(path).convert("L")
    w, h = im.size
    height = max(1, int(width * (h / w) * 0.5))   # chars are ~2x tall
    im = im.resize((width, height))
    px = im.getdata()
    mn, mx = 0, 255
    lines = []
    for y in range(height):
        row = []
        for x in range(width):
            v = px[y * width + x]
            idx = int((mx - (v - mn)) / (mx - mn + 1e-9) * (len(RAMP) - 1))
            row.append(RAMP[idx])
        lines.append("".join(row))
    print(f"### {Path(path).name}  ({w}x{h}px)  ASCII {width}x{height}")
    print("\n".join(lines))
    print()


if __name__ == "__main__":
    for p in sys.argv[1:]:
        if p.isdigit():
            continue
        view(p, int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 72)
