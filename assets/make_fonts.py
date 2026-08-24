"""Build the static brand fonts bundled with the PDF renderer.

    python assets/make_fonts.py

Google publishes Sora, Inter and JetBrains Mono only as *variable* fonts, and
ReportLab renders a variable font at its default instance - so every weight
would come out Regular and the design would lose its typographic hierarchy
entirely. fontTools pins each weight axis to a real value instead, producing
the static TTFs that ``reports/fonts.py`` registers.

All three families are SIL Open Font License, so redistributing the generated
statics with this application is permitted. The generated files are committed
so a deployment never depends on network access at build time.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

try:
    from fontTools.ttLib import TTFont
    from fontTools.varLib import instancer
except ImportError:  # pragma: no cover - build-time tool only
    print("fontTools is required: pip install fonttools", file=sys.stderr)
    raise SystemExit(1)

ASSETS = Path(__file__).resolve().parent
OUT = ASSETS / "fonts"
BASE = "https://raw.githubusercontent.com/google/fonts/main/ofl"

#: (source path, cache name, [(weight, output filename)])
JOBS: list[tuple[str, str, list[tuple[int, str]]]] = [
    (
        "sora/Sora%5Bwght%5D.ttf", "Sora-var.ttf",
        [(700, "Sora-Bold.ttf"), (800, "Sora-ExtraBold.ttf")],
    ),
    (
        "inter/Inter%5Bopsz,wght%5D.ttf", "Inter-var.ttf",
        [(400, "Inter-Regular.ttf"), (600, "Inter-SemiBold.ttf")],
    ),
    (
        "jetbrainsmono/JetBrainsMono%5Bwght%5D.ttf", "JBMono-var.ttf",
        [(400, "JetBrainsMono-Regular.ttf"), (500, "JetBrainsMono-Medium.ttf")],
    ),
]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / ".var-cache"
    cache.mkdir(exist_ok=True)

    for remote, cached, weights in JOBS:
        source = cache / cached
        if not source.exists():
            print(f"  downloading {cached} ...")
            urllib.request.urlretrieve(f"{BASE}/{remote}", source)

        for weight, filename in weights:
            font = TTFont(source)
            axes = {axis.axisTag: axis for axis in font["fvar"].axes}
            pins: dict[str, float] = {"wght": weight}
            # Inter also carries an optical-size axis; pin it to its default
            # so the instance is fully static.
            if "opsz" in axes:
                pins["opsz"] = axes["opsz"].defaultValue

            static = instancer.instantiateVariableFont(
                font, pins, inplace=False, updateFontNames=True
            )
            dest = OUT / filename
            static.save(dest)
            print(f"  {filename:30s} wght={weight}  {dest.stat().st_size:>8,} bytes")

    print(f"\nWrote {len(list(OUT.glob('*.ttf')))} static fonts to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
