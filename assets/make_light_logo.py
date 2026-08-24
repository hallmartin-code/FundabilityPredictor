"""Generate the light-on-dark variant of the TEN Capital footer logo.

The supplied mark sets the "TEN CAPITAL NETWORK" wordmark in near-black ink,
which disappears against the navy report surface. This lifts that ink to the
brand's --ink-100 while leaving the three coloured figures (coral, amber,
teal) exactly as they are.

    python assets/make_light_logo.py
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

ASSETS = Path(__file__).resolve().parent
SOURCE = ASSETS / "TEN_Capital_logo_footer.png"
DEST = ASSETS / "TEN_Capital_logo_footer_light.png"

#: Anything darker than this on every channel is treated as ink, not artwork.
INK_CEILING = 110
#: --ink-100 from the brand palette.
INK_LIGHT = (243, 246, 250)


def main() -> int:
    source = Image.open(SOURCE).convert("RGBA")
    width, height = source.size
    access = source.load()
    pixels = [access[x, y] for y in range(height) for x in range(width)]
    lifted = 0
    out: list[tuple[int, int, int, int]] = []

    for r, g, b, a in pixels:
        if a and max(r, g, b) < INK_CEILING:
            out.append((*INK_LIGHT, a))
            lifted += 1
        else:
            out.append((r, g, b, a))

    result = Image.new("RGBA", source.size)
    result.putdata(out)
    result.save(DEST)

    print(f"  lifted {lifted:,} ink pixels of {len(pixels):,}")
    print(f"  wrote {DEST.name} ({DEST.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
