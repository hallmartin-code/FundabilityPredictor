"""Register the TEN Capital brand fonts with ReportLab.

Sora, Inter and JetBrains Mono are bundled as static TTFs under
``assets/fonts`` (see ``assets/make_fonts.py``). Google publishes them only as
variable fonts, and ReportLab renders a variable font at its default instance,
so every weight would come out Regular - the statics are pinned to real
weights instead.

If the files are missing, registration falls back to the built-in Helvetica
family. The report still renders correctly, just without the brand typeface,
which is the right failure mode for a deployment that lost its assets.
"""

from __future__ import annotations

import logging
from pathlib import Path

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from config.settings import ASSETS_DIR

LOGGER = logging.getLogger(__name__)

FONT_DIR: Path = ASSETS_DIR / "fonts"

#: Registered name -> filename.
BRAND_FONTS: dict[str, str] = {
    "Sora-Bold": "Sora-Bold.ttf",
    "Sora-ExtraBold": "Sora-ExtraBold.ttf",
    "Inter": "Inter-Regular.ttf",
    "Inter-SemiBold": "Inter-SemiBold.ttf",
    "JetBrainsMono": "JetBrainsMono-Regular.ttf",
    "JetBrainsMono-Medium": "JetBrainsMono-Medium.ttf",
}

#: What each brand font degrades to when its file is unavailable.
FALLBACKS: dict[str, str] = {
    "Sora-Bold": "Helvetica-Bold",
    "Sora-ExtraBold": "Helvetica-Bold",
    "Inter": "Helvetica",
    "Inter-SemiBold": "Helvetica-Bold",
    "JetBrainsMono": "Helvetica",
    "JetBrainsMono-Medium": "Helvetica-Bold",
}

_resolved: dict[str, str] | None = None


def register_fonts() -> dict[str, str]:
    """Register the brand fonts once and return name -> usable font name."""
    global _resolved
    if _resolved is not None:
        return _resolved

    resolved: dict[str, str] = {}
    missing: list[str] = []

    for name, filename in BRAND_FONTS.items():
        path = FONT_DIR / filename
        if not path.exists():
            missing.append(filename)
            resolved[name] = FALLBACKS[name]
            continue
        try:
            pdfmetrics.registerFont(TTFont(name, str(path)))
            resolved[name] = name
        except Exception:  # noqa: BLE001 - a bad font file must not stop a report
            LOGGER.exception("Could not register %s; falling back to Helvetica.", filename)
            resolved[name] = FALLBACKS[name]

    if missing:
        LOGGER.warning(
            "Brand fonts not found in %s (%s). Rendering with Helvetica instead; "
            "run assets/make_fonts.py to rebuild them.",
            FONT_DIR, ", ".join(missing),
        )
    else:
        # Lets ReportLab resolve <b>/<i> inside Paragraph markup.
        try:
            pdfmetrics.registerFontFamily(
                "Inter", normal="Inter", bold="Inter-SemiBold",
                italic="Inter", boldItalic="Inter-SemiBold",
            )
        except Exception:  # noqa: BLE001
            LOGGER.debug("Could not register the Inter family mapping.", exc_info=True)

    _resolved = resolved
    return resolved


def brand_font(name: str) -> str:
    """The usable font name for a brand font, registering on first use."""
    return register_fonts().get(name, FALLBACKS.get(name, "Helvetica"))


def using_brand_fonts() -> bool:
    """True when the real brand typefaces are in play, not the fallback."""
    return register_fonts().get("Inter") == "Inter"
