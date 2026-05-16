"""Render report text into a page image.

ColPali was trained on *document-page screenshots*, not raw photos. To use
it as a retriever over our text reports we render each report as a clean
page image - which is the input modality ColPali actually expects. This
is a key design point in the model comparison: ColPali needs this
rendering step, BiomedCLIP does not.
"""

from __future__ import annotations

import textwrap

from PIL import Image, ImageDraw, ImageFont

# A4-ish portrait canvas at a modest DPI - enough detail for ColPali patches.
_PAGE_SIZE = (1240, 1754)
_MARGIN = 80
_LINE_HEIGHT = 34
_WRAP_WIDTH = 95


def _load_font(size: int = 26) -> ImageFont.ImageFont:
    """Best-effort load of a readable font, falling back to PIL's default."""
    for path in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",  # macOS
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_report_page(report_text: str, title: str = "RADIOLOGY REPORT") -> Image.Image:
    """Render a radiology report as a white page image for ColPali indexing."""
    page = Image.new("RGB", _PAGE_SIZE, "white")
    draw = ImageDraw.Draw(page)

    title_font = _load_font(34)
    body_font = _load_font(26)

    draw.text((_MARGIN, _MARGIN), title, fill="black", font=title_font)
    y = _MARGIN + 70

    for paragraph in report_text.splitlines() or [report_text]:
        for line in textwrap.wrap(paragraph, width=_WRAP_WIDTH) or [""]:
            draw.text((_MARGIN, y), line, fill="black", font=body_font)
            y += _LINE_HEIGHT
            if y > _PAGE_SIZE[1] - _MARGIN:  # stop at the page boundary
                return page
    return page
