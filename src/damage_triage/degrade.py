"""Photo degradations for the robustness test: what claim photos look like in real life.

Each function takes and returns a PIL image of the same size, and is deterministic, so the
degraded test set is identical on every run. They are applied to the held-out test photos only;
the model never trains on them, so the drop in score shows how it copes with worse photos.
"""

from __future__ import annotations

import io
from collections.abc import Callable

from PIL import Image, ImageEnhance, ImageFilter


def blur(image: Image.Image) -> Image.Image:
    """Out-of-focus or shaky phone shot."""
    return image.filter(ImageFilter.GaussianBlur(radius=max(1.0, min(image.size) / 160)))


def low_light(image: Image.Image) -> Image.Image:
    """Evening or underground car park: darker, flatter contrast."""
    dark = ImageEnhance.Brightness(image).enhance(0.45)
    return ImageEnhance.Contrast(dark).enhance(0.75)


def jpeg(image: Image.Image, quality: int = 15) -> Image.Image:
    """Heavily compressed photo, e.g. forwarded through a messaging app."""
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def low_res(image: Image.Image, factor: float = 0.25) -> Image.Image:
    """Small thumbnail scaled back up."""
    w, h = image.size
    small = image.resize((max(1, int(w * factor)), max(1, int(h * factor))), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


DEGRADATIONS: dict[str, Callable[[Image.Image], Image.Image]] = {
    "blur": blur,
    "low_light": low_light,
    "jpeg": jpeg,
    "low_res": low_res,
}
