"""Corner polygon mask for VCE frames."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw


def make_corner_polygon_mask(size: int = 336) -> np.ndarray:
    """Return a boolean mask (H, W) that is True where the image is valid.

    VCE frames have dark corners from the capsule optics. This mask cuts
    them out using the same polygon as the original notebook.
    """
    img = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(img)

    c = size / 2
    r = size / 2
    # octagon approximation
    d = r * (1 - 1 / np.sqrt(2)) * 0.55
    polygon = [
        (d, 0), (size - d, 0),
        (size, d), (size, size - d),
        (size - d, size), (d, size),
        (0, size - d), (0, d),
    ]
    draw.polygon(polygon, fill=255)
    return np.array(img, dtype=bool)


def apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Zero out pixels outside the valid region."""
    out = image.copy()
    out[~mask] = 0
    return out
