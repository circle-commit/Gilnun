"""Draws tactile paving into sidewalk photos for the Android tactile-paving tests.

No real photo of Korean tactile paving was available, so these paint a perspective strip
of yellow tiles (bars or domes, lit from the top left) into repository photos. They test
the geometry and the texture rule, not real-world color and lighting.

Run from the repository root (numpy and Pillow):
    python frontend/Android/tests/fixtures/make_tactile_fixtures.py
"""

from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent


def portrait(path: Path, width: int = 360) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    crop_width = round(image.height * 9 / 16)
    left = (image.width - crop_width) // 2
    image = image.crop((left, 0, left + crop_width, image.height)).resize((width, round(width * 16 / 9)), Image.BILINEAR)
    return np.asarray(image).astype(float)


def paving(base, center_bottom, center_top, kind, color=(232, 188, 35), half_width=0.17, seed=0):
    """A strip from `center_bottom` at the feet to `center_top` mid-frame (fractions of the width)."""

    out = base.copy()
    height, width, _ = out.shape
    rng = np.random.default_rng(seed)
    horizon, top = int(height * 0.40), int(height * 0.50)
    for y in range(top, height):
        t = (y - horizon) / (height - horizon)
        half = width * half_width * t
        center = width * (center_top + (center_bottom - center_top) * (y - top) / (height - top))
        x0, x1 = int(max(0, center - half)), int(min(width, center + half))
        v = 6 / t
        for x in range(x0, x1):
            u = (x - (center - half)) / (2 * half) * 5
            fu, fv = u % 1 - 0.5, v % 1 - 0.5
            pixel = np.array(color, float) * (0.92 + 0.08 * rng.random())
            if kind == "dots" and np.hypot(fu, fv) < 0.32:
                pixel *= 1.15 - 0.6 * (fu + fv)
            elif kind == "bars" and abs(fu) < 0.22:
                pixel *= 1.12 - 0.9 * fu
            elif kind == "flat":
                pixel = np.array(color, float)
            out[y, x] = np.clip(pixel, 0, 255)
    return out


def save(array, name):
    # JPEG, like camera frames: its compression noise is part of what the texture rule must ignore.
    Image.fromarray(array.astype(np.uint8)).save(OUT / name, quality=90)


if __name__ == "__main__":
    plaza = portrait(REPO_ROOT / "test_images/Bbox_0240_MP_SEL_042078.jpg")
    street = portrait(REPO_ROOT / "datasets/yolo_sidewalk/images/train/Bbox_0001_MP_SEL_000001.jpg")
    save(paving(plaza, 0.5, 0.5, "bars"), "tactile_straight.jpg")
    save(paving(street, 0.45, 0.8, "dots"), "tactile_turn_right.jpg")
    save(paving(plaza, 0.2, 0.2, "bars", seed=1), "tactile_left.jpg")
    # Smooth yellow paint of the same color and shape must not count as paving.
    save(paving(plaza, 0.5, 0.5, "flat"), "yellow_paint_strip.jpg")
    # A wide yellow area is not a path to follow.
    save(paving(plaza, 0.5, 0.5, "dots", half_width=0.6), "yellow_wide_area.jpg")
    print("fixtures written to", OUT)
