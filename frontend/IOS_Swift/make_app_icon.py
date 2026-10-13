"""Draw the app icon: an eye whose iris shows the road ahead.

Yellow, the color of tactile paving, on dark navy gives the highest contrast for low
vision. Writes the three appearances into the asset catalog: the default icon (opaque),
and dark and tinted versions with a transparent background, which the system fills.

Run from the repository root (requires numpy and opencv):
    python frontend/IOS_Swift/make_app_icon.py
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


ICON_SET = Path(__file__).parent / "Gilnun" / "Assets.xcassets" / "AppIcon.appiconset"
SIZE = 1024
SUPERSAMPLING = 4  # for the polygons; circles are antialiased by their distance fields

YELLOW = (0xFF, 0xD2, 0x00)
NAVY_TOP = (0x1A, 0x36, 0x52)
NAVY_BOTTOM = (0x0A, 0x18, 0x28)

# Geometry in pixels of the 1024 x 1024 icon; heights are relative to its center.
EYE_HALF_WIDTH = 420
EYE_HALF_HEIGHT = 285
EYE_STROKE = 64
IRIS_RADIUS = 185
VANISHING_Y = -95  # where the road's lines would meet
ROAD_BOTTOM_Y = 230
LINES_TOP_Y = -60  # the lines stop short of the vanishing point
EDGE_LINE = (110, 165)  # inner and outer edge of each side line at ROAD_BOTTOM_Y
CENTER_DASH_HALF_WIDTH = 20  # at ROAD_BOTTOM_Y
CENTER_DASHES = [(0.30, 0.43), (0.55, 0.78), (0.86, 1.10)]  # fractions of the way down from VANISHING_Y

CENTER = SIZE / 2
ys, xs = np.mgrid[0:SIZE, 0:SIZE].astype(np.float64) + 0.5


def circle(radius: float, cx: float = CENTER, cy: float = CENTER) -> np.ndarray:
    """Signed distance to a circle: negative inside."""

    return np.hypot(xs - cx, ys - cy) - radius


def eye(inset: float = 0.0) -> np.ndarray:
    """Signed distance to the eye: the overlap of two discs, shrunk by `inset`."""

    radius = (EYE_HALF_WIDTH**2 + EYE_HALF_HEIGHT**2) / (2 * EYE_HALF_HEIGHT)
    offset = radius - EYE_HALF_HEIGHT
    return np.maximum(circle(radius - inset, cy=CENTER + offset), circle(radius - inset, cy=CENTER - offset))


def coverage(distance: np.ndarray) -> np.ndarray:
    return np.clip(0.5 - distance, 0.0, 1.0)


def polygons(shapes: list[list[tuple[float, float]]]) -> np.ndarray:
    canvas = np.zeros((SIZE * SUPERSAMPLING, SIZE * SUPERSAMPLING), np.uint8)
    for shape in shapes:
        points = np.round(np.array(shape) * SUPERSAMPLING * 16).astype(np.int32)
        cv2.fillPoly(canvas, [points], 255, shift=4)
    return cv2.resize(canvas, (SIZE, SIZE), interpolation=cv2.INTER_AREA).astype(np.float64) / 255


def road_lines() -> list[list[tuple[float, float]]]:
    """Side lines and center dashes of a road that narrows toward the vanishing point."""

    vanishing, bottom, top = CENTER + VANISHING_Y, CENTER + ROAD_BOTTOM_Y, CENTER + LINES_TOP_Y

    def toward_vanishing(offset: float, y: float) -> float:
        """x at height y on the line from (center + offset, bottom) to the vanishing point."""
        return CENTER + offset * (y - vanishing) / (bottom - vanishing)

    shapes = []
    inner, outer = EDGE_LINE
    for side in (-1, 1):
        shapes.append([
            (toward_vanishing(side * inner, top), top),
            (toward_vanishing(side * outer, top), top),
            (CENTER + side * outer, bottom),
            (CENTER + side * inner, bottom),
        ])
    for start, end in CENTER_DASHES:
        upper, lower = vanishing + start * (bottom - vanishing), vanishing + end * (bottom - vanishing)
        shapes.append([
            (toward_vanishing(-CENTER_DASH_HALF_WIDTH, upper), upper),
            (toward_vanishing(CENTER_DASH_HALF_WIDTH, upper), upper),
            (toward_vanishing(CENTER_DASH_HALF_WIDTH, lower), lower),
            (toward_vanishing(-CENTER_DASH_HALF_WIDTH, lower), lower),
        ])
    return shapes


def artwork() -> np.ndarray:
    """Opacity of the yellow artwork: the eye's outline, and the iris minus the road's lines."""

    outline = coverage(np.maximum(eye(), -eye(EYE_STROKE)))
    iris = coverage(circle(IRIS_RADIUS))
    return np.maximum(outline, iris * (1 - polygons(road_lines())))


def main() -> None:
    alpha = artwork()[..., None]
    yellow = np.array(YELLOW, np.float64)
    fade = ys[..., None] / SIZE
    navy = np.array(NAVY_TOP) * (1 - fade) + np.array(NAVY_BOTTOM) * fade

    def save(name: str, rgb: np.ndarray, opacity: np.ndarray | None = None) -> None:
        pixels = rgb if opacity is None else np.concatenate([rgb, opacity * 255], axis=2)
        bgr = pixels[..., [2, 1, 0, 3][: pixels.shape[2]]]
        cv2.imwrite(str(ICON_SET / name), np.round(bgr).astype(np.uint8))
        print(f"Wrote {ICON_SET / name}")

    save("AppIcon.png", navy * (1 - alpha) + yellow * alpha)
    save("AppIcon-Dark.png", np.broadcast_to(yellow, alpha.shape[:2] + (3,)), alpha)
    save("AppIcon-Tinted.png", np.full(alpha.shape[:2] + (3,), 255.0), alpha)


if __name__ == "__main__":
    main()
