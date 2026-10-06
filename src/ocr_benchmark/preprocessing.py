from __future__ import annotations

from pathlib import Path
from typing import Literal

import cv2
import numpy as np


PreprocessMode = Literal["original", "enhance", "binary"]
PREPROCESS_MODES: tuple[PreprocessMode, ...] = ("original", "enhance", "binary")


def preprocess_image(
    image_path: str | Path, mode: PreprocessMode = "original"
) -> np.ndarray:
    """Load an image and apply one small, reproducible preprocessing variant."""

    path = Path(image_path)
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to read image: {path}")

    if mode == "original":
        return image

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

    if mode == "enhance":
        return enhanced
    if mode == "binary":
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
        return cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )[1]

    raise ValueError(f"Unknown preprocessing mode: {mode}")


def save_preprocessed(
    image_path: str | Path,
    output_path: str | Path,
    mode: PreprocessMode,
) -> Path:
    """Write a preprocessing variant for adapters that require file paths."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), preprocess_image(image_path, mode)):
        raise OSError(f"Unable to write preprocessed image: {output}")
    return output
