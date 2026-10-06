"""Tools for benchmarking multilingual OCR engines."""

from .schema import OCRBlock, OCRResult
from .metrics import evaluate_text

__all__ = ["OCRBlock", "OCRResult", "evaluate_text"]
