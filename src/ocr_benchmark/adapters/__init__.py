from .base import OCRAdapter
from .easyocr_adapter import EasyOCRAdapter
from .paddleocr_vl_adapter import PaddleOCRVLAdapter
from .tesseract_adapter import TesseractAdapter

__all__ = [
    "OCRAdapter",
    "EasyOCRAdapter",
    "PaddleOCRVLAdapter",
    "TesseractAdapter",
]
